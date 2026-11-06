"""Check every bullet of the generated note against the transcript.

Each bullet gets:
  - the transcript lines that best support it (evidence),
  - a support score (word overlap, plus semantic similarity when an
    embedding model is available),
  - hard checks: every number (doses, durations, vitals) and every drug
    name in the bullet must actually appear in the transcript.

Known limit: this measures support, not meaning. It will not catch a
negation flip ("no fever" written as "fever").
"""
import difflib
import re

from .llm import NOT_DISCUSSED, is_empty_item

SUPPORTED, WEAK, UNSUPPORTED, NA = "supported", "weak", "unsupported", "n/a"

STOPWORDS = set("""
a an the and or but of to in on at for with by from as is are was were be been being
it its this that these those i you he she we they me him her us them my your his our their
do does did have has had not no yes so if then than there here very also just about into
over under again some any all can could would should will shall may might must
patient patients reports reported report complains complaint states stated mentions
mentioned presents presented history doctor notes noted denies denied describes
""".split())

_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_SMALL.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
               "seventy": 70, "eighty": 80, "ninety": 90})
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _stem(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def content_tokens(text):
    return {_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def numbers_in(text):
    """Digits plus spoken numbers ("five hundred" -> 500)."""
    values = {float(n) for n in _NUM_RE.findall(text)}
    total, cur = 0, None
    for w in re.findall(r"[a-z]+|\d+", text.lower()):
        if w in _SMALL:
            cur = (cur or 0) + _SMALL[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        elif w == "and" and (cur or total):
            continue
        else:
            if cur is not None or total:
                values.add(float(total + (cur or 0)))
            total, cur = 0, None
    if cur is not None or total:
        values.add(float(total + (cur or 0)))
    return values


class Grounder:
    def __init__(self, lines, embedder=None, supported_at=0.5, weak_at=0.3):
        self.lines = lines
        self.embedder = embedder
        self.supported_at, self.weak_at = supported_at, weak_at

        # Windows of 1 and 2 consecutive lines, so a fact split across a
        # question and its answer ("How long?" / "Three days") still matches.
        self.windows = [(i, i) for i in range(len(lines))]
        self.windows += [(i, i + 1) for i in range(len(lines) - 1)]
        self.window_text = [" ".join(lines[a:b + 1]) for a, b in self.windows]
        self.window_tokens = [content_tokens(t) for t in self.window_text]

        full = " ".join(lines)
        self.transcript_numbers = numbers_in(full)
        self.vocab = sorted(set(re.findall(r"[a-z]+", full.lower())))

        self.window_emb = None
        if embedder is not None and self.window_text:
            self.window_emb = embedder.encode(self.window_text, convert_to_tensor=True, normalize_embeddings=True)

    # ------------------------------------------------------------------
    def _scores(self, claim):
        tokens = content_tokens(claim)
        if not tokens:
            return None
        lex = [len(tokens & wt) / len(tokens) for wt in self.window_tokens]
        if self.window_emb is None:
            return lex
        from sentence_transformers import util
        q = self.embedder.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
        cos = util.cos_sim(q, self.window_emb)[0].tolist()
        return [0.6 * l + 0.4 * max(c, 0.0) for l, c in zip(lex, cos)]

    def _missing_numbers(self, claim):
        return sorted({n for n in numbers_in(claim) if n not in self.transcript_numbers})

    def _drug_found(self, name):
        words = [w for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 3]
        if not words:
            return True
        return any(difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.8) for w in words)

    # ------------------------------------------------------------------
    def check(self, claim, drug_name=None):
        if is_empty_item(claim) or claim == NOT_DISCUSSED:
            return {"text": claim, "status": NA, "score": None, "evidence": [], "issues": []}

        scores = self._scores(claim)
        if not scores:
            return {"text": claim, "status": WEAK, "score": 0.0, "evidence": [], "issues": ["nothing to verify"]}

        best = max(range(len(scores)), key=scores.__getitem__)
        score = scores[best]
        a, b = self.windows[best]
        evidence = self.lines[a:b + 1] if score > 0 else []

        issues = []
        missing = self._missing_numbers(claim)
        if missing:
            issues.append("number not in transcript: " + ", ".join(f"{n:g}" for n in missing))
        if drug_name and drug_name.lower() != "not stated" and not self._drug_found(drug_name):
            issues.append(f"drug '{drug_name}' not in transcript")

        if issues or score < self.weak_at:
            status = UNSUPPORTED
        elif score < self.supported_at:
            status = WEAK
        else:
            status = SUPPORTED
        return {"text": claim, "status": status, "score": round(score, 2), "evidence": evidence, "issues": issues}

    def check_note(self, note):
        """Return {section: [check, ...]} in the same order as the note.
        Medication lines also get a drug-name check."""
        results = {}
        for section, items in note.items():
            out = []
            for item in items:
                drug = item.split("|")[0].strip() if section == "Medications" else None
                out.append(self.check(item.replace("|", " "), drug_name=drug))
            results[section] = out
        return results


def support_rate(checks):
    """Share of checkable bullets that are fully supported."""
    flat = [c for items in checks.values() for c in items if c["status"] != NA]
    if not flat:
        return None
    return sum(c["status"] == SUPPORTED for c in flat) / len(flat)
