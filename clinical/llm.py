"""SOAP-note generation with Qwen: chat template, long-transcript chunking
(map-reduce) and parsing of the model output into sections."""
import re

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SECTIONS = ["Chief Complaint", "Subjective", "Objective", "Assessment", "Plan", "Medications"]
NOT_DISCUSSED = "Not discussed"

SYSTEM_PROMPT = """You are a clinical documentation assistant. You write SOAP notes from doctor-patient conversations.

Rules:
- Use ONLY information stated in the conversation. Never invent findings, diagnoses, doses or tests.
- Write short bullet points that start with "- ". No paragraphs and no extra commentary.
- If a section has no information, write "- Not discussed".
- Use exactly these section headers, in this order:

Chief Complaint:
Subjective:
Objective:
Assessment:
Plan:
Medications:

What goes where:
- Chief Complaint: the main reason for the visit, in a few words.
- Subjective: symptoms, their duration and severity, past medical history, allergies and habits the patient reports.
- Objective: vital signs, examination findings and test results that are mentioned.
- Assessment: the diagnosis or impression the doctor states. Do not make your own diagnosis.
- Plan: tests ordered, advice given, referrals and follow-up.
- Medications: one line per drug in the form: name | dose | frequency | duration. Write "not stated" for any missing part."""

MERGE_PROMPT = """These are partial SOAP notes written from consecutive parts of ONE consultation.
Merge them into a single SOAP note with the same section headers and rules.
Remove duplicate points. If parts conflict, prefer the later part for Assessment, Plan and Medications.

{partials}"""


# ----------------------------------------------------------------- loading

def load_llm(model_name=DEFAULT_MODEL, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.to(device).eval()
    return tokenizer, model


# -------------------------------------------------------------- generation

def generate(tokenizer, model, messages, max_new_tokens=512):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic output
            repetition_penalty=1.1,   # mild: medical terms legitimately repeat
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _note_messages(conversation, part=None):
    header = f"Conversation (part {part[0]} of {part[1]}):" if part else "Conversation:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}\n{conversation}\n\nWrite the SOAP note."},
    ]


def chunk_lines(tokenizer, lines, max_tokens=1800):
    """Group transcript lines into chunks under a token budget, never
    splitting a line, so no part of the conversation is silently dropped."""
    chunks, current, size = [], [], 0
    for line in lines:
        n = len(tokenizer.encode(line, add_special_tokens=False))
        if current and size + n > max_tokens:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        chunks.append(current)
    return chunks


def summarize(tokenizer, model, lines, max_chunk_tokens=1800, max_new_tokens=512, progress=None):
    """Return (note_text, partial_notes). Long conversations are summarized
    chunk by chunk, then merged."""
    chunks = chunk_lines(tokenizer, lines, max_chunk_tokens)
    if len(chunks) <= 1:
        if progress:
            progress("Writing the note…")
        return generate(tokenizer, model, _note_messages("\n".join(lines)), max_new_tokens), []

    partials = []
    for i, chunk in enumerate(chunks, 1):
        if progress:
            progress(f"Summarizing part {i} of {len(chunks)}…")
        partials.append(generate(tokenizer, model, _note_messages("\n".join(chunk), (i, len(chunks))), max_new_tokens))

    if progress:
        progress("Merging parts into one note…")
    joined = "\n\n".join(f"Part {i}:\n{p}" for i, p in enumerate(partials, 1))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MERGE_PROMPT.format(partials=joined)},
    ]
    return generate(tokenizer, model, messages, max_new_tokens), partials


# ----------------------------------------------------------------- parsing

_HEADER_RE = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(chief complaints?|subjective|objective|assessment|plan|medications?)"
    r"\s*\**\s*(?::\s*\**\s*(.*)|$)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_EMPTY = {"none", "n/a", "na", "not discussed", "not mentioned", "not stated", "nil"}


def _canonical(header):
    h = header.lower()
    if h.startswith("chief"):
        return "Chief Complaint"
    if h.startswith("medication"):
        return "Medications"
    return header.capitalize()


def is_empty_item(item):
    return item.strip().strip(".").lower() in _EMPTY


def parse_note(text):
    """Split model output into {section: [bullet, ...]} for all six sections."""
    note = {s: [] for s in SECTIONS}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEADER_RE.match(line)
        if m:
            current = _canonical(m.group(1))
            inline = (m.group(2) or "").strip().strip("*").strip()
            if inline:
                note[current].append(_BULLET_RE.sub("", inline))
            continue
        if current:
            item = _BULLET_RE.sub("", line).strip().strip("*").strip()
            if item:
                note[current].append(item)
    for s in SECTIONS:
        items = [i for i in note[s] if not is_empty_item(i)]
        note[s] = items or [NOT_DISCUSSED]
    return note


def parse_medications(items):
    meds = []
    for item in items:
        if is_empty_item(item):
            continue
        parts = [p.strip() for p in item.split("|")]
        parts += ["not stated"] * (4 - len(parts))
        meds.append({"name": parts[0], "dose": parts[1], "frequency": parts[2], "duration": parts[3]})
    return meds
