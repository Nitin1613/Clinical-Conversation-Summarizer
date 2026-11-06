"""Rule-based removal of personal identifiers before text is stored or exported.

This is a safety net, not a guarantee: it catches common patterns (phones,
emails, Aadhaar-style numbers, titled or self-introduced names). Review
exports before sharing them.
"""
import re

_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"

PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ID_NUMBER", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),  # Aadhaar-style
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
]

# Keep the title / phrase, replace only the name that follows it.
NAME_PATTERNS = [
    re.compile(r"\b((?:Mr|Mrs|Ms|Miss|Dr|Shri|Smt)\.?\s+)(" + _NAME + r")"),
    re.compile(r"\b((?:[Mm]y name is|[Tt]his is|[Cc]all me)\s+)(" + _NAME + r")"),
]


def redact(text: str):
    """Return (redacted_text, {category: count})."""
    counts = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    for pattern in NAME_PATTERNS:
        text, n = pattern.subn(lambda m: m.group(1) + "[NAME]", text)
        if n:
            counts["NAME"] = counts.get("NAME", 0) + n
    return text, counts
