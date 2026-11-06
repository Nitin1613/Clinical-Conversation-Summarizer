"""Export the note as Markdown, PDF, or a FHIR-style JSON bundle."""
import html
import json
import uuid
from datetime import datetime, timezone

from .llm import SECTIONS

STATUS_TAG = {"supported": "OK", "weak": "CHECK", "unsupported": "UNSUPPORTED", "n/a": ""}
DISCLAIMER = "AI-generated draft. A clinician must review and confirm every point before use."

# LOINC section codes (FHIR-style, not validated against a profile)
LOINC = {
    "Chief Complaint": ("10154-3", "Chief complaint"),
    "Subjective": ("10164-2", "History of present illness"),
    "Objective": ("29545-1", "Physical findings"),
    "Assessment": ("51848-0", "Assessment"),
    "Plan": ("18776-5", "Plan of care"),
    "Medications": ("10160-0", "History of medication use"),
}


def _now():
    return datetime.now(timezone.utc).astimezone()


def _tagged(check):
    tag = STATUS_TAG.get(check["status"], "")
    line = f"[{tag}] {check['text']}" if tag else check["text"]
    if check["issues"]:
        line += f"  ({'; '.join(check['issues'])})"
    return line


def to_markdown(checks, transcript_lines):
    out = [f"# Clinical Note", f"_{DISCLAIMER}_", f"Generated: {_now():%Y-%m-%d %H:%M}", ""]
    for s in SECTIONS:
        out.append(f"## {s}")
        out += [f"- {_tagged(c)}" for c in checks.get(s, [])]
        out.append("")
    out.append("## Transcript")
    out += [f"{i}. {line}" for i, line in enumerate(transcript_lines, 1)]
    return "\n".join(out) + "\n"


def to_pdf(checks, transcript_lines):
    from fpdf import FPDF

    def safe(t):  # core PDF fonts are Latin-1 only
        return t.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    def para(text, size=10, style="", gap=1):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 5 + size * 0.1, safe(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    para("Clinical Note", 16, "B", 2)
    para(DISCLAIMER, 9, "I")
    para(f"Generated: {_now():%Y-%m-%d %H:%M}", 9, "", 4)
    for s in SECTIONS:
        para(s, 12, "B")
        for c in checks.get(s, []):
            para("- " + _tagged(c), 10, "", 0)
        pdf.ln(3)
    pdf.add_page()
    para("Transcript", 12, "B")
    for i, line in enumerate(transcript_lines, 1):
        para(f"{i}. {line}", 9, "", 0)
    return bytes(pdf.output())


def to_fhir_bundle(checks, medications):
    """A FHIR R4-style document Bundle: one Composition with SOAP sections and
    one MedicationStatement per drug. Meant as an interchange starting point."""
    comp_id = str(uuid.uuid4())
    now = _now().isoformat(timespec="seconds")

    sections = []
    for s in SECTIONS:
        code, display = LOINC[s]
        items = "".join(f"<li>{html.escape(c['text'])}</li>" for c in checks.get(s, []))
        sections.append({
            "title": s,
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
            "text": {"status": "generated", "div": f'<div xmlns="http://www.w3.org/1999/xhtml"><ul>{items}</ul></div>'},
        })

    entries = [{
        "fullUrl": f"urn:uuid:{comp_id}",
        "resource": {
            "resourceType": "Composition",
            "status": "preliminary",
            "type": {"coding": [{"system": "http://loinc.org", "code": "11488-4", "display": "Consult note"}]},
            "date": now,
            "author": [{"display": "AI draft - pending clinician review"}],
            "title": "Clinical Conversation Summary",
            "section": sections,
        },
    }]
    for m in medications:
        dosage = ", ".join(v for v in (m["dose"], m["frequency"], m["duration"]) if v and v != "not stated")
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "MedicationStatement",
                "status": "unknown",
                "medicationCodeableConcept": {"text": m["name"]},
                "dosage": [{"text": dosage or "not stated"}],
            },
        })
    bundle = {"resourceType": "Bundle", "type": "document", "timestamp": now, "entry": entries}
    return json.dumps(bundle, indent=2)
