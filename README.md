<<<<<<< HEAD
# Clinical Conversation Summarizer

Record or upload a doctor-patient conversation and get a speaker-labelled transcript plus a SOAP note in which every point is checked against what was actually said.

```
audio ─► Whisper ─► pyannote (who spoke) ─► redaction ─► editable transcript
      ─► Qwen 2.5 (chat template, chunked for long visits) ─► SOAP note
      ─► grounding check per bullet ─► Markdown / PDF / FHIR-style JSON
```

## What changed from v1

| v1 | v2 |
|---|---|
| Raw prompt to an instruct model | Proper chat template (system + user) |
| `max_new_tokens=100`, summary got cut off | 512 tokens |
| Input truncated at 1024 tokens | Long visits split into chunks, summarized, then merged |
| One block of text, no speakers | Doctor / Patient labels via pyannote, editable, with a swap button |
| 4 loose headings | SOAP: Chief Complaint, Subjective, Objective, Assessment, Plan, Medications (name, dose, frequency, duration) |
| No hallucination check | Each bullet linked to supporting transcript lines; numbers and drug names must appear in the transcript |
| English only | Hindi / Hinglish via Whisper's translate task |
| Re-ran everything on each click | Session state, cached models |
| Screen only | Markdown, PDF and FHIR-style JSON exports, with identifiers redacted |
| — | LoRA fine-tuning and evaluation scripts |

## Setup

```bash
pip install -r requirements.txt
# ffmpeg is required by Whisper:  sudo apt install ffmpeg   (or: choco install ffmpeg)
streamlit run app.py
```

**Diarization (optional).** Create a Hugging Face token, accept the conditions on
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then
`export HF_TOKEN=hf_...` or paste it in the sidebar. Without it, the app still
works and you can set roles by hand in the transcript table.

## Project layout

```
app.py                 Streamlit UI
clinical/asr.py        Whisper, timestamped segments, translate option
clinical/diarize.py    pyannote, segment-to-speaker matching, Doctor/Patient guess
clinical/redact.py     names, phones, emails, Aadhaar-style numbers
clinical/llm.py        prompts, chunked map-reduce, SOAP parsing
clinical/grounding.py  evidence lines, support score, number and drug checks
clinical/export.py     Markdown, PDF, FHIR-style bundle
finetune_lora.py       LoRA fine-tuning on dialogue -> note CSVs
evaluate.py            ROUGE + groundedness, base vs adapter
```

## Fine-tuning and evaluation

```bash
python finetune_lora.py --train-csv data/train.csv --dialogue-col dialogue --note-col note
python evaluate.py --test-csv data/test.csv --limit 50                                  # base
python evaluate.py --test-csv data/test.csv --limit 50 --adapter adapters/qwen-soap     # fine-tuned
```

Public sources: MTS-Dialog (github.com/abachaa/MTS-Dialog) and ACI-Bench
(github.com/wyim/aci-bench). Rename their columns with the `--*-col` flags, and
reformat reference notes into the six headers above before training.

## Known limits

- Grounding measures support, not meaning. A negation flip ("no fever" → "fever") can pass.
- Whisper segments can span two speakers. Fix the role in the table if needed.
- Redaction is rule-based. Review exports before sharing.
- The FHIR output is a starting point and is not validated against a profile.
- This is a documentation aid. A clinician must review every note.
=======
# 🩺 Clinical Conversation Summarizer

An AI-powered medical assistant that records doctor-patient conversations, transcribes them using Whisper ASR, and generates structured clinical summaries using Qwen LLM.

---

# 🚀 Features

- 🎤 Record doctor-patient conversations
- 📝 Automatic speech-to-text transcription
- 🧠 AI-generated structured clinical summaries
- ⚡ Lightweight and CPU-friendly
- 🖥 Simple Streamlit interface

---

# 🧠 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Speech Recognition | Whisper |
| LLM | Qwen2.5-0.5B-Instruct |
| Backend | Python |
| Audio Processing | FFmpeg |

---

# 📂 Project Structure

```bash
Clinical-Conversation-Summarizer/
│
├── app.py
├── README.md
└── requirements.txt
 
>>>>>>> origin/main
