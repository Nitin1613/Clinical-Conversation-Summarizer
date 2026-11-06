<<<<<<< HEAD
"""Clinical Conversation Summarizer.

Record or upload a doctor-patient conversation -> Whisper transcript ->
speaker diarization (Doctor / Patient) -> identifier redaction -> SOAP note
from Qwen -> every point checked against the transcript -> export.
"""
import os
import tempfile

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from clinical import asr, diarize, export, grounding, llm, redact

st.set_page_config(page_title="Clinical Conversation Summarizer", page_icon="🩺", layout="wide")

ROLES = ["Doctor", "Patient", "Attendant", "Speaker"]
ICON = {"supported": "✅", "weak": "⚠️", "unsupported": "❌", "n/a": "▫️"}
LANGS = {"Auto-detect": None, "English": "en", "Hindi / Hinglish": "hi"}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=1)
    lang_label = st.selectbox("Spoken language", list(LANGS))
    translate = st.toggle("Translate speech to English", value=True,
                          help="Recommended for Hindi or Hinglish. The small LLM works best on English.")
    st.divider()
    hf_token = st.text_input("Hugging Face token (for diarization)", type="password",
                             value=os.environ.get("HF_TOKEN", ""))
    use_diar = st.toggle("Separate speakers", value=bool(hf_token))
    num_speakers = st.number_input("Number of speakers (0 = auto)", 0, 6, 2)
    st.divider()
    do_redact = st.toggle("Redact names, phones and IDs", value=True)
    model_name = st.selectbox("Language model", [llm.DEFAULT_MODEL, "Qwen/Qwen2.5-1.5B-Instruct"])
    adapter_path = st.text_input("LoRA adapter folder (optional)", value=os.environ.get("ADAPTER_PATH", ""))
    use_emb = st.toggle("Semantic grounding check", value=True,
                        help="Adds a small MiniLM embedding model to the word-overlap check.")


# ---------------------------------------------------------- cached models
@st.cache_resource(show_spinner="Loading Whisper…")
def get_asr(size):
    return asr.load_asr(size)


@st.cache_resource(show_spinner="Loading diarization model…")
def get_diarizer(token):
    return diarize.load_diarizer(token)


@st.cache_resource(show_spinner="Loading language model…")
def get_llm(name, adapter):
    return llm.load_llm(name, adapter or None)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


state = st.session_state
for key, default in {"audio": None, "df": None, "df_version": 0, "result": None,
                     "redactions": {}, "language": None}.items():
    state.setdefault(key, default)

# --------------------------------------------------------------------- UI
st.title("🩺 Clinical Conversation Summarizer")
st.caption("Record or upload a consultation. You get a speaker-labelled transcript and a SOAP note "
           "where every point is checked against what was actually said. " + export.DISCLAIMER)

tab_rec, tab_up = st.tabs(["🎤 Record", "📁 Upload"])
with tab_rec:
    rec = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop Recording",
                       just_once=True, use_container_width=True, key="mic")
    if rec:
        state.audio = (rec["bytes"], "." + rec.get("format", "wav"))
with tab_up:
    up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "webm", "ogg", "flac"])
    if up is not None:
        state.audio = (up.getvalue(), os.path.splitext(up.name)[1] or ".wav")

# ------------------------------------------------------- 1. transcription
if state.audio:
    st.audio(state.audio[0])
    if st.button("1 · Transcribe", type="primary"):
        data, suffix = state.audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            path = f.name
        try:
            audio = asr.load_audio(path)
        finally:
            os.remove(path)

        with st.spinner("Transcribing…"):
            segments, state.language = asr.transcribe(get_asr(whisper_size), audio,
                                                      LANGS[lang_label], translate)
        if not segments:
            st.error("No speech was detected in this audio.")
            st.stop()

        diarized = False
        if use_diar and hf_token:
            try:
                with st.spinner("Separating speakers…"):
                    turns = diarize.diarize(get_diarizer(hf_token), audio, num_speakers=num_speakers or None)
                diarize.assign_speakers(segments, turns)
                diarized = True
            except Exception as e:
                st.warning(f"Speaker separation failed, continuing without it: {e}")
        if not diarized:
            for s in segments:
                s["speaker"] = "SPEAKER_00"

        totals = {}
        if do_redact:
            for s in segments:
                s["text"], counts = redact.redact(s["text"])
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
        state.redactions = totals

        roles = diarize.guess_roles(segments)
        state.df = pd.DataFrame({
            "Time": [f"{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}" for s in segments],
            "Role": [roles[s["speaker"]] for s in segments],
            "Text": [s["text"] for s in segments],
        })
        state.df_version += 1
        state.result = None

# --------------------------------------------- 2. review / fix transcript
if state.df is not None:
    st.subheader("Transcript")
    info = []
    if state.language:
        info.append(f"Detected language: {state.language}")
    if state.redactions:
        info.append("Redacted: " + ", ".join(f"{v} {k.lower()}" for k, v in state.redactions.items()))
    if info:
        st.caption(" · ".join(info))
    st.caption("Fix any wrong words or speaker roles here before generating the note.")

    edited = st.data_editor(
        state.df,
        key=f"editor_{state.df_version}",
        hide_index=True,
        disabled=["Time"],
        column_config={
            "Role": st.column_config.SelectboxColumn("Role", options=ROLES, required=True, width="small"),
            "Text": st.column_config.TextColumn("Text", width="large"),
        },
    )

    c1, c2 = st.columns([1, 4])
    if c1.button("⇄ Swap Doctor / Patient"):
        swap = {"Doctor": "Patient", "Patient": "Doctor"}
        edited = edited.copy()
        edited["Role"] = edited["Role"].map(lambda r: swap.get(r, r))
        state.df = edited
        state.df_version += 1
        st.rerun()

    # ------------------------------------------------------ 3. SOAP note
    if c2.button("2 · Generate SOAP note", type="primary"):
        lines = [f"{r.Role}: {r.Text.strip()}" for r in edited.itertuples() if str(r.Text).strip()]
        tok, model = get_llm(model_name, adapter_path.strip())
        status = st.empty()
        note_text, partials = llm.summarize(tok, model, lines, progress=status.info)
        status.empty()

        note = llm.parse_note(note_text)
        meds = llm.parse_medications(note["Medications"])
        grounder = grounding.Grounder(lines, get_embedder() if use_emb else None)
        state.result = {"lines": lines, "raw": note_text, "partials": partials,
                        "checks": grounder.check_note(note), "meds": meds}

# ------------------------------------------------------------- 4. results
res = state.result
if res:
    st.divider()
    st.subheader("SOAP note")
    rate = grounding.support_rate(res["checks"])
    if rate is not None:
        flagged = sum(c["status"] != "supported" for v in res["checks"].values() for c in v if c["status"] != "n/a")
        st.metric("Points supported by the transcript", f"{rate:.0%}",
                  delta=f"{flagged} to review" if flagged else "all clear",
                  delta_color="inverse" if flagged else "normal")
    st.caption("✅ supported · ⚠️ weakly supported, check it · ❌ not found in the transcript")

    left, right = st.columns(2)
    for i, section in enumerate(llm.SECTIONS):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(f"#### {section}")
            if section == "Medications" and res["meds"]:
                st.dataframe(pd.DataFrame(res["meds"]), hide_index=True)
            for c in res["checks"][section]:
                st.markdown(f"{ICON[c['status']]} {c['text'].replace('|', '·')}")
                if c["issues"]:
                    st.caption("⚠ " + "; ".join(c["issues"]))
                if c["evidence"]:
                    st.caption("↳ " + "  /  ".join(c["evidence"]))

    with st.expander("Raw model output"):
        st.code(res["raw"])
        for i, p in enumerate(res["partials"], 1):
            st.markdown(f"**Part {i}**")
            st.code(p)

    st.subheader("Export")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Markdown", export.to_markdown(res["checks"], res["lines"]),
                       "clinical_note.md", "text/markdown")
    d2.download_button("PDF", export.to_pdf(res["checks"], res["lines"]),
                       "clinical_note.pdf", "application/pdf")
    d3.download_button("FHIR-style JSON", export.to_fhir_bundle(res["checks"], res["meds"]),
                       "clinical_note.fhir.json", "application/json")
=======
import streamlit as st
import whisper
import tempfile
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from streamlit_mic_recorder import mic_recorder

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Clinical Conversation Summarizer",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# LOAD WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")


# =========================================================
# LOAD QWEN MODEL
# =========================================================

@st.cache_resource
def load_qwen():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


# =========================================================
# INITIALIZE MODELS
# =========================================================

whisper_model = load_whisper()

tokenizer, qwen_model = load_qwen()


# =========================================================
# UI
# =========================================================

st.title("🩺 Clinical Conversation Summarizer")

st.write(
    "Record doctor-patient conversation, "
    "transcribe it, and generate a structured "
    "clinical summary."
)


# =========================================================
# RECORD AUDIO
# =========================================================

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True,
    use_container_width=True
)


# =========================================================
# PROCESS AUDIO
# =========================================================

if audio:

    st.success("Recording completed!")

    # -----------------------------------------------------
    # SAVE AUDIO
    # -----------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(audio["bytes"])

        temp_audio_path = temp_audio.name

    # -----------------------------------------------------
    # TRANSCRIBE AUDIO
    # -----------------------------------------------------

    st.info("📝 Transcribing audio...")

    result = whisper_model.transcribe(temp_audio_path)

    transcript = result["text"]

    # -----------------------------------------------------
    # SHOW TRANSCRIPT
    # -----------------------------------------------------

    st.subheader("📝 Transcript")

    st.write(transcript)

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an expert medical information extraction system.

Your task is to extract structured clinical information.

IMPORTANT RULES:
- Return ONLY bullet points.
- Do NOT write paragraphs.
- Do NOT explain anything.
- Do NOT add extra text.
- Follow the exact format below.

FORMAT:

Chief Complaints:
- ...

Symptoms:
- ...

Medical History:
- ...

Diagnosis / Clinical Impression:
- ...

Conversation:
{transcript}

Structured Clinical Summary:
"""

    # -----------------------------------------------------
    # TOKENIZE
    # -----------------------------------------------------

    st.info("🧠 Generating clinical summary...")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # -----------------------------------------------------
    # GENERATE SUMMARY
    # -----------------------------------------------------

    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.3,
        eos_token_id=tokenizer.eos_token_id
    )

    # -----------------------------------------------------
    # DECODE OUTPUT
    # -----------------------------------------------------

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # -----------------------------------------------------
    # CLEAN OUTPUT
    # -----------------------------------------------------

    if "Structured Clinical Summary:" in generated_text:

        summary_text = generated_text.split(
            "Structured Clinical Summary:"
        )[-1].strip()

    else:

        summary_text = generated_text.strip()

    # -----------------------------------------------------
    # SHOW SUMMARY
    # -----------------------------------------------------

    st.subheader("🧠 Clinical Summary")

    st.write(summary_text)
>>>>>>> origin/main
