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