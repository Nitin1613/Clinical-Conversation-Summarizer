"""Speech-to-text with Whisper, returning timestamped segments."""


def load_asr(size: str = "base"):
    import whisper
    return whisper.load_model(size)


def load_audio(path: str):
    """Decode any audio file (via ffmpeg) to 16 kHz mono float32."""
    import whisper
    return whisper.load_audio(path)


def transcribe(model, audio, language=None, translate=False):
    """Return (segments, detected_language).

    Each segment is {"start", "end", "text"}. Set translate=True to turn
    Hindi / Hinglish speech into English text, which the small LLM handles far
    better than code-mixed input.
    """
    import torch

    result = model.transcribe(
        audio,
        language=language,
        task="translate" if translate else "transcribe",
        fp16=torch.cuda.is_available(),
        condition_on_previous_text=False,  # reduces repeated-phrase loops
    )
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in result.get("segments", [])
        if s["text"].strip()
    ]
    return segments, result.get("language")
