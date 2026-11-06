"""Speaker diarization (pyannote) and Doctor / Patient role assignment."""

import os
import re

PIPELINE_ID = "pyannote/speaker-diarization-3.1"

DOCTOR_CUES = (
    "prescribe", "mg", "tablet", "capsule", "syrup", "twice a day", "once a day",
    "examine", "blood test", "x-ray", "x ray", "scan", "follow up", "follow-up",
    "let me check", "since when", "how long", "any history", "i'll give",
    "i will give", "take this", "blood pressure", "come back",
)


def load_diarizer(token=None):
    """Load the pyannote pipeline. Needs a Hugging Face token with the model's
    conditions accepted. Returns None when no token is available."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        return None
    from pyannote.audio import Pipeline
    import torch

    try:  # pyannote >= 4 uses `token`, 3.x uses `use_auth_token`
        pipe = Pipeline.from_pretrained(PIPELINE_ID, token=token)
    except TypeError:
        pipe = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    return pipe


def diarize(pipe, audio, sample_rate=16000, num_speakers=None):
    """Run diarization on a 16 kHz mono numpy waveform.
    Returns a list of (start, end, speaker_label)."""
    import torch

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    annotation = getattr(out, "speaker_diarization", out)  # pyannote 4 wraps it
    return [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]


def assign_speakers(segments, turns):
    """Label each Whisper segment with the speaker whose turns overlap it most."""
    for seg in segments:
        best, best_overlap = None, 0.0
        for start, end, spk in turns:
            overlap = min(seg["end"], end) - max(seg["start"], start)
            if overlap > best_overlap:
                best, best_overlap = spk, overlap
        if best is None and turns:  # no overlap: use the nearest turn
            mid = (seg["start"] + seg["end"]) / 2
            best = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2]
        seg["speaker"] = best or "SPEAKER_00"
    return segments


def guess_roles(segments):
    """Heuristic: the speaker asking most questions and using clinical
    language is the Doctor. The user can correct this in the UI."""
    speakers = sorted({s.get("speaker", "SPEAKER_00") for s in segments})
    if len(speakers) == 1:
        return {speakers[0]: "Speaker"}

    scores = {spk: 0 for spk in speakers}
    for seg in segments:
        text = seg["text"].lower()
        score = 2 * text.count("?")
        score += sum(1 for cue in DOCTOR_CUES if re.search(r"\b" + re.escape(cue) + r"\b", text))
        scores[seg.get("speaker", "SPEAKER_00")] += score

    doctor = max(scores, key=scores.get)
    return {spk: ("Doctor" if spk == doctor else "Patient") for spk in speakers}
