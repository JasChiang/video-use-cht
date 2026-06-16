"""Transcribe a video locally with Breeze-ASR-25 (Traditional Chinese) via WhisperX.

This is the video-use-cht backend: a drop-in replacement for the ElevenLabs
Scribe backend used by upstream video-use. It produces the SAME JSON shape that
``pack_transcripts.py`` consumes — a top-level ``{"words": [...]}`` list whose
entries are typed ``word`` | ``spacing`` | ``audio_event``, each carrying
``start``/``end`` seconds, ``text``, and ``speaker_id`` (e.g. ``speaker_0``).

Pipeline:
    ffmpeg                         -> mono 16 kHz wav
    whisperx (faster-whisper)      -> segments + text, using Breeze-ASR-25 CT2 weights
    whisperx align (wav2vec2)      -> precise word-level start/end
    pyannote diarization           -> per-word speaker (optional, needs HF token)
    adapter                        -> Scribe-compatible words[] (synthesizes spacing)

Model: SoybeanMilk/faster-whisper-Breeze-ASR-25 (CTranslate2 build of
MediaTek-Research/Breeze-ASR-25, Apache-2.0). Override with $BREEZE_MODEL.

Apple Silicon note: CTranslate2 has no Metal/MPS GPU path, so transcription runs
on CPU (Accelerate). That is fine for offline editing — it is not real-time but
it is local and private. Alignment/diarization run on CPU here too for safety.

Usage:
    python helpers/transcribe_breeze.py <video_path>
    python helpers/transcribe_breeze.py <video_path> --language en
    python helpers/transcribe_breeze.py <video_path> --num-speakers 2
    python helpers/transcribe_breeze.py <video_path> --no-diarize
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Friendly aliases -> HF model ids. Pick per source:
#   breeze25 = Taiwanese Mandarin + English code-switch (中 + 英). Pre-built CT2 weights.
#   breeze26 = BreezeASR-Taigi: Taiwanese Hokkien + Mandarin (台 + 中), outputs Han characters.
#              No public CT2 build yet -> needs local ct2-transformers-converter (see README).
MODEL_ALIASES = {
    "breeze25": "SoybeanMilk/faster-whisper-Breeze-ASR-25",
    "25": "SoybeanMilk/faster-whisper-Breeze-ASR-25",
    "breeze26": "MediaTek-Research/Breeze-ASR-26",
    "26": "MediaTek-Research/Breeze-ASR-26",
    "taigi": "MediaTek-Research/Breeze-ASR-26",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.strip().lower(), name)


DEFAULT_MODEL = resolve_model(os.environ.get("BREEZE_MODEL", "breeze25"))
DEFAULT_LANGUAGE = "zh"  # Breeze is tuned for Taiwanese Mandarin; code-switch English stays inline.
DEFAULT_DEVICE = os.environ.get("BREEZE_DEVICE", "cpu")  # CT2 has no MPS path on Apple Silicon.
DEFAULT_COMPUTE_TYPE = os.environ.get("BREEZE_COMPUTE_TYPE", "int8")  # int8 keeps CPU RAM/latency sane.

# WhisperX models are large; loading one per worker would OOM. Cache + serialize.
_LOCKS = threading.Lock()
_asr_models: dict[tuple, object] = {}
_align_cache: dict[str, tuple] = {}
_diar_pipeline = None
_INFER_LOCK = threading.Lock()  # serialize CPU inference across batch threads


def load_api_key() -> str:
    """Return the Hugging Face token (used for pyannote diarization).

    Named ``load_api_key`` to stay drop-in compatible with the Scribe backend
    that ``transcribe_batch.py`` imports. The token is OPTIONAL: without it,
    transcription still runs and diarization is silently skipped (every word
    gets speaker_id ``speaker_0``).
    """
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                    return v.strip().strip('"').strip("'")
    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(env_key, "")
        if v:
            return v
    return ""  # optional; diarization disabled when empty


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _get_asr_model(model_name: str, device: str, compute_type: str, language: str):
    import whisperx

    key = (model_name, device, compute_type, language)
    with _LOCKS:
        if key not in _asr_models:
            _asr_models[key] = whisperx.load_model(
                model_name, device, compute_type=compute_type, language=language
            )
        return _asr_models[key]


def _get_align_model(language: str, device: str):
    import whisperx

    with _LOCKS:
        if language not in _align_cache:
            _align_cache[language] = whisperx.load_align_model(
                language_code=language, device=device
            )
        return _align_cache[language]


# pyannote.audio 4.x ships speaker-diarization-community-1 as its flagship pipeline,
# and even loading the older 3.1 pulls community-1 embedding artifacts — so target it
# directly. Accept its gated terms once at:
#   https://huggingface.co/pyannote/speaker-diarization-community-1
DIARIZE_MODEL = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")


def _get_diarizer(hf_token: str, device: str, model_name: str = DIARIZE_MODEL):
    global _diar_pipeline
    with _LOCKS:
        if _diar_pipeline is None:
            # DiarizationPipeline moved modules across whisperx versions.
            try:
                from whisperx.diarize import DiarizationPipeline
            except ImportError:  # older layout
                from whisperx import DiarizationPipeline  # type: ignore
            # The token kwarg was renamed token<-use_auth_token across versions.
            try:
                _diar_pipeline = DiarizationPipeline(
                    model_name=model_name, token=hf_token, device=device
                )
            except TypeError:
                _diar_pipeline = DiarizationPipeline(
                    model_name=model_name, use_auth_token=hf_token, device=device
                )
        return _diar_pipeline


def _norm_speaker(raw) -> str | None:
    """Map whisperx 'SPEAKER_00' -> Scribe-style 'speaker_0' so pack_transcripts
    strips the prefix and shows 'S0'."""
    if raw is None:
        return None
    s = str(raw)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"speaker_{int(digits)}"
    return s


def _to_scribe_words(segments: list[dict]) -> list[dict]:
    """Flatten whisperx aligned segments into a Scribe-compatible words[] list.

    Emits a 'word' entry per token and synthesizes a 'spacing' entry for the
    gap to the next token, so the JSON is shape-identical to ElevenLabs Scribe
    output and pack_transcripts' silence/speaker breaking works unchanged.
    """
    words_out: list[dict] = []
    last_end: float | None = None

    flat: list[dict] = []
    for seg in segments:
        seg_speaker = seg.get("speaker")
        for w in seg.get("words", []):
            text = (w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            # Alignment occasionally drops timestamps (e.g. pure-digit tokens);
            # carry the timeline forward so downstream math never sees None.
            if start is None:
                start = last_end if last_end is not None else 0.0
            if end is None:
                end = start
            flat.append({
                "text": text,
                "start": float(start),
                "end": float(end),
                "speaker_id": _norm_speaker(w.get("speaker", seg_speaker)),
            })
            last_end = float(end)

    for i, w in enumerate(flat):
        words_out.append({
            "type": "word",
            "start": w["start"],
            "end": w["end"],
            "text": w["text"],
            "speaker_id": w["speaker_id"],
        })
        if i + 1 < len(flat):
            gap_start, gap_end = w["end"], flat[i + 1]["start"]
            if gap_end > gap_start:
                words_out.append({
                    "type": "spacing",
                    "start": gap_start,
                    "end": gap_end,
                    "text": " ",
                })
    return words_out


def run_breeze(
    audio_path: Path,
    hf_token: str = "",
    language: str | None = None,
    num_speakers: int | None = None,
    diarize: bool = True,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    verbose: bool = True,
) -> dict:
    """Run the full Breeze/WhisperX pipeline and return a Scribe-shaped payload."""
    import whisperx

    lang = language or DEFAULT_LANGUAGE

    with _INFER_LOCK:  # CPU-bound; never run two transcriptions concurrently
        audio = whisperx.load_audio(str(audio_path))

        model = _get_asr_model(model_name, device, compute_type, lang)
        result = model.transcribe(audio, batch_size=16, language=lang)
        detected_lang = result.get("language", lang)

        align_model, metadata = _get_align_model(detected_lang, device)
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        if diarize:
            if not hf_token:
                if verbose:
                    print("  ! no HF token — skipping diarization (single speaker)", flush=True)
            else:
                diarizer = _get_diarizer(hf_token, device)
                diar_kwargs = {}
                if num_speakers:
                    diar_kwargs["num_speakers"] = num_speakers
                diarize_segments = diarizer(audio, **diar_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)

    words = _to_scribe_words(result.get("segments", []))
    return {
        "language_code": detected_lang,
        "text": "".join(
            w["text"] for w in words if w.get("type") == "word"
        ),
        "words": words,
    }


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    diarize: bool = True,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Transcribe a single video with Breeze. Returns path to transcript JSON.

    Signature mirrors the Scribe backend (``api_key`` here is the HF token) so
    ``transcribe_batch.py`` works unchanged. Cached: returns existing path
    immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        if verbose:
            print(f"  transcribing {video.stem} with Breeze-ASR-25 (cpu)", flush=True)
        payload = run_breeze(
            audio,
            hf_token=api_key,
            language=language,
            num_speakers=num_speakers,
            diarize=diarize,
            model_name=model_name,
            verbose=verbose,
        )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with Breeze-ASR-25 (WhisperX)")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir", type=Path, default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language", type=str, default=None,
        help=f"ISO language code (default: {DEFAULT_LANGUAGE}). Use 'en' for English-only takes.",
    )
    ap.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Model id or alias: breeze25 (中 + 英, default), breeze26/taigi (台 + 中). "
             "Or any HF/CT2 model path.",
    )
    ap.add_argument(
        "--num-speakers", type=int, default=None,
        help="Number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization (single-speaker footage).",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    hf_token = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=hf_token,
        language=args.language,
        num_speakers=args.num_speakers,
        diarize=not args.no_diarize,
        model_name=resolve_model(args.model),
    )


if __name__ == "__main__":
    main()
