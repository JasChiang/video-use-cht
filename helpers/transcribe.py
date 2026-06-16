"""Transcription entrypoint for video-use-cht.

Upstream video-use used ElevenLabs Scribe here. This fork swaps in a local
Breeze-ASR-25/26 (WhisperX) backend — see ``transcribe_breeze.py``. This module
stays as the stable import surface: ``transcribe_batch.py`` and any other caller
keep doing ``from transcribe import load_api_key, transcribe_one`` and get the
Breeze backend transparently. The produced JSON is shape-compatible with what
``pack_transcripts.py`` expects (Scribe-style ``words[]``).

``api_key`` throughout means the Hugging Face token used for pyannote
diarization, and it is OPTIONAL (no token -> diarization skipped).

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2 --model breeze25
    python helpers/transcribe.py <video_path> --model taigi --no-diarize
"""

from __future__ import annotations

from transcribe_breeze import (  # re-exported for backward-compatible imports
    DEFAULT_MODEL,
    load_api_key,
    resolve_model,
    transcribe_one,
)
from transcribe_breeze import main  # CLI is identical; delegate to the backend

__all__ = ["DEFAULT_MODEL", "load_api_key", "resolve_model", "transcribe_one", "main"]


if __name__ == "__main__":
    main()
