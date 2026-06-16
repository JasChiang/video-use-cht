"""Detect speech disfluencies (stutters / repeated words) that Whisper hides.

WHY THIS EXISTS
---------------
Breeze-ASR (and any Whisper-family model) is trained on clean transcripts, so it
NORMALIZES disfluencies away: a stuttered "去去去" comes back as a single "去",
with the neighbouring characters' timestamps stretched to fill the audio. The
repeats are in the audio but NOT in the transcript, so transcript-driven cutting
can't see them.

THE TRICK (fully local, no API)
-------------------------------
Run a wav2vec2 **CTC** model as a parallel "verbatim" track. CTC has no language
model suppressing repeats, so it emits "去去去" / "很多很多" with per-character
timing. Then CROSS-REFERENCE with the Breeze transcript:

    CTC has the repeat  +  Breeze collapsed it   -> real stutter  -> CUT extras
    CTC has the repeat  +  Breeze ALSO has it     -> legit word    -> KEEP
                                                     (媽媽 / 等等 / 看看 / 走走)

That cross-reference is the whole point — raw CTC over-detects (it also flags
genuine reduplication and its own mishearings); Breeze is the sanity check.

OUTPUT
------
Writes <edit>/disfluencies_<stem>.json: a list of {start, end, unit, count_ctc,
count_breeze, kind} cut intervals (the redundant copies to remove, keeping the
count Breeze kept, min 1). Feed these into an EDL as complement keep-ranges.

Model: jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn (the same CTC weights
WhisperX already downloads for Chinese alignment — usually cached). Override with
$DISFLUENCY_MODEL.

Usage:
    python helpers/detect_disfluencies.py <video> [--edit-dir DIR]
    python helpers/detect_disfluencies.py <video> --min-unit 1 --max-unit 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = os.environ.get(
    "DISFLUENCY_MODEL", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
)

# The CTC model emits Simplified Chinese; Breeze emits Traditional. Cross-reference
# matching needs both in the same script, or 妈(CTC)≠媽(Breeze) is a false positive.
try:
    import opencc as _opencc
    _S2T = _opencc.OpenCC("s2t")
    def to_traditional(s: str) -> str:
        return _S2T.convert(s)
except Exception:
    def to_traditional(s: str) -> str:  # graceful fallback (cross-ref may miss S/T pairs)
        return s


def load_audio_16k(video: Path) -> "np.ndarray":  # noqa: F821
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(video),
         "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def ctc_chars(audio, model_name: str) -> list[tuple[float, str]]:
    """Run wav2vec2 CTC greedy decode → [(start_sec, char), ...] verbatim."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
    iv = proc(audio, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(iv).logits[0]
    ids = torch.argmax(logits, dim=-1)
    blank = model.config.pad_token_id
    sec_per_frame = (len(audio) / 16000) / logits.shape[0]

    seq: list[tuple[float, str]] = []
    prev = blank
    for f, i in enumerate(ids.tolist()):
        if i != prev and i != blank:
            ch = proc.tokenizer.decode([i]).strip()
            if ch:
                seq.append((round(f * sec_per_frame, 3), to_traditional(ch)))
        prev = i
    return seq


def detect_repeats(seq: list[tuple[float, str]], min_unit: int, max_unit: int):
    """Find runs where a unit of `u` chars repeats >= 2 times back-to-back.

    Returns list of (unit_str, [unit_start_times], end_time).
    Prefers longer units first so '很多很多' isn't split into '多多'.
    """
    reps = []
    i = 0
    n = len(seq)
    while i < n:
        matched = False
        for u in range(max_unit, min_unit - 1, -1):
            if i + 2 * u > n:
                continue
            unit = "".join(c for _, c in seq[i:i + u])
            starts = [seq[i][0]]
            j = i + u
            while j + u <= n and "".join(c for _, c in seq[j:j + u]) == unit:
                starts.append(seq[j][0])
                j += u
            if len(starts) >= 2:
                end = seq[j - 1 + 0][0] if j - 1 < n else seq[-1][0]
                # end = start of last char of last unit's next; use last unit end approx
                last_unit_end = seq[min(j, n) - 1][0]
                reps.append((unit, starts, last_unit_end))
                i = j
                matched = True
                break
        if not matched:
            i += 1
    return reps


def breeze_text_in(transcript: dict, t0: float, t1: float, pad: float = 0.4) -> str:
    chars = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        s = w.get("start")
        if s is None:
            continue
        if t0 - pad <= s <= t1 + pad:
            chars.append((w.get("text") or "").strip())
    return "".join(chars)


def count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    c = 0
    i = 0
    while (k := haystack.find(needle, i)) != -1:
        c += 1
        i = k + len(needle)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect Whisper-hidden stutters via wav2vec2 CTC + Breeze cross-reference")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit dir holding transcripts/ (default: <video_parent>/edit)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--min-unit", type=int, default=1, help="Min repeating unit length in chars")
    ap.add_argument("--max-unit", type=int, default=2, help="Max repeating unit length in chars")
    ap.add_argument("--pad", type=float, default=0.3,
                    help="Seconds added after each cut unit so the syllable tail is removed")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    tpath = edit_dir / "transcripts" / f"{video.stem}.json"
    if not tpath.exists():
        sys.exit(f"Breeze transcript not found: {tpath} (run transcribe first)")
    transcript = json.loads(tpath.read_text())

    print(f"  running wav2vec2 CTC verbatim track on {video.name} ...", flush=True)
    audio = load_audio_16k(video)
    seq = ctc_chars(audio, args.model)
    reps = detect_repeats(seq, args.min_unit, args.max_unit)

    # unit duration estimate (median spacing between adjacent CTC chars)
    if len(seq) > 1:
        gaps = sorted(seq[i + 1][0] - seq[i][0] for i in range(len(seq) - 1))
        unit_char_dur = max(0.12, gaps[len(gaps) // 2])
    else:
        unit_char_dur = 0.2

    cuts = []
    kept_legit = []
    for unit, starts, last_end in reps:
        n_ctc = len(starts)
        t0, t1 = starts[0], last_end
        bz = breeze_text_in(transcript, t0, t1)
        n_bz = count_occurrences(bz, unit)
        keep = max(1, n_bz)
        if n_ctc > keep:
            # cut the extra copies: from the (keep)-th unit start to the end of the run
            cut_start = round(starts[keep], 3)
            cut_end = round(last_end + len(unit) * unit_char_dur + args.pad, 3)
            cuts.append({
                "start": cut_start, "end": cut_end,
                "unit": unit, "count_ctc": n_ctc, "count_breeze": n_bz,
                "kind": "stutter",
            })
        else:
            kept_legit.append((unit, n_ctc, t0))

    out = edit_dir / f"disfluencies_{video.stem}.json"
    out.write_text(json.dumps({"cuts": cuts}, ensure_ascii=False, indent=2))

    print(f"\n=== stutters to CUT ({len(cuts)}) ===")
    for c in cuts:
        print(f"  🔴 「{c['unit']}」 CTC×{c['count_ctc']} / Breeze×{c['count_breeze']}"
              f"  → cut [{c['start']:.2f}–{c['end']:.2f}]")
    if kept_legit:
        print(f"\n=== kept (legit reduplication, Breeze agrees) ===")
        for unit, n, t in kept_legit:
            print(f"  🟢 「{unit}」×{n} @ {t:.2f}")
    print(f"\nwrote {out.name} ({len(cuts)} cut intervals)")


if __name__ == "__main__":
    main()
