---
name: video-use-install
description: Install video-use-cht into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and wire up ffmpeg + the local Breeze-ASR (WhisperX) pipeline so the user can start editing immediately. Transcription is local — no transcription API key.
---

# video-use install

Use this file only for first-time install or reconnect. For daily editing, read `SKILL.md`. Always read `helpers/` — that's where the scripts live.

## What you're doing

You're setting up a conversation-driven video editor for the user. After install, the user drops raw footage into any folder, runs their agent (`claude`, `codex`, etc.) there, and says "edit these into a launch video." You do the rest by reading `SKILL.md`.

Three things must exist on this machine:

1. The `video-use` repo cloned somewhere stable.
2. `ffmpeg` on `$PATH` (plus optional `yt-dlp` for online sources).
3. Python deps installed (`whisperx` etc.) in a 3.10–3.12 venv. Transcription runs **locally** with Breeze-ASR — no transcription API key. `HF_TOKEN` is optional (pyannote diarization only).

And one thing must be true about the current agent:

4. It can discover `SKILL.md` — either via a global skills directory (`~/.claude/skills/`, `~/.codex/skills/`) or via a `CLAUDE.md` / system-prompt import.

## Install prompt contract

- Do everything yourself. Only ask the user for confirmation before `brew install`, and (only for multi-speaker footage) for an optional `HF_TOKEN`.
- Prefer a stable clone path like `~/Developer/video-use-cht` (not `/tmp`, not `~/Downloads`).
- The skill references helpers by bare name (`transcribe.py`, `render.py`). That works because SKILL.md and `helpers/` ship together — keep them as siblings when you register the skill.
- After install, verify by running one real command against one real file. Don't declare success on file-existence checks alone.

## Steps

### 1. Clone to a stable path

```bash
test -d ~/Developer/video-use-cht || git clone <your-fork-url> ~/Developer/video-use-cht
cd ~/Developer/video-use-cht
```

If the repo is already there, `git pull --ff-only` and continue.

### 2. Install Python deps

```bash
# Prefer uv if available; fall back to pip.
command -v uv >/dev/null && uv sync || pip install -e .
```

`pyproject.toml` lists `requests`, `librosa`, `matplotlib`, `pillow`, `numpy`. No console scripts — helpers are invoked directly as `python helpers/<name>.py`.

### 3. Install ffmpeg (+ optional yt-dlp)

`ffmpeg` and `ffprobe` are hard requirements. `yt-dlp` is only needed if the user wants to pull sources from URLs. Animation engines such as HyperFrames, Remotion, and Manim are installed lazily the first time a project actually needs them.

```bash
# macOS
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v yt-dlp >/dev/null || brew install yt-dlp     # optional

# Debian / Ubuntu
# sudo apt-get update && sudo apt-get install -y ffmpeg
# pip install yt-dlp

# Arch
# sudo pacman -S ffmpeg yt-dlp
```

If `brew` / `apt` / `pacman` requires a sudo prompt, tell the user the exact command and wait. Do not invent a password.

### 4. Register the skill with the current agent

Figure out which agent you are running under, and register once. A symlink of the whole repo directory is the right shape — helpers/ needs to sit next to SKILL.md.

- **Claude Code** (`~/.claude/` present):

    ```bash
    mkdir -p ~/.claude/skills
    ln -sfn ~/Developer/video-use ~/.claude/skills/video-use
    ```

- **Codex** (`$CODEX_HOME` set, or `~/.codex/` present):

    ```bash
    mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
    ln -sfn ~/Developer/video-use "${CODEX_HOME:-$HOME/.codex}/skills/video-use"
    ```

- **Hermes / Openclaw / another agent with a skills directory**: symlink `~/Developer/video-use` into that agent's skills directory under the name `video-use`. If the agent has no skills directory, add a line to its system prompt / config pointing at `~/Developer/video-use/SKILL.md` (e.g. an `@~/Developer/video-use/SKILL.md` import in a `CLAUDE.md`-equivalent).

If you can't tell which agent you're in, ask the user once: "which agent am I running under — Claude Code, Codex, or something else?" Then pick the right target.

### 5. Transcription model (local) + optional diarization token

Transcription is **local** via Breeze-ASR (WhisperX). There is **no transcription API key**. The
faster-whisper Breeze-ASR-25 weights download automatically from Hugging Face on first run.

`HF_TOKEN` is OPTIONAL and only enables speaker diarization (pyannote). Without it, transcription
still works and every word is tagged `speaker_0`.

1. Only if the user has **multi-speaker** footage and wants speaker labels, ask once:

    > For multi-speaker footage I can label who's speaking using pyannote. That needs a free Hugging
    > Face token. Create one at https://huggingface.co/settings/tokens, accept the terms for
    > `pyannote/speaker-diarization-community-1`, and paste it here. Or skip — single-speaker editing
    > works fine without it.

    When the user pastes a token, write it to `.env`:

    ```bash
    printf 'HF_TOKEN=%s\n' "$TOKEN" > ~/Developer/video-use-cht/.env
    chmod 600 ~/Developer/video-use-cht/.env
    ```

    Never echo the token back in tool output. Never commit `.env`.

2. (Optional) For Taiwanese-Hokkien (台語) footage, set up Breeze-ASR-26 — see "Breeze-ASR-26 (Taigi)
   setup" in `README.md`. Default `breeze25` covers Mandarin + English.

### 6. Verify end-to-end

Run one real thing. Prefer the lightest verification that still proves the pipeline is wired up:

```bash
~/Developer/video-use-cht/.venv/bin/python -c "import whisperx; print('whisperx OK')"
~/Developer/video-use-cht/.venv/bin/python ~/Developer/video-use-cht/helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
ffprobe -version | head -1
```

A full transcription test downloads ~3 GB of model weights on first run. It's free (local), but
only run it when the user hands you their first clip unless they ask.

### 7. Hand off

Tell the user, in one short message:

- Where the skill is installed (`~/Developer/video-use`).
- That they should `cd` into their footage folder and start their agent there (e.g. `claude`).
- That a good first message is: *"edit these into a launch video"* or *"inventory these takes and propose a strategy."*
- That all outputs land in `<videos_dir>/edit/` — the repo stays clean.

## Keeping the skill current

- `cd ~/Developer/video-use && git pull --ff-only` pulls the latest code. The symlink auto-picks it up on the next run.
- If `pyproject.toml` changed deps, re-run `uv sync` / `pip install -e .` after pulling.

## Cold-start reminders

- Symlink the **whole directory**, not just `SKILL.md`. The helpers need to sit next to it.
- If `.env` exists but the key is empty, treat it the same as missing — don't assume existence means validity.
- `ffmpeg` from static builds works fine. Any modern (≥ 4.x) build is enough.
- `yt-dlp` is optional. Don't block install on it; install lazily the first time a user asks to pull from a URL.
- Node.js/npm are only needed for HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- HyperFrames, Remotion, and Manim are optional animation engines. Don't install or prefer one globally during setup; pick the engine per animation slot in `SKILL.md`. HyperFrames can run through `npx --yes hyperframes ...` in the slot directory. Remotion can be scaffolded with `npx create-video@latest` or installed inside the slot before rendering.
- Transcription is local and free, but the first run downloads ~3 GB of model weights — don't trigger it during install verification unless the user asks.
- Transcription runs on CPU on Apple Silicon (CT2 has no Metal path). It's not real-time; that's expected for offline editing.
- If the user is on Linux without a package manager Claude recognizes, print the manual `ffmpeg` install URL and wait rather than guessing.
