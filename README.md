# Video Toolbox

Turn a video into **summaries, captions, and highlight reels** — fully local, on a
single 8GB GPU, using open models. Five tasks from one web UI (`app.py`), multi-selectable
per run. It's a pipeline of small single-purpose steps, not an end-to-end model: each step
runs as its own subprocess so only one model holds VRAM at a time, and shared steps
(download, Whisper, scoreboard+audio fuse, scene captions) run **once per job** no matter
how many selected tasks need them.

See [TODO.md](TODO.md) for the phased build log.

## The five tasks

### 1. Football recap (`src/recap.py`)
A timestamped match report from three fused signals:
- **Commentary ASR** — Whisper (`medium`, auto language) transcribes the full audio; this is
  the primary evidence.
- **Scoreboard OCR** — EasyOCR reads the top-left board on sparsely sampled frames (0.5 fps,
  seek-based). It anchors on the MM:SS clock or the team codes, so any broadcast works with
  no per-channel calibration; a confirmed score change = a candidate goal with a board-clock time.
- **Audio spikes** — per-second RMS loudness vs a local rolling baseline marks crowd surges.

`fuse.py` clusters signals within 10s into candidate events, then the LLM (gemma3:12b via
Ollama) runs two passes: first it pins the final score from the commentary (unless the
title/filename already has one, e.g. "Man Utd 4-4 Everton"), then it writes `## Summary`
and `## Goals` with scorers and board-clock times. OCR is treated as weak corroboration
only — commentary decides.

### 2. Video summary (`src/summarise.py`)
Any video, not just football. The Whisper transcript goes to gemma3:12b, which writes a
`## Summary` plus timestamped `## Chapters`. If the transcript is too thin to carry a
summary (< 200 words — silent footage, music-only), the pipeline automatically runs the
visual-captions step first and the summary leans on the scene descriptions instead.

### 3. Subtitles (`src/asr.py` + export)
The Whisper transcript's timed segments, downloadable as SRT or VTT. Language is
auto-detected; non-English speech stays in its language.

### 4. Visual captions (`src/visual_captions.py`)
Timed descriptions of what's *on screen*, independent of audio:
ffmpeg scene-change detection (threshold 0.30, boundaries ≥ 8s apart, capped at 30 scenes,
with a fixed 30s grid as fallback for static video) → one keyframe per scene, downscaled to
≤ 896px → **gemma3:12b vision** describes each frame in one sentence (it reads names off
jerseys). Output is a `{start, end, text}` timeline rendered in the UI and exportable as
SRT/VTT — and it doubles as the evidence for summarising speech-poor videos.

### 5. Football highlights (`src/highlights.py`)
Cuts the fused events into one reel:
1. **Rank** — score changes +100 (goals always make the cut), plus the crowd-surge dB,
   plus 10 if nearby commentary contains a keyword (goal, penalty, red card, ...).
2. **Select** — pad each moment into a window (goals −20s/+10s for buildup + celebration,
   others −12s/+6s) and take top-ranked windows until the target length fills (a soft
   budget: goals are kept even past it).
3. **Merge + cut** — windows re-sort chronologically, overlaps merge, ffmpeg re-encodes each
   cut (libx264 — frame-accurate cuts, clean joins) and concatenates into `highlights.mp4`,
   with the reasons in `cutlist.json` under the player.

## What runs on GPU vs CPU

| Engine | Device | Used by |
|---|---|---|
| Whisper ASR (openai-whisper `medium`) | GPU (CUDA; CPU fallback, much slower) | recap, summary, subtitles, highlights |
| Scoreboard OCR (EasyOCR) | GPU (CPU fallback) | recap, highlights |
| gemma3:12b text + vision (Ollama) | GPU with partial CPU offload (12B > 8GB VRAM) | recap, summary, visual captions |
| Audio spikes (ffmpeg RMS + numpy) | CPU | recap, highlights |
| Scene detect, cutting, concat (ffmpeg/libx264) | CPU | visual captions, highlights |
| Download (yt-dlp) | CPU / network | any URL input |

Steps run sequentially as subprocesses, so the 8GB card never holds two models at once.

## Local setup

```bash
# 1. Python env (miniforge/conda, Python 3.12)
conda create -n video_summ python=3.12
conda activate video_summ
pip install -r requirements.txt

# 2. GPU torch — the default pip torch is CPU-only; Blackwell cards (RTX 50xx, sm_120)
#    need the CUDA 12.8 build. Whisper + EasyOCR pick it up automatically.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. ffmpeg + ffprobe on PATH  (audio analysis, scene detection, cutting)
#    https://ffmpeg.org/download.html  (or: winget install ffmpeg / apt install ffmpeg)

# 4. Ollama + the model (recap, summary, visual captions) — must be running
#    https://ollama.com
ollama pull gemma3:12b

# 5. optional: deno on PATH — lets yt-dlp solve YouTube's JS challenges (avoids 403s).
#    Only needed for URL inputs; local files skip it.
```

## Run

```bash
python app.py          # any python with Flask works; pipeline steps auto-locate
                       # the video_summ env. Then open http://localhost:5000
```

Pick tasks (multi-select), paste a YouTube link or upload a file, and results stream in as
each task finishes. Or from the CLI:

```bash
python src/pipeline.py data/raw/match.mp4 --tasks subtitles,highlights --target 300
python src/pipeline.py "https://youtu.be/..." --tasks football_recap
```

Per-video artifacts land in `data/out/<video-stem>/`: `transcript.json`, `events.json`,
`visual_captions.json`, `recap.md`, `summary.md`, `highlights.mp4`, `cutlist.json`.

## Environment this was built on

RTX 5060 Laptop (8GB, Blackwell sm_120), Windows 11, Python 3.12 (miniforge), Ollama, ffmpeg.
