# Football Match Summariser

Turn a football match video into a **timestamped text recap** of key events plus
**auto-extracted highlight clips** — fully local, on a single 8GB GPU, using open models.

Captioning lives inside the pipeline as the per-segment descriptions that feed the recap.

## Approach (pipeline, not end-to-end training)

Training a video-language model on 8GB VRAM isn't realistic, and a pipeline teaches
more of the real engineering anyway:

1. **Scoreboard OCR** — read score + clock; a score change => candidate goal. *(Step 1, here)*
2. **Audio spikes** — RMS energy peaks => crowd/commentary surges (goals, near-misses, cards).
3. **Fuse** — merge OCR + audio timestamps into a deduplicated event index.
4. **Caption** — BLIP captions on frames around each candidate moment.
5. **Recap** — local quantized LLM (Ollama) turns the structured events into a recap.
6. **Highlights** — ffmpeg cuts clips around high-signal moments.

Optional portfolio flex: a LoRA on BLIP for football vocabulary (corner, free kick, ...).

See [TODO.md](TODO.md) for the phased plan.

## Setup

```bash
conda activate video_summ          # created during scaffold
pip install -r requirements.txt    # if not already installed
```

## Run Step 1 (scoreboard OCR)

```bash
# 1. drop a clip (5-10 min, with at least one goal) at:
#    data/raw/clip.mp4

# 2. dump sample frames to find the scoreboard, then set SCOREBOARD_BOX in src/config.py
python src/frames.py data/raw/clip.mp4 --count 12

# 3. detect score changes (candidate goals)
python src/scoreboard_ocr.py data/raw/clip.mp4
```

## Environment

RTX 5060 Laptop (8GB, Blackwell sm_120), Windows, Python 3.12 (miniforge), Ollama, ffmpeg.
GPU torch for BLIP (Step 4) needs the CUDA 12.8 build; Step 1 OCR runs on CPU.
