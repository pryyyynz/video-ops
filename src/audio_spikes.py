"""Step 2: find loud moments (crowd/commentary surges) from the audio track.

Extracts mono audio with ffmpeg, computes a per-second RMS loudness curve (dB),
smooths it, and flags spikes - windows well above the clip's own baseline. Loud
moments cluster around goals, near-misses, and cards.

Usage:
    python src/audio_spikes.py data/raw/clip.mp4
"""
import argparse
import subprocess

import numpy as np

SR = 16000       # analysis sample rate (mono audio)
WIN = 1.0        # loudness window, seconds
SMOOTH = 3       # moving-average windows to smooth the loudness curve
BASELINE = 31    # rolling-median baseline width, windows (~30s of local context)
K = 1.5          # sensitivity: flag windows this many robust-std devs above the local baseline


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def load_audio(video_path, sr=SR):
    p = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed:\n" + p.stderr.decode(errors="ignore")[-600:])
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def loudness_db(samples, sr=SR, win=WIN, smooth=SMOOTH):
    n = int(sr * win)
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-6)
    if smooth > 1:
        db = np.convolve(db, np.ones(smooth) / smooth, mode="same")
    return db


def rolling_median(x, width):
    r = width // 2
    pad = np.pad(x, r, mode="edge")
    return np.array([np.median(pad[i:i + width]) for i in range(len(x))])


def find_spikes(db, win=WIN, baseline=BASELINE, k=K):
    res = db - rolling_median(db, baseline)                 # loudness above local norm
    med = np.median(res)
    scale = 1.4826 * np.median(np.abs(res - med)) + 1e-6    # robust std (silence-proof)
    loud = res > med + k * scale
    spikes, i = [], 0
    while i < len(loud):
        if not loud[i]:
            i += 1
            continue
        j = i
        while j < len(loud) and loud[j]:
            j += 1
        peak = i + int(np.argmax(res[i:j]))   # strongest window in this run
        spikes.append((peak * win, float(res[peak])))
        i = j
    return spikes


def run(video_path):
    db = loudness_db(load_audio(video_path))
    spikes = find_spikes(db)
    for t, res in spikes:
        print(f"  {mmss(t)}  +{res:.1f} dB above local baseline")
    print(f"\n{len(spikes)} loud moment(s) over {len(db)}s of audio.")
    return spikes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    args = ap.parse_args()
    run(args.video)
