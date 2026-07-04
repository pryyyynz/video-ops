"""Fetch a short football test clip into data/raw/ with yt-dlp.

Downloads at <=720p (enough for scoreboard OCR, keeps the file small) plus audio
(needed later for Step 2 audio spikes), and can trim to a time range so you don't
pull a whole 2-hour match.

Usage:
    # whole video
    python src/download.py "https://www.youtube.com/watch?v=..." --name clip

    # just minutes 12:00-20:00 (only fetches that section)
    python src/download.py "https://..." --name clip --start 12:00 --end 20:00
"""
import argparse
import subprocess
import sys

import config


def download(url, name, start=None, end=None):
    config.RAW.mkdir(parents=True, exist_ok=True)
    out = config.RAW / f"{name}.%(ext)s"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]",
        "--merge-output-format", "mp4",
        "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "5",
        "-o", str(out),
    ]
    if start and end:
        cmd += ["--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts"]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    print(f"Saved to {config.RAW / (name + '.mp4')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--name", default="clip", help="output filename stem in data/raw/")
    p.add_argument("--start", help="clip start, e.g. 12:00 or 00:12:00")
    p.add_argument("--end", help="clip end, e.g. 20:00")
    a = p.parse_args()
    download(a.url, a.name, a.start, a.end)
