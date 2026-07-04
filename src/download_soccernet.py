"""
download_soccernet.py

Downloads SoccerNet data for the football match summariser pipeline.

Before running:
1. Labels/annotations need no password and download immediately.
2. To get raw match video, first request access via the NDA form linked
   from https://www.soccer-net.org/data — SoccerNet will email you a
   password. Video is licensed through the French league (LFP), so this
   step can't be skipped or automated.
3. Install the SDK (in the conda env):
       pip install SoccerNet

Note for THIS project: scoreboard OCR needs readable digits, so use
--resolution 720p if you'll run the OCR pipeline on the video. 224p is only
fine for the label-driven work SoccerNet was designed for.

Usage:
    # Labels only (no password needed) — good for a first pass: you get
    # ground-truth goal/card/sub timestamps to check your pipeline against.
    python download_soccernet.py --out ./data/soccernet

    # Labels + 720p video for a couple of splits (asks for the NDA password)
    python download_soccernet.py --out ./data/soccernet --videos --resolution 720p
"""

import argparse
from getpass import getpass

from SoccerNet.Downloader import SoccerNetDownloader as SNdl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet data")
    parser.add_argument("--out", default="./data/soccernet", help="Local directory to store the dataset")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits to fetch"
    )
    parser.add_argument(
        "--videos", action="store_true", help="Also download match video (requires the NDA password)"
    )
    parser.add_argument(
        "--resolution",
        choices=["224p", "720p"],
        default="224p",
        help="Video resolution — use 720p if you'll run scoreboard OCR on it",
    )
    args = parser.parse_args()

    downloader = SNdl(LocalDirectory=args.out)

    print("Downloading action-spotting labels (v2)...")
    downloader.downloadGames(files=["Labels-v2.json"], split=args.splits)

    print("Downloading camera-change labels...")
    downloader.downloadGames(files=["Labels-cameras.json"], split=args.splits)

    if args.videos:
        downloader.password = getpass("SoccerNet video password (from the NDA email): ")
        files = [f"1_{args.resolution}.mkv", f"2_{args.resolution}.mkv"]
        print(f"Downloading {args.resolution} match video for splits {args.splits} — this can take a while.")
        downloader.downloadGames(files=files, split=args.splits)

    print(f"\nDone. Data is under {args.out}")


if __name__ == "__main__":
    main()
