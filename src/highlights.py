"""Cut the high-signal moments of a match into one highlights reel.

Reads events.json (fuse.py) and transcript.json (asr.py) from the video's out
dir. Score changes are must-keeps; the rest rank by crowd-surge size with a
boost when the nearby commentary sounds like a big moment. Each kept moment is
padded into a clip window (goals get a long run-up), overlapping windows merge,
and ffmpeg cuts + concatenates them (re-encoded, so joins are clean) into
out/<video>/highlights.mp4, with the cut list in cutlist.json for the UI.

Usage:
    python src/highlights.py data/raw/match.mp4 --target 300
"""
import argparse
import json
import re
import shutil
import subprocess

import config

KEYWORDS = ("goal", "penalt", "red card", "sent off", "crossbar", "off the post",
            "what a save", "great save", "big chance", "free kick")
PAD_GOAL = (20.0, 10.0)   # seconds before/after a score change (buildup + celebration)
PAD_OTHER = (12.0, 6.0)
MERGE_GAP = 2.0           # windows this close together become one clip
MIN_CLIP = 3.0


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def duration_of(video):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(video)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def rank(events, transcript):
    """Score each candidate event; higher = more highlight-worthy."""
    ranked = []
    for e in events:
        goal = next((s for s in e["signals"] if s["source"] == "score_change"), None)
        surge = max((float(m.group(1)) for s in e["signals"] if s["source"] == "audio_spike"
                     and (m := re.search(r"\+([\d.]+)", s["detail"]))), default=0.0)
        nearby = " ".join(seg["text"].lower() for seg in transcript
                          if e["t"] - 15 <= seg["start"] <= e["t"] + 10)
        keyword = next((k for k in KEYWORDS if k in nearby), None)

        reason = []
        if goal:
            board = f" (board {goal['board']})" if goal.get("board") else ""
            reason.append(f"GOAL {goal['detail']}{board}")
        if surge:
            reason.append(f"crowd +{surge:.1f} dB")
        if keyword and not goal:
            reason.append(f"commentary: '{keyword}'")
        ranked.append({
            "t": e["t"], "goal": bool(goal),
            "score": (100 if goal else 0) + surge + (10 if keyword else 0),
            "reason": "; ".join(reason) or "candidate moment",
        })
    ranked.sort(key=lambda r: (-r["score"], r["t"]))
    return ranked


def choose_windows(ranked, target, duration):
    """Take moments in rank order until the reel fills; goals are always kept."""
    total, chosen = 0.0, []
    for r in ranked:
        if total >= target and not r["goal"]:
            break
        pre, post = PAD_GOAL if r["goal"] else PAD_OTHER
        a, b = max(0.0, r["t"] - pre), min(duration, r["t"] + post)
        if b - a < MIN_CLIP:
            continue
        chosen.append((a, b, r["reason"]))
        total += b - a

    chosen.sort()
    merged = []
    for a, b, reason in chosen:
        if merged and a <= merged[-1][1] + MERGE_GAP:
            pa, pb, pr = merged[-1]
            merged[-1] = (pa, max(pb, b), pr if reason in pr else f"{pr}; {reason}")
        else:
            merged.append((a, b, reason))
    return merged


def cut(video, windows, out_path):
    tmp = out_path.parent / "hl_clips"
    tmp.mkdir(exist_ok=True)
    try:
        clips = []
        for i, (a, b, _) in enumerate(windows):
            clip = tmp / f"clip_{i:03d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.2f}", "-i", str(video),
                 "-t", f"{b - a:.2f}", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(clip)],
                check=True,
            )
            clips.append(clip)
        concat = tmp / "concat.txt"
        concat.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
             "-c", "copy", "-movflags", "+faststart", str(out_path)],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(video_path, target=300):
    d = config.out_dir(video_path)
    events = json.loads((d / "events.json").read_text())["events"]
    try:
        transcript = json.loads((d / "transcript.json").read_text(encoding="utf-8"))["segments"]
    except FileNotFoundError:
        transcript = []  # ranking just loses the commentary boost
    duration = duration_of(video_path)
    if not events:
        raise SystemExit("No candidate events in events.json - nothing to cut.")

    windows = choose_windows(rank(events, transcript), target, duration)
    print(f"{len(windows)} clip(s), {sum(b - a for a, b, _ in windows):.0f}s total "
          f"(target {target}s):")
    for a, b, reason in windows:
        print(f"  {mmss(a)}-{mmss(b)}  {reason}")

    out = d / "highlights.mp4"
    cut(video_path, windows, out)
    (d / "cutlist.json").write_text(json.dumps(
        [{"start": a, "end": b, "text": reason} for a, b, reason in windows], indent=2))
    print(f"\nReel: {out} ({duration_of(out):.0f}s)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--target", type=int, default=300, help="target reel length, seconds")
    a = p.parse_args()
    run(a.video, a.target)
