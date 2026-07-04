"""Step 3: fuse OCR score-changes + audio spikes into one candidate-event index.

Runs both detectors, then clusters timestamps within MERGE_WINDOW seconds into
single candidate events. An event backed by both sources (score change AND a
crowd surge) is a high-confidence candidate. Writes data/events.json for the
captioning step to consume.

Usage:
    python src/fuse.py data/raw/clip.mp4
"""
import argparse
import json

import audio_spikes
import config
import scoreboard_ocr

MERGE_WINDOW = 10.0  # seconds: signals this close together are one event


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def fuse(signals, window=MERGE_WINDOW):
    """signals: list of (t, source, detail, board) -> list of candidate-event dicts."""
    events = []
    for t, source, detail, board in sorted(signals, key=lambda s: s[0]):
        sig = {"t": t, "source": source, "detail": detail}
        if board:
            sig["board"] = board
        if events and t - events[-1]["cluster_end"] <= window:
            e = events[-1]
            e["cluster_end"] = t
            e["signals"].append(sig)
        else:
            events.append({"t": t, "cluster_end": t, "signals": [sig]})
    for e in events:
        del e["cluster_end"]
        e["sources"] = sorted({s["source"] for s in e["signals"]})
        e["confidence"] = "high" if len(e["sources"]) > 1 else "low"
    return events


def run(video_path):
    print("-- scoreboard OCR --")
    changes = scoreboard_ocr.run(video_path)
    print("\n-- audio spikes --")
    spikes = audio_spikes.run(video_path)

    signals = [
        (t, "score_change", f"{old[0]}-{old[1]} -> {new[0]}-{new[1]}", clock)
        for t, old, new, clock in changes
    ] + [
        (t, "audio_spike", f"+{res:.1f} dB", None)
        for t, res in spikes
    ]
    events = fuse(signals)

    print("\n-- candidate events --")
    for e in events:
        details = "; ".join(f"{s['source']}({s['detail']})" for s in e["signals"])
        print(f"  {mmss(e['t'])}  [{e['confidence']:4}]  {details}")

    out = config.out_dir(video_path) / "events.json"
    out.write_text(json.dumps({"video": str(video_path), "events": events}, indent=2))
    print(f"\n{len(events)} candidate event(s) -> {out}")
    return events


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    a = p.parse_args()
    run(a.video)
