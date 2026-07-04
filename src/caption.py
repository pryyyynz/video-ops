"""Step 4: caption frames around each candidate event with BLIP.

Reads data/events.json (from fuse.py), samples FRAME_OFFSETS frames around each
event timestamp, captions them with BLIP, and writes the captions back into
events.json for the recap step.

Usage:
    python src/caption.py
"""
import json

import cv2
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

import config

MODEL = "Salesforce/blip-image-captioning-base"
FRAME_OFFSETS = [0.0, 2.0]  # seconds after the event timestamp


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def grab_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def run():
    events_path = config.DATA / "events.json"
    data = json.loads(events_path.read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP on {device}...")
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL).to(device)

    cap = cv2.VideoCapture(data["video"])
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {data['video']}")

    for e in data["events"]:
        captions = []
        for off in FRAME_OFFSETS:
            img = grab_frame(cap, e["t"] + off)
            if img is None:
                continue
            inputs = processor(img, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            captions.append(processor.decode(out[0], skip_special_tokens=True))
        e["captions"] = captions
        print(f"  {mmss(e['t'])}  {captions}")

    cap.release()
    events_path.write_text(json.dumps(data, indent=2))
    print(f"\nCaptions written back to {events_path}")


if __name__ == "__main__":
    run()
