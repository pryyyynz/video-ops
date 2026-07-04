"""Step 1: read the scoreboard (score, and clock if shown) to find goals.

Scoreboards sit in the top-left corner but differ between broadcasts, so instead of
a per-broadcast pixel box, read_scoreboard() OCRs the whole top-left region and
anchors on the two team-code tokens (e.g. MUN ... EVE), reading the score between
them. That works whether or not the scoreboard shows a clock, and keeps banners and
sponsors outside the two codes from leaking in. A score change, confirmed over two
reads, is a goal, tagged with the board clock if one is shown.

Usage:
    python src/scoreboard_ocr.py data/raw/clip.mp4
"""
import argparse
import re

import cv2
import easyocr
import torch

import config

# Match clock, e.g. "8:25" / "36:18" / "120:00". Colon OR dot (OCR often reads the
# colon as a dot). Used to find the clock and to strip it when reading the score.
CLOCK_RE = re.compile(r"(\d{1,3})[.:](\d{2})")
CODE_RE = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{2,4}$")   # team abbr., >=1 letter: MUN, GIR, B04, S04


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def crop(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def read_scoreboard(reader, frame):
    """Return (score, clock) from the top-left region.

    score is (home, away) or None; clock is "M:SS" or None. Handles both scoreboard
    topologies: inline (TEAM1 1-1 TEAM2, score between the codes) and stacked (two
    rows, TEAM score / TEAM score). Anchors on the team-code tokens.
    """
    toks = []   # (xleft, ycenter, height, text)
    # mag_ratio=2 upscales internally so EasyOCR detects small single-digit scores.
    for box, text, _ in reader.readtext(crop(frame, config.SCOREBOARD_REGION), mag_ratio=2):
        ys = [p[1] for p in box]
        toks.append((min(p[0] for p in box), sum(ys) / len(ys), max(max(ys) - min(ys), 1), text.strip()))

    clock = None
    for _, _, _, text in toks:
        m = CLOCK_RE.search(text)
        if m and int(m.group(1)) <= 130 and int(m.group(2)) < 60:
            clock = f"{int(m.group(1))}:{m.group(2)}"
            break

    codes = [t for t in toks if CODE_RE.match(t[3])]   # team abbreviations
    ints = []                                          # (xleft, yc, h, value)
    for xl, yc, h, text in toks:
        s = CLOCK_RE.sub(" ", text)
        if any(c.isalpha() for c in s):                # skip clock / team / sponsor tokens
            continue
        for n in re.findall(r"\d{1,2}", s):
            if int(n) <= 20:                           # football scores are small
                ints.append((xl, yc, h, int(n)))

    # inline: two codes on the same row, score between them (MUN 1-1 EVE)
    for xl_i, yc_i, h_i, _ in codes:
        for xl_j, yc_j, h_j, _ in codes:
            if xl_i >= xl_j or abs(yc_i - yc_j) > max(h_i, h_j):
                continue
            ymid, h = (yc_i + yc_j) / 2, max(h_i, h_j)
            between = [(xl, v) for xl, yc, _, v in ints
                       if xl_i < xl < xl_j and abs(yc - ymid) <= 0.6 * h]
            between.sort(key=lambda z: z[0])   # by x only; stable keeps digit order within a token
            if len(between) >= 2:
                return (between[0][1], between[1][1]), clock

    # stacked: each team on its own row with its score to the right (GIR 0 / OVI 2)
    rows = []
    for cxl, cyc, ch, _ in codes:
        right = [(xl, v) for xl, yc, ih, v in ints
                 if xl > cxl and abs(yc - cyc) <= max(ch, ih)]
        right.sort(key=lambda z: z[0])
        if right:
            rows.append((cyc, right[0][1]))
    if len(rows) >= 2:
        rows.sort()                                    # top row = home
        return (rows[0][1], rows[1][1]), clock

    return None, clock


def run(video_path):
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / config.SAMPLE_FPS)))

    last_score = None
    last_clock = None         # most recent clock read, to tag changes whose exact frame missed it
    pending = None            # (score, first_seen_t) awaiting a 2nd confirming read
    changes = []
    sampled = detected = 0
    for frame_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)   # seek: decode only sampled frames, not all
        ok, frame = cap.read()
        if not ok:
            continue
        sampled += 1
        t = frame_idx / src_fps
        score, clock = read_scoreboard(reader, frame)
        if clock:
            last_clock = clock
        if score:
            detected += 1
            if score == last_score:
                pending = None
            elif pending and pending[0] == score:   # confirmed over 2 detections
                board = clock or last_clock
                old = "start" if last_score is None else f"{last_score[0]}-{last_score[1]} ->"
                print(f"  {mmss(pending[1])}  {old} {score[0]}-{score[1]}  (board {board})")
                if last_score is not None:
                    changes.append((pending[1], last_score, score, board))
                last_score = score
                pending = None
            else:
                pending = (score, t)   # first sighting; hold

    cap.release()

    print(f"\nScoreboard read in {detected}/{sampled} sampled frames; "
          f"{len(changes)} confirmed score change(s).")
    return changes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    a = p.parse_args()
    run(a.video)
