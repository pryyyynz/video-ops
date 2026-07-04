"""Central config for the football match summariser MVP."""
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"        # input match clips
FRAMES = DATA / "frames"  # extracted frames (calibration / debug)
CLIPS = DATA / "clips"    # output highlight clips
OUT = DATA / "out"        # per-video pipeline outputs (transcript/events/recap)

# Frame sampling rate for scoreboard OCR (frames per second). The score changes
# slowly, so 0.5 (every 2s) is plenty and halves the OCR pass.
SAMPLE_FPS = 0.5

# Scoreboard crop as fractions of (width, height): (x0, y0, x1, y1).
# Find these with src/frames.py, then set them here. None = whole frame.
# Scoreboards sit in the top-left corner. This generous region covers it for any
# broadcast; read_scoreboard() then locates the score + clock within it by pattern
# (find the MM:SS clock, read the score off the same bar row), so there is no
# per-broadcast pixel calibration.
SCOREBOARD_REGION = (0.0, 0.0, 0.42, 0.22)


def out_dir(video_path):
    """Per-video output directory, so results from different videos don't clobber."""
    d = OUT / Path(video_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d
