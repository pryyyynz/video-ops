"""Local web frontend for the video toolbox.

Pick one or more tasks (summarise / caption / cut highlights), then paste a
YouTube link or upload a video. Shared pipeline steps (download, Whisper,
scoreboard+audio fuse, scene captions) run once per job; each task's result
card appears as soon as its finisher lands.

Run:  conda run -n video_summ python app.py     (then open http://localhost:5000)
"""
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, "src")
import config
import pipeline
from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB uploads
app.config["TEMPLATES_AUTO_RELOAD"] = True  # pick up index.html edits without a restart

JOBS = {}  # job_id -> state dict

TASK_LABELS = {
    "football_recap": "Football recap",
    "video_summary": "Video summary",
    "subtitles": "Subtitles",
    "visual_captions": "Visual captions",
    "highlights": "Highlights",
}

WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

# Rough ETA model (seconds) from measured runs; the countdown self-corrects as it runs.
def job_eta(tasks, duration):
    eta = 15
    if set(tasks) & pipeline.NEEDS_ASR:
        eta += 45 + 0.6 * (duration or 0)      # whisper + (usually) fuse dominate
    for task, cost in (("football_recap", 55), ("video_summary", 55),
                       ("visual_captions", 150), ("highlights", 60)):
        if task in tasks:
            eta += cost
    return int(eta)


def build_result(job, stem, task, video, fmt):
    """Map a finished task's artifact file(s) to the result payload the UI renders."""
    d = config.out_dir(video)
    if task == "football_recap":
        return {"type": "markdown", "markdown": (d / "recap.md").read_text(encoding="utf-8")}
    if task == "video_summary":
        return {"type": "markdown", "markdown": (d / "summary.md").read_text(encoding="utf-8")}
    if task in ("subtitles", "visual_captions"):
        src = d / ("transcript.json" if task == "subtitles" else "visual_captions.json")
        segments = json.loads(src.read_text(encoding="utf-8"))["segments"]
        job["captions"][task] = (segments, fmt)
        return {"type": "captions", "segments": segments, "download": f"/captions/{stem}/{task}"}
    # highlights
    job["media"] = str(d / "highlights.mp4")
    items = json.loads((d / "cutlist.json").read_text(encoding="utf-8"))
    return {"type": "video", "src": f"/media/{stem}", "items": items}


def worker(stem, tasks, source, final, title, fmt, target_min, whisper):
    job = JOBS[stem]

    def task_done(task, video):
        result = build_result(job, stem, task, video, fmt)
        result["task"] = task
        job["results"].append(result)

    try:
        pipeline.run_tasks(source, stem, tasks, whisper=whisper, final=final, title=title,
                           target=target_min * 60,
                           progress=lambda s: job.update(stage=s),
                           task_start=lambda t: job.update(current=t),
                           task_done=task_done)
        job.update(stage="Done", done=True, current=None)
    except Exception as e:  # surface any step failure to the UI
        job.update(stage="Failed", done=True, error=str(e), current=None)


def ts(seconds, sep=","):
    h, m, s = int(seconds // 3600), int(seconds % 3600 // 60), seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def captions_file(segments, fmt):
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for s in segments:
            lines += [f"{ts(s['start'], '.')} --> {ts(s['end'], '.')}", s["text"], ""]
    else:
        lines = []
        for i, s in enumerate(segments, 1):
            lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["text"], ""]
    return "\n".join(lines)


@app.post("/process")
def start():
    stem = "job_" + uuid.uuid4().hex[:8]
    wanted = (request.form.get("tasks") or "football_recap").split(",")
    tasks = [t for t in TASK_LABELS if t in wanted]   # validate; run order is pipeline's
    if not tasks:
        return jsonify(error="Pick at least one task."), 400
    fmt = request.form.get("format", "srt")
    fmt = fmt if fmt in ("srt", "vtt") else "srt"
    try:
        target_min = max(1, min(30, int(request.form.get("target", "5"))))
    except ValueError:
        target_min = 5
    whisper = request.form.get("whisper", "medium")
    whisper = whisper if whisper in WHISPER_MODELS else "medium"

    url = (request.form.get("url") or "").strip()
    final, title = None, None
    if url:
        source = url
    elif "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        config.RAW.mkdir(parents=True, exist_ok=True)
        dest = config.RAW / f"{stem}{Path(f.filename).suffix or '.mp4'}"
        f.save(dest)
        source = str(dest)
        title = Path(f.filename).stem                    # original name -> teams / score
        final = pipeline.title_score(title)
    else:
        return jsonify(error="Provide a URL or a file."), 400

    try:
        duration, _ = pipeline.video_meta(source)
    except Exception:
        duration = 0
    eta = job_eta(tasks, duration)

    run_order = [t for t in pipeline.RUN_ORDER if t in tasks]
    JOBS[stem] = dict(stage="Queued", tasks=run_order, current=None, results=[], captions={},
                      eta=eta, duration=duration, started=time.time(),
                      done=False, error=None, media=None)
    threading.Thread(target=worker, args=(stem, tasks, source, final, title, fmt, target_min, whisper),
                     daemon=True).start()
    return jsonify(job_id=stem, eta=eta, duration=duration)


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    elapsed = int(time.time() - job["started"])
    return jsonify(stage=job["stage"], done=job["done"], error=job["error"],
                   tasks=[{"id": t, "label": TASK_LABELS[t]} for t in job["tasks"]],
                   current=job["current"], results=job["results"],
                   elapsed=elapsed, remaining=max(0, job["eta"] - elapsed),
                   eta=job["eta"], duration=job["duration"])


@app.get("/captions/<job_id>/<task>")
def captions(job_id, task):
    job = JOBS.get(job_id)
    pair = (job or {}).get("captions", {}).get(task)
    if not pair:
        return jsonify(error="no captions for this job/task"), 404
    segments, fmt = pair
    return Response(captions_file(segments, fmt),
                    mimetype="text/vtt" if fmt == "vtt" else "text/plain",
                    headers={"Content-Disposition": f"attachment; filename={task}.{fmt}"})


@app.get("/media/<job_id>")
def media(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("media"):
        return jsonify(error="no media for this job"), 404
    return send_file(job["media"], conditional=True)  # conditional: range requests for seeking


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
