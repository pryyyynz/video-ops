"""Local web frontend for the football match summariser.

Paste a YouTube link or upload a video; it runs the full pipeline and shows the
recap, with a duration-based time estimate.

Run:  conda run -n video_summ python app.py     (then open http://localhost:5000)
"""
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, "src")
import config
import pipeline
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB uploads

JOBS = {}  # job_id -> state dict

# Rough ETA model (seconds) from measured runs; the countdown self-corrects as it runs.
ETA_BASE, ETA_PER_SEC, ETA_RECAP = 45, 0.6, 55


def estimate(duration):
    return int(ETA_BASE + ETA_PER_SEC * (duration or 0) + ETA_RECAP)


def worker(stem, source, final, title):
    job = JOBS[stem]
    try:
        recap_path = pipeline.process(source, stem, progress=lambda s: job.update(stage=s), final=final, title=title)
        job.update(stage="Done", done=True, recap=Path(recap_path).read_text(encoding="utf-8"))
    except Exception as e:  # surface any step failure to the UI
        job.update(stage="Failed", done=True, error=str(e))


@app.post("/process")
def start():
    stem = "job_" + uuid.uuid4().hex[:8]
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
    JOBS[stem] = dict(stage="Queued", eta=estimate(duration), duration=duration,
                      started=time.time(), done=False, error=None, recap=None)
    threading.Thread(target=worker, args=(stem, source, final, title), daemon=True).start()
    return jsonify(job_id=stem, eta=JOBS[stem]["eta"], duration=duration)


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="unknown job"), 404
    elapsed = int(time.time() - job["started"])
    return jsonify(stage=job["stage"], done=job["done"], error=job["error"], recap=job["recap"],
                   elapsed=elapsed, remaining=max(0, job["eta"] - elapsed),
                   eta=job["eta"], duration=job["duration"])


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Football Match Summariser</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
 :root{color-scheme:dark}
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0e1116;color:#e6edf3;margin:0;padding:2rem;display:flex;justify-content:center}
 .card{width:100%;max-width:760px}
 h1{font-size:1.4rem;margin:0 0 .25rem}
 p.sub{color:#8b949e;margin:0 0 1.5rem}
 input[type=text]{width:100%;padding:.7rem;border-radius:8px;border:1px solid #30363d;background:#161b22;color:#e6edf3;font-size:1rem;box-sizing:border-box}
 .row{display:flex;gap:.75rem;align-items:center;margin:.75rem 0}
 .divider{color:#6e7681;font-size:.85rem;text-align:center;margin:.5rem 0}
 button{padding:.7rem 1.2rem;border-radius:8px;border:0;background:#238636;color:#fff;font-size:1rem;cursor:pointer}
 button:disabled{background:#30363d;cursor:not-allowed}
 .panel{margin-top:1.5rem;padding:1rem 1.25rem;border:1px solid #30363d;border-radius:10px;background:#161b22;display:none}
 .bar{height:8px;background:#30363d;border-radius:4px;overflow:hidden;margin:.75rem 0}
 .bar>div{height:100%;width:0;background:#238636;transition:width .5s}
 .stage{font-weight:600}
 .muted{color:#8b949e;font-size:.9rem}
 #result{margin-top:1.5rem;padding:.25rem 1.5rem 1rem;border:1px solid #30363d;border-radius:10px;background:#161b22;display:none}
 #result h2{border-bottom:1px solid #30363d;padding-bottom:.3rem}
 .err{color:#f85149}
</style></head>
<body><div class="card">
 <h1>&#9917; Football Match Summariser</h1>
 <p class="sub">Paste a YouTube link or upload a match video. Runs locally: Whisper + scoreboard OCR + LLM recap.</p>
 <input id="url" type="text" placeholder="https://youtu.be/...">
 <div class="divider">&mdash; or &mdash;</div>
 <div class="row"><input id="file" type="file" accept="video/*"></div>
 <div class="row"><button id="go">Summarise</button><span id="hint" class="muted"></span></div>
 <div id="progress" class="panel">
   <div class="stage" id="stage">Starting...</div>
   <div class="bar"><div id="fill"></div></div>
   <div class="muted"><span id="remaining"></span> &middot; <span id="etatext"></span></div>
 </div>
 <div id="result"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>{s=Math.max(0,Math.round(s));return Math.floor(s/60)+":"+String(s%60).padStart(2,'0')};
$('go').onclick=async()=>{
 const url=$('url').value.trim(), file=$('file').files[0];
 if(!url&&!file){$('hint').textContent='Enter a link or choose a file.';return;}
 $('go').disabled=true;$('hint').textContent='';$('result').style.display='none';
 const fd=new FormData(); if(url)fd.append('url',url); else fd.append('file',file);
 let r; try{r=await(await fetch('/process',{method:'POST',body:fd})).json();}
 catch(e){$('hint').textContent='Request failed.';$('go').disabled=false;return;}
 if(r.error){$('hint').textContent=r.error;$('go').disabled=false;return;}
 $('progress').style.display='block';
 $('etatext').textContent='est. '+fmt(r.eta)+' for a '+fmt(r.duration)+' video';
 poll(r.job_id);
};
async function poll(id){
 let s; try{s=await(await fetch('/status/'+id)).json();}catch(e){return setTimeout(()=>poll(id),2500);}
 $('stage').textContent=s.stage;
 $('fill').style.width=Math.min(100,100*s.elapsed/Math.max(1,s.eta))+'%';
 $('remaining').textContent=s.done?'done':('~'+fmt(s.remaining)+' left');
 if(s.done){
   $('go').disabled=false;
   if(s.error){$('stage').innerHTML='<span class="err">Failed: '+s.error+'</span>';return;}
   $('fill').style.width='100%';
   const el=$('result');el.style.display='block';
   el.innerHTML=window.marked?marked.parse(s.recap):'<pre>'+s.recap+'</pre>';
   return;
 }
 setTimeout(()=>poll(id),2000);
}
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
