"""Shared Ollama client: text prompts, optionally with images (vision models)."""
import json
import urllib.request

URL = "http://localhost:11434/api/generate"


def ask(prompt, model="gemma3:12b", num_ctx=16384, images=None, temperature=0.1, timeout=900):
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }
    if images:
        payload["images"] = images  # base64-encoded JPEG/PNG, for vision models
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()
