#!/usr/bin/env python3
"""Video smoke test: POST test.mp4 (red ball moving left->right; green rectangle appears halfway) as a data URL."""
import base64, json, os, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.mp4")
data = base64.b64encode(open(VIDEO, "rb").read()).decode()
body = {"model": "GLM53Flash", "max_tokens": 400, "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}},
            {"type": "text", "text": "Describe what happens in this short video: which shapes appear, their colors, and how they move or change over time."}]}]}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=900))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:800]); sys.exit(2)
m = r["choices"][0]["message"]; u = r["usage"]
print(f"usage={u}  {time.time()-t0:.1f}s")
print("content:", m.get("content"))
c = (m.get("content") or "").lower()
print("VIDEO TEST", "OK" if ("red" in c and ("circle" in c or "ball" in c or "dot" in c) and "green" in c) else "CHECK MANUALLY")
