#!/usr/bin/env python3
"""Quick correctness + speed check against the GLM53Flash server (no deps beyond stdlib)."""
import json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
def chat(msgs, max_tokens=256, effort=None, stream=False, **kw):
    body = {"model": "GLM53Flash", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, **kw}
    if effort: body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
    m = r["choices"][0]["message"]; u = r["usage"]
    return m, u, dt
print("models:", [m["id"] for m in json.load(urllib.request.urlopen(URL + "/v1/models"))["data"]])
for q in ["What is 17 * 23? Answer with just the number.",
          "Name the capital of Australia in one word."]:
    m, u, dt = chat([{"role": "user", "content": q}], max_tokens=512, effort="low")
    print(f"\nQ: {q}\nreasoning: {str(m.get('reasoning') or m.get('reasoning_content'))[:200]!r}\ncontent: {m.get('content')!r}"
          f"\nusage: {u}  {u['completion_tokens']/dt:.1f} tok/s over {dt:.1f}s")
# medium generation for a steadier decode rate
m, u, dt = chat([{"role": "user", "content": "Write a 150-word paragraph about the history of the bicycle."}], max_tokens=400, effort="low")
print(f"\nparagraph: {m.get('content')[:300]!r}...\nusage: {u}  decode {u['completion_tokens']/dt:.1f} tok/s over {dt:.1f}s")
