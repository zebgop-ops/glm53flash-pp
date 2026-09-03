#!/usr/bin/env python3
"""Needle-in-haystack recall at a target token depth against GLM53Flash (stdlib only).
usage: needle.py [tokens=30000] [url]"""
import json, sys, time, random, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8002"
random.seed(N)
code = f"{random.randint(1000,9999)}-{random.choice('ABCDEFGH')}{random.randint(10,99)}"
# ~4.6 tokens per filler sentence; vary the wording so prefix caching cannot short-circuit
fill = [f"Sector {i} reported nominal readings on cycle {random.randint(1,999)}." for i in range(N // 12)]
pos = int(len(fill) * 0.55)
fill.insert(pos, f"IMPORTANT: the maintenance access code for bay seven is {code}. Remember it.")
doc = " ".join(fill)
q = "What is the maintenance access code for bay seven? Reply with only the code."
body = {"model": "GLM53Flash", "max_tokens": 200, "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "messages": [{"role": "user", "content": doc + "\n\n" + q}]}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=1800))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:400]); sys.exit(2)
dt = time.time() - t0
m = r["choices"][0]["message"]; u = r["usage"]
ok = code in (m.get("content") or "")
print(f"prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']} time={dt:.1f}s "
      f"prefill~{u['prompt_tokens']/dt:.0f} tok/s")
print("expected:", code, "| content:", repr((m.get("content") or "")[:120]), "| reasoning:", repr(str(m.get("reasoning") or m.get("reasoning_content"))[:120]))
print("RECALL", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
