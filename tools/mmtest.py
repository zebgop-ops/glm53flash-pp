#!/usr/bin/env python3
"""Multimodal smoke test: send a generated PNG (blue square, red circle on white) as a data URL."""
import base64, json, struct, sys, time, urllib.request, zlib
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
W = H = 256
def px(x, y):
    if 40 <= x < 120 and 40 <= y < 120: return (30, 60, 220)          # blue square, top-left
    if (x - 180) ** 2 + (y - 170) ** 2 < 45 ** 2: return (220, 30, 30) # red circle, bottom-right
    return (255, 255, 255)
raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(W)) for y in range(H))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
data_url = "data:image/png;base64," + base64.b64encode(png).decode()
body = {"model": "GLM53Flash", "max_tokens": 300, "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "List every shape in this image with its color and rough position. One line per shape."}]}]}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=600))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:600]); sys.exit(2)
m = r["choices"][0]["message"]; u = r["usage"]
print(f"usage={u}  {time.time()-t0:.1f}s")
print("reasoning:", repr(str(m.get("reasoning") or m.get("reasoning_content"))[:300]))
print("content:", m.get("content"))
c = (m.get("content") or "").lower()
ok = ("blue" in c and "red" in c and ("square" in c or "rectangle" in c) and ("circle" in c or "round" in c or "dot" in c))
print("MM TEST", "OK" if ok else "CHECK MANUALLY")
