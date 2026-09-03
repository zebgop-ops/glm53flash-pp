#!/usr/bin/env python3
"""Concurrent decode throughput: N simultaneous 300-token generations (stdlib only).
usage: bench.py [concurrency=4] [url]"""
import json, sys, time, threading, urllib.request
C = int(sys.argv[1]) if len(sys.argv) > 1 else 4
URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8002"
topics = ["the history of the bicycle", "how volcanoes form", "the rules of chess", "why the sky is blue",
          "the water cycle", "how bread rises", "the life of honeybees", "how sailboats sail upwind"]
res = [None] * C
def one(i):
    body = {"model": "GLM53Flash", "max_tokens": 300, "temperature": 0.7, "ignore_eos": True,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": f"Write a detailed essay about {topics[i % len(topics)]} (variant {i})."}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1200)); dt = time.time() - t0
    res[i] = (r["usage"]["completion_tokens"], dt)
ths = [threading.Thread(target=one, args=(i,)) for i in range(C)]
t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
tot = sum(r[0] for r in res)
print(f"C={C}: {tot} tokens in {wall:.1f}s -> aggregate {tot/wall:.1f} tok/s, per-stream {tot/wall/C:.1f} tok/s")
