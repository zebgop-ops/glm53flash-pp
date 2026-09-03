#!/usr/bin/env python3
"""Hit-path soak: N multi-turn conversations over one shared document, concurrent,
sampled, with planted-code recall and constant-token loop detection.
This is the trigger for the PP + MTP + prefix-cache state corruption
(FINDINGS.md section 9): every turn re-sends the whole conversation -> prefix-cache
hits, several slots busy, MTP on.   usage: hitpath.py [conversations=12] [turns=6] [conc=6]"""
import json, os, random, sys, time, urllib.request, concurrent.futures as cf, uuid
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, "/home/r/qwen38-flashnext-pp/tools"):
    sys.path.insert(0, p)
from detect import find_loop
URL = os.environ.get("GLM53_URL", "http://localhost:8002/v1/chat/completions")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 6
random.seed(7)
words = "alpha beta gamma delta kernel tensor stream block table rank stage draft accept state cache page slot".split()
DOC = " ".join(random.choice(words) for _ in range(14000))   # ~20k tokens of shared prefix
def call(msgs):
    body = {"model": "GLM53Flash", "messages": msgs, "max_tokens": 400, "temperature": 1.0,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or ""), r["usage"]
def conversation(ci):
    rnd = random.Random(ci)
    code = f"{rnd.randint(10000, 99999)}"
    doc = DOC.split(" "); doc.insert(rnd.randint(1000, 13000), f"(the access code for room {ci} is {code})")
    msgs = [{"role": "user", "content": f"Document {uuid.uuid4()}:\n{' '.join(doc)}\n\nAcknowledge in one line."}]
    loops, recalls, tokens = 0, [], 0
    for t in range(TURNS):
        content, reasoning, usage = call(msgs)
        tokens += usage["completion_tokens"]
        for txt in (content, reasoning):
            reps, unit = find_loop(txt) if txt else (0, "")
            if reps:
                loops += 1; print(f"  LOOP conv{ci} turn{t}: reps={reps} unit={unit[:12]!r} tail={txt[-60:]!r}", flush=True); break
        msgs.append({"role": "assistant", "content": content or "..."})
        if t == TURNS - 2:
            msgs.append({"role": "user", "content": f"What is the access code for room {ci}? Digits only."})
        elif t == TURNS - 1:
            ok = code in content; recalls.append(ok)
            if not ok: print(f"  MISS conv{ci}: expected {code}, got {content[:160]!r}", flush=True)
            break
        else:
            msgs.append({"role": "user", "content": rnd.choice([
                "Summarize the document's third quarter in two sentences.",
                "List five distinct words from the document.",
                "Write a short poem using words from the document.",
                "Count how many times 'kernel' appears; an estimate is fine.",
                "Describe the document's structure briefly."])})
    return loops, recalls, tokens
t0 = time.time()
with cf.ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(conversation, range(N)))
loops = sum(r[0] for r in res); rec = [x for r in res for x in r[1]]; toks = sum(r[2] for r in res)
print(f"conversations={N} turns={TURNS} conc={CONC}: loops={loops} / {N*TURNS} responses, "
      f"recall {sum(rec)}/{len(rec)} exact, {toks} completion tokens in {time.time()-t0:.0f}s")
