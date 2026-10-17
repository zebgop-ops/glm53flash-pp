#!/usr/bin/env python3
"""GPQA-Diamond (198 multiple-choice, fingertap mirror: options embedded, answer letter) on a served build.
Greedy, server-default reasoning effort, concurrency 6, generous token budget (reasoning counts).
usage: run_gpqa.py <tag> [url] [model]"""
import json, os, re, sys, time, urllib.request, concurrent.futures as cf
HERE=os.path.dirname(os.path.abspath(__file__))
tag=sys.argv[1]; URL=(sys.argv[2] if len(sys.argv)>2 else "http://localhost:8006")+"/v1/chat/completions"
MODEL=sys.argv[3] if len(sys.argv)>3 else "GLM53Flash-Uncensored"
OUT=f"{HERE}/results/{tag}"; os.makedirs(OUT, exist_ok=True)
rows=json.load(open(f"{HERE}/data/gpqa_diamond.json"))["rows"]
def chat(prompt, max_tokens=65536):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0}
    err=None
    for _ in range(3):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=7200))
            m=r["choices"][0]["message"]; return (m.get("content") or ""), r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason")
        except Exception as e: err=e; time.sleep(5)
    return f"__ERROR__ {err}", 0, "error"
def item(i, row):
    content, toks, fin = chat(row["question"]+"\n\nThink it through carefully, then finish with a final line of the form:\nAnswer: <letter>")
    m=re.findall(r"Answer:\s*\**\s*\(?([ABCD])\)?", content, re.I); pred=m[-1].upper() if m else None
    if pred is None:
        m=re.findall(r"\b([ABCD])\b", content[-80:]); pred=m[-1] if m else None
    gold=str(row["answer"]).strip()[0].upper()
    return {"i":i,"ok":pred==gold,"gold":gold,"pred":pred,"tokens":toks,"finish":fin}
t=time.time(); res=[]
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(lambda a: item(*a), enumerate(rows)):
        res.append(r)
        if len(res)%20==0: print(f"  gpqa: {len(res)}/{len(rows)} acc so far {sum(x['ok'] for x in res)/len(res):.3f} (truncated {sum(x['finish']=='length' for x in res)})", flush=True)
json.dump(res, open(f"{OUT}/gpqa_diamond.json","w"), indent=1)
acc=sum(x["ok"] for x in res)/len(res); trunc=sum(x["finish"]=="length" for x in res); err=sum(x["finish"]=="error" for x in res); toks=sum(x["tokens"] for x in res)
line=f"gpqa_diamond: {sum(x['ok'] for x in res)}/{len(res)} = {acc:.3f} | truncated {trunc} | errors {err} | unparsed {sum(x['pred'] is None for x in res)} | {toks} completion tokens ({toks//len(res)} avg) in {time.time()-t:.0f}s"
open(f"{OUT}/summary_gpqa.txt","w").write(f"{tag} {MODEL} {URL}\n{line}\n"); print("SUMMARY", tag, "|", line, flush=True)
