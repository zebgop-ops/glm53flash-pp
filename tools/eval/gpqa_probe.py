#!/usr/bin/env python3
"""Re-run selected GPQA items (default: the truncated ones from results/<tag>/gpqa_diamond.json) with the
reasoning text saved, and classify each: degenerate loop (detect.find_loop on the reasoning) vs genuine long
deliberation. usage: gpqa_probe.py <tag> [url] [model] [budget] [idx,idx,...]"""
import json, os, re, sys, time, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/r/qwen38-flashnext-pp/tools"); from detect import find_loop
HERE=os.path.dirname(os.path.abspath(__file__)); tag=sys.argv[1]
URL=(sys.argv[2] if len(sys.argv)>2 else "http://localhost:8006")+"/v1/chat/completions"
MODEL=sys.argv[3] if len(sys.argv)>3 else "GLM53Flash-Uncensored"; BUDGET=int(sys.argv[4]) if len(sys.argv)>4 else 65536
rows=json.load(open(f"{HERE}/data/gpqa_diamond.json"))["rows"]
res=json.load(open(f"{HERE}/results/{tag}/gpqa_diamond.json"))
idx=[int(x) for x in sys.argv[5].split(",")] if len(sys.argv)>5 else [r["i"] for r in res if r["finish"]=="length"]
print("probing", len(idx), "items:", idx, flush=True)
def one(i):
    body={"model":MODEL,"messages":[{"role":"user","content":rows[i]["question"]+"\n\nThink it through carefully, then finish with a final line of the form:\nAnswer: <letter>"}],"max_tokens":BUDGET,"temperature":0}
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=7200))
    m=r["choices"][0]["message"]; reasoning=str(m.get("reasoning") or m.get("reasoning_content") or ""); content=m.get("content") or ""
    reps,unit=find_loop(reasoning[-20000:]); tail=reasoning[-300:]
    # coarse repetition measure: share of the last 20k chars made of the single most common 60-char line
    lines=[l for l in reasoning[-20000:].splitlines() if len(l)>20]; top=max(set(lines), key=lines.count) if lines else ""; share=lines.count(top)/max(1,len(lines))
    out={"i":i,"finish":r["choices"][0].get("finish_reason"),"tokens":r["usage"]["completion_tokens"],"gold":rows[i]["answer"],"content":content[-120:],
         "loop_reps":reps,"loop_unit":unit[:12],"top_line_share":round(share,2),"top_line":top[:80],"tail":tail}
    json.dump({"reasoning":reasoning,"content":content}, open(f"{HERE}/results/{tag}/probe_{i}.json","w"))
    print(f"item {i}: finish={out['finish']} tokens={out['tokens']} loop_reps={reps} top_line_share={share:.2f} content={content[-60:]!r}\n   tail: {tail[-200:]!r}", flush=True); return out
with cf.ThreadPoolExecutor(4) as ex: outs=list(ex.map(one, idx))
json.dump(outs, open(f"{HERE}/results/{tag}/probe_summary.json","w"), indent=1)
