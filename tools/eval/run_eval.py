#!/usr/bin/env python3
"""Accuracy eval for a served GLM-5.3-Flash build: GSM8K (first 200 test items) + HumanEval (164).
Greedy (temperature 0), server-default reasoning effort, concurrency 6. Results -> eval/results/<tag>/.
usage: run_eval.py <tag> [url] [model]      e.g. run_eval.py glm53u http://localhost:8003 GLM53Flash-Uncensored"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request, concurrent.futures as cf
HERE=os.path.dirname(os.path.abspath(__file__))
tag=sys.argv[1]; URL=(sys.argv[2] if len(sys.argv)>2 else "http://localhost:8003")+"/v1/chat/completions"
MODEL=sys.argv[3] if len(sys.argv)>3 else "GLM53Flash-Uncensored"
OUT=f"{HERE}/results/{tag}"; os.makedirs(OUT, exist_ok=True)
def chat(prompt, max_tokens=12000):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0}
    for attempt in range(3):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=3600))
            m=r["choices"][0]["message"]; return (m.get("content") or ""), r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason")
        except Exception as e:
            err=e; time.sleep(5)
    return f"__ERROR__ {err}", 0, "error"
def num(s):
    s=s.replace(",","").replace("$","")
    m=re.findall(r"-?\d+(?:\.\d+)?", s); return m[-1] if m else None
def gsm_item(i, row):
    content, toks, fin = chat(row["question"]+"\n\nSolve step by step, then give the final answer as a number on the last line in the form: #### <number>")
    gold=num(row["answer"].split("####")[-1]); tail=content.split("####")[-1] if "####" in content else content[-200:]
    pred=num(tail)
    ok = pred is not None and gold is not None and abs(float(pred)-float(gold))<1e-6
    return {"i":i,"ok":ok,"gold":gold,"pred":pred,"tokens":toks,"finish":fin}
def he_item(i, row):
    content, toks, fin = chat("Complete the following Python function. Return the complete function (with the signature and any needed imports) in a single ```python code block and nothing else.\n\n```python\n"+row["prompt"]+"```")
    blocks=re.findall(r"```(?:python)?\n(.*?)```", content, re.S); code=blocks[-1] if blocks else content
    prog=code+"\n\n"+row["test"]+f"\n\ncheck({row['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"t.py"); open(p,"w").write(prog)
        try:
            res=subprocess.run([sys.executable,"-I",p],capture_output=True,text=True,timeout=20,cwd=d); ok=res.returncode==0; err=res.stderr[-200:]
        except subprocess.TimeoutExpired: ok=False; err="timeout"
    return {"i":i,"task":row["task_id"],"ok":ok,"tokens":toks,"finish":fin,"err":"" if ok else err}
def run(name, data, fn):
    t=time.time(); res=[]
    with cf.ThreadPoolExecutor(6) as ex:
        for r in ex.map(lambda a: fn(*a), enumerate(data)):
            res.append(r)
            if len(res)%20==0: print(f"  {name}: {len(res)}/{len(data)} acc so far {sum(x['ok'] for x in res)/len(res):.3f}", flush=True)
    json.dump(res, open(f"{OUT}/{name}.json","w"), indent=1)
    acc=sum(x["ok"] for x in res)/len(res); trunc=sum(x["finish"]=="length" for x in res); err=sum(x["finish"]=="error" for x in res)
    toks=sum(x["tokens"] for x in res)
    line=f"{name}: {sum(x['ok'] for x in res)}/{len(res)} = {acc:.3f} | truncated {trunc} | errors {err} | {toks} completion tokens in {time.time()-t:.0f}s"
    print(line, flush=True); return line
lines=[run("gsm8k", json.load(open(f"{HERE}/data/gsm8k_200.json")), gsm_item),
       run("humaneval", json.load(open(f"{HERE}/data/humaneval.json")), he_item)]
open(f"{OUT}/summary.txt","w").write(f"{tag} {MODEL} {URL}\n"+"\n".join(lines)+"\n"); print("SUMMARY", tag, "|", " || ".join(lines))
