#!/usr/bin/env python3
"""Print MTP acceptance from the vLLM /metrics counters (stdlib only)."""
import sys, urllib.request, re
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
txt = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
def tot(name):
    return sum(float(m.group(1)) for m in re.finditer(r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)', txt, re.M))
drafts = tot("vllm:spec_decode_num_drafts_total"); draft_tok = tot("vllm:spec_decode_num_draft_tokens_total")
acc = tot("vllm:spec_decode_num_accepted_tokens_total"); gen = tot("vllm:generation_tokens_total")
per_pos = {m.group(1): float(m.group(2)) for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)', txt, re.M)}
if drafts:
    print(f"drafts={drafts:.0f} draft_tokens={draft_tok:.0f} accepted={acc:.0f} generated={gen:.0f}")
    print(f"acceptance rate={acc/max(draft_tok,1):.1%}  mean accepted/draft={acc/drafts:.2f}  tokens/step~{1+acc/drafts:.2f}")
    if per_pos: print("per-position:", {k: f"{v/drafts:.0%}" for k, v in sorted(per_pos.items(), key=lambda x: int(x[0]))})
else:
    print("no spec-decode counters yet (drafts=0)")
