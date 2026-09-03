#!/usr/bin/env python3
"""Suggest per-rank KV budgets (GLM53_KV0..3, bytes) from a boot's log.
Reads 'Available KV cache memory: X GiB' per Worker_PP<i> from `docker logs glm53flash-pp`,
subtracts a safety margin (default 0.5 GiB; 1.0 GiB on the last rank, which hosts the MTP
drafter and its transient scratch), and prints the env assignment plus the expected gain vs
the tightest-rank value that vLLM applies to every rank by default."""
import re, subprocess, sys
GiB = 2**30
margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
last_margin = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
log = subprocess.run(["docker", "logs", "glm53flash-pp"], capture_output=True, text=True).stdout + \
      subprocess.run(["docker", "logs", "glm53flash-pp"], capture_output=True, text=True).stderr
avail = {}
# vLLM logs 'Available KV cache memory' on rank 0 only; every rank logs its own
# 'Replace gpu_memory_utilization config with `--kv-cache-memory=<bytes>`' suggestion.
# Each rank's line carries two suggestions: the util-equivalent value first and the
# "maximize KV cache memory" value (all free memory beyond weights+activation) last.
util_eq = {}
for m in re.finditer(r"Worker_PP(\d)([^\n]*)", log):
    vals = re.findall(r"--kv-cache-memory=(\d+)", m.group(2))
    if vals:
        r = int(m.group(1))
        util_eq[r] = int(vals[0]) / GiB
        avail[r] = int(vals[-1]) / GiB
if util_eq:
    print("util-equivalent KV per rank (GiB, what the current boot uses on the tightest rank):",
          {r: round(v, 2) for r, v in sorted(util_eq.items())})
consumed = {int(m.group(1)): float(m.group(2)) for m in re.finditer(
    r"Worker_PP(\d)[^\n]*Actual usage is ([0-9.]+) GiB for consumed memory", log)}
if consumed:
    print("consumed (weights+non-torch) per rank (GiB):", consumed)
if len(avail) < 2:
    sys.exit("no per-rank 'Available KV cache memory' lines found (boot not past profiling?)")
ranks = sorted(avail)
last = ranks[-1]
print("MAXIMUM KV per rank if all free memory is used (GiB):", {r: round(avail[r], 2) for r in ranks})
tight = min(avail.values())
budget = {r: avail[r] - (last_margin if r == last else margin) for r in ranks}
print("suggested budgets (GiB):", {r: round(budget[r], 2) for r in ranks})
print("env:", " ".join(f"GLM53_KV{r}={int(budget[r]*GiB)}" for r in ranks))
print(f"default pool is bounded by the tightest rank ({tight:.2f} GiB on every rank); "
      f"per-rank budgets add {sum(budget.values()) - tight*len(ranks):+.2f} GiB total, "
      f"but the pool grows only as much as the NEW tightest rank ({min(budget.values()):.2f} GiB) allows "
      f"unless vLLM sizes the pool per rank.")
