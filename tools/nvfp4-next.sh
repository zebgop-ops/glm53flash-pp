#!/bin/bash
# One decisive post-reboot sequence for GLM53U (run on a CLEAN driver only; any GPU fault degrades
# every card until the next reboot, so results after a fault are meaningless):
#   1. replica under the ceiling: 36 GiB filler + full Marlin prep of layers 3-12 (real tensors,
#      overlay's chunked scale factor) on GPU 3, synchronous launches.  Pass -> the overlay is
#      sound and rank 0's ~48 GiB resident + ~4.5 GiB transient is far from the ~63 GiB fault zone.
#   2. only then: boot GLM53U (async, GLM53_MARLIN_DIAG=1 logs the scale tensors per layer),
#      wait for health, smoke test.
set -uo pipefail
cd /home/r/glm53-run
OV=/home/r/glm53-run/patch/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
echo "== step 1: replica (GPU 3, FILL 36, layers 3-12) =="
timeout 1500 docker run --rm -i --gpus '"device=3"' -e CUDA_LAUNCH_BLOCKING=1 -e FILL_GB=36 -e LAYERS=3-12 \
  -v /home/r/.cache/huggingface:/hf:ro -v /home/r/glm53-run/models/glm53-uncensored-nvfp4-mtp:/model:ro \
  -v $OV:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:ro \
  --entrypoint python3 vllm/vllm-openai:glm53-flash - G < nvfp4_marlin_test.py 2>&1 | grep -v "^INFO\|^WARNING\|^\[W\|real tensors:" | tail -14 | tee stageG-clean.log
grep -q "ALL REQUESTED STAGES PASSED" stageG-clean.log || { echo "STEP 1 FAILED -> stop (driver likely degraded now)"; exit 1; }
echo "== step 2: boot GLM53U =="
: > launch-nvfp4.log; GLM53_MARLIN_DIAG=1 ./run-glm53flash-uncensored-nvfp4-pp4.sh > launch-nvfp4.log 2>&1; echo "launched: $(tail -1 launch-nvfp4.log)"
for i in $(seq 1 240); do
  curl -sf -o /dev/null localhost:8003/health && { echo "UP after ~$((i*5))s"; break; }
  docker ps --format '{{.Names}}' | grep -q '^glm53nvfp4-pp$' || { echo "CONTAINER EXITED after ~$((i*5))s"; break; }
  docker logs glm53nvfp4-pp 2>&1 | grep -q "illegal memory\|AcceleratorError" && { echo "FAULT after ~$((i*5))s"; break; }
  sleep 5
done
docker logs glm53nvfp4-pp 2>&1 | grep -E "GPU KV cache size|MoE backend|nvfp4 scale-factor input|Model loading took|Loading weights took|illegal memory|Error" | sed -E 's/.*\] //' | cut -c1-200 | awk '!seen[$0]++' | head -24
curl -sf -o /dev/null localhost:8003/health && GLM53_URL=http://localhost:8003/v1/chat/completions python3 - <<'PY'
import json, urllib.request, os
URL=os.environ["GLM53_URL"]
for q in ["What is 17*23? Answer with the number only.", "Name the capital of Australia in one word."]:
    body={"model":"GLM53Flash-Uncensored","messages":[{"role":"user","content":q}],"max_tokens":300,"temperature":0}
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600))
    print("smoke:", repr((r["choices"][0]["message"].get("content") or "").strip()[:60]), r["usage"]["completion_tokens"], "tok")
PY
