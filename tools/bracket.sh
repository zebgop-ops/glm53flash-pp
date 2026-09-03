#!/bin/bash
# Step the needle length up, stop at the first failure; report health after each step.
# usage: bracket.sh [lengths...]   (default 30000 60000 100000 150000 200000 250000)
cd "$(dirname "$0")"
LENS=${@:-30000 60000 100000 150000 200000 250000}
for L in $LENS; do
  echo "===== needle $L  ($(date +%T)) ====="
  timeout 2400 python3 needle.py "$L"; rc=$?
  H=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:8002/health)
  X=$(journalctl -k --since "3 min ago" --no-pager 2>/dev/null | grep -c Xid)
  BC=$(docker logs --since 3m glm53flash-pp 2>&1 | grep -m1 "GLM53_BOUNDS_CHECK" | cut -c1-300)
  echo "rc=$rc health=$H xid_recent=$X ${BC:+bounds_check: $BC}"
  if [ "$rc" != "0" ] || [ "$H" != "200" ] || [ "$X" != "0" ]; then echo "STOP at $L"; break; fi
done
docker logs --since 30m glm53flash-pp 2>&1 | grep -iE "illegal|AcceleratorError|BOUNDS_CHECK|Traceback" | head -5
