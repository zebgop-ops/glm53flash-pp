#!/usr/bin/env python3
"""Build a model dir that is orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 plus the MTP block from
RedHatAI/GLM-5.3-Flash-NVFP4 (byte-identical NVFP4 format; RedHat kept layer 45 with FP8-block experts).
  - symlinks every orcarouter file except config.json / model.safetensors.index.json
  - symlinks RedHat's model_mtp.safetensors
  - config.json: RedHat's config (same base architecture; it also carries the linear_* KDA fields
    orcarouter's config omits, and a quantization_config with group_1 for the layer-45 FP8 experts
    plus an ignore list in the checkpoint's own naming)
  - model.safetensors.index.json: orcarouter's weight_map + the MTP file's tensors
usage: make-nvfp4-mtp-dir.py [outdir]"""
import glob, json, os, struct, sys
HUB = os.path.expanduser("~/.cache/huggingface/hub")
# Symlink targets are written in the CONTAINER's view of the HF cache (the launcher mounts
# ~/.cache/huggingface at /hf and this directory at /model), so they resolve inside the container.
CONTAINER_HUB = "/hf/hub"
def ctarget(host_path):
    rp = os.path.realpath(host_path); assert rp.startswith(HUB), rp
    return CONTAINER_HUB + rp[len(HUB):]
def snap(repo):
    d = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/"), key=os.path.getmtime)
    assert d, f"no snapshot for {repo}"; return d[-1]
ORCA, RH = snap("orcarouter/GLM-5.3-Flash-Uncensored-NVFP4"), snap("RedHatAI/GLM-5.3-Flash-NVFP4")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/r/glm53-run/models/glm53-uncensored-nvfp4-mtp"
os.makedirs(OUT, exist_ok=True)
mtp = os.path.join(RH, "model_mtp.safetensors"); assert os.path.exists(mtp), "MTP file not downloaded yet"
idx = json.load(open(os.path.join(ORCA, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(ORCA, s))]
assert not missing, f"orcarouter shards missing: {missing[:3]} (+{len(missing)-3})"
for f in os.listdir(ORCA):
    if f in ("config.json", "model.safetensors.index.json"): continue
    dst = os.path.join(OUT, f)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(ctarget(os.path.join(ORCA, f)), dst)
dst = os.path.join(OUT, "model_mtp.safetensors")
if os.path.lexists(dst): os.remove(dst)
os.symlink(ctarget(mtp), dst)
with open(mtp, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
mtp_keys = [k for k in hdr if k != "__metadata__"]
assert all(".layers.45." in k for k in mtp_keys), "unexpected non-layer-45 tensor in MTP file"
for k in mtp_keys: idx["weight_map"][k] = "model_mtp.safetensors"
idx.setdefault("metadata", {})["total_size"] = idx.get("metadata", {}).get("total_size", 0) + os.path.getsize(mtp)
json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(RH, "config.json")))
ocfg = json.load(open(os.path.join(ORCA, "config.json")))
tc, otc = cfg.get("text_config", cfg), ocfg.get("text_config", ocfg)
for k in ("num_hidden_layers", "n_routed_experts", "num_nextn_predict_layers", "vocab_size", "hidden_size"):
    assert tc.get(k) == otc.get(k), (k, tc.get(k), otc.get(k))
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print(f"{OUT}: {len(os.listdir(OUT))} entries, index {len(idx['weight_map'])} tensors ({len(mtp_keys)} from MTP), config from RedHat")
