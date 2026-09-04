#!/usr/bin/env python3
"""Isolate the boot-time fault of GLM53U (NVFP4 experts on sm_80): the first MoE layer's
process_weights_after_loading -> prepare_nvfp4_moe_layer_for_marlin -> per-expert
ops.gptq_marlin_repack died with an illegal memory access on every rank.
Runs INSIDE the image on ONE GPU with CUDA_LAUNCH_BLOCKING=1 so the faulting kernel is named.
Stages (each isolated, real GLM-5.3-Flash geometry E=288, K=4096, N=2048):
  A  dense gptq_marlin_repack (the call the FP4 MoE path makes), random data, w13 and w2 shapes
  B  batched gptq_marlin_moe_repack (the call the int4 GPTQ MoE path makes; known good here)
  C  prepare_nvfp4_moe_layer_for_marlin on a fake layer, random data
  D  same with the REAL layer-3 expert tensors from /model (needs -v composite dir at /model)
  F  prepare_moe_fp8_layer_for_marlin with the drafter's FP8-block geometry (layer 45)
usage: docker run --rm -i --gpus '"device=0"' -e CUDA_LAUNCH_BLOCKING=1 -v ~/.cache/huggingface:/hf:ro \
         -v /home/r/glm53-run/models/glm53-uncensored-nvfp4-mtp:/model:ro --entrypoint python3 \
         vllm/vllm-openai:glm53-flash - < nvfp4_marlin_test.py [stages]"""
import os, sys, time, traceback, json, struct
import torch
stages = sys.argv[1] if len(sys.argv) > 1 else "ABCDF"
dev = torch.device("cuda:0")
E, K, N = 288, 4096, 2048
def stage(name, fn):
    t = time.time()
    try:
        r = fn(); torch.cuda.synchronize()
        print(f"[{name}] OK {time.time()-t:.1f}s {r if r is not None else ''}", flush=True)
    except Exception as e:
        print(f"[{name}] FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
        traceback.print_exc(limit=4); sys.exit(1)   # a CUDA fault poisons the context: stop here
from vllm import _custom_ops as ops
def A():
    perm = torch.empty(0, dtype=torch.int, device=dev)
    for tag, size_n, size_k in (("w13", 2 * N, K), ("w2", K, N)):
        w = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=dev)
        q = w.view(torch.int32).T.contiguous()          # [size_k/8, size_n]
        out = ops.gptq_marlin_repack(q, perm, size_k, size_n, 4, False)
        print(f"   dense repack {tag}: in {tuple(q.shape)} -> out {tuple(out.shape)}", flush=True)
    return "dense gptq_marlin_repack fine"
def B():
    perm = torch.empty((E, 0), dtype=torch.int, device=dev)
    for tag, size_n, size_k in (("w13", 2 * N, K), ("w2", K, N)):
        w = torch.randint(0, 256, (E, size_n, size_k // 2), dtype=torch.uint8, device=dev)
        q = w.view(torch.int32).transpose(1, 2).contiguous()   # [E, size_k/8, size_n]
        out = ops.gptq_marlin_moe_repack(q, perm, size_k, size_n, 4, False)
        print(f"   batched moe repack {tag}: in {tuple(q.shape)} -> out {tuple(out.shape)}", flush=True)
    return "batched gptq_marlin_moe_repack fine"
class FakeLayer:
    def __init__(self):
        self.num_experts, self.hidden_size, self.intermediate_size_per_partition = E, K, N
        self.params_dtype = torch.bfloat16; self.workspace = None
def prep(w13, w13s, w13g, w2, w2s, w2g):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    out = prepare_nvfp4_moe_layer_for_marlin(layer=FakeLayer(), w13=w13, w13_scale=w13s, w13_scale_2=w13g,
                                             w2=w2, w2_scale=w2s, w2_scale_2=w2g, is_act_and_mul=True)
    return "shapes " + str([tuple(t.shape) for t in out])
def C():
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
    w13s = (torch.rand(E, 2 * N, K // 16, device=dev) * 2).to(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
    w2s = (torch.rand(E, K, N // 16, device=dev) * 2).to(torch.float8_e4m3fn)
    g13 = torch.rand(E, device=dev) + 0.5; g2 = torch.rand(E, device=dev) + 0.5
    return prep(w13, w13s, 1.0 / g13, w2, w2s, 1.0 / g2)
def load_layer3():
    from safetensors import safe_open
    idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
    pre = "model.language_model.layers.3.mlp.experts."
    names = {f"{pre}{e}.{m}.{s}": (e, m, s) for e in range(E) for m in ("gate_proj", "up_proj", "down_proj")
             for s in ("weight_packed", "weight_scale", "weight_global_scale")}
    byfile = {}
    for n in names: byfile.setdefault(idx[n], []).append(n)
    T = {}
    for f, ns in byfile.items():
        with safe_open("/model/" + f, framework="pt", device="cpu") as fh:
            for n in ns: T[names[n]] = fh.get_tensor(n)
    w13 = torch.stack([torch.cat([T[(e, "gate_proj", "weight_packed")], T[(e, "up_proj", "weight_packed")]], 0) for e in range(E)]).to(dev)
    w13s = torch.stack([torch.cat([T[(e, "gate_proj", "weight_scale")], T[(e, "up_proj", "weight_scale")]], 0) for e in range(E)]).to(dev)
    w2 = torch.stack([T[(e, "down_proj", "weight_packed")] for e in range(E)]).to(dev)
    w2s = torch.stack([T[(e, "down_proj", "weight_scale")] for e in range(E)]).to(dev)
    g13 = torch.stack([T[(e, "gate_proj", "weight_global_scale")].reshape(()) for e in range(E)]).to(dev)
    g2 = torch.stack([T[(e, "down_proj", "weight_global_scale")].reshape(()) for e in range(E)]).to(dev)
    print(f"   real tensors: w13 {tuple(w13.shape)} {w13.dtype}, w13s {tuple(w13s.shape)} {w13s.dtype}, w2 {tuple(w2.shape)}, g13 range {g13.min().item():.3g}..{g13.max().item():.3g}", flush=True)
    return w13, w13s, g13, w2, w2s, g2
def D():
    w13, w13s, g13, w2, w2s, g2 = load_layer3()
    return prep(w13, w13s, 1.0 / g13, w2, w2s, 1.0 / g2)
def F():
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
    class L(torch.nn.Module):
        pass
    layer = L(); layer.num_experts, layer.hidden_size, layer.intermediate_size_per_partition = E, K, N
    layer.params_dtype = torch.bfloat16; layer.weight_block_size = [128, 128]
    layer.w13_weight = torch.nn.Parameter((torch.randn(E, 2 * N, K, device=dev) * 0.05).to(torch.float8_e4m3fn), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter((torch.randn(E, K, N, device=dev) * 0.05).to(torch.float8_e4m3fn), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(torch.rand(E, 2 * N // 128, K // 128, device=dev), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.rand(E, K // 128, N // 128, device=dev), requires_grad=False)
    prepare_moe_fp8_layer_for_marlin(layer, size_k_first=True) if "size_k_first" in prepare_moe_fp8_layer_for_marlin.__code__.co_varnames else prepare_moe_fp8_layer_for_marlin(layer)
    return f"fp8 marlin moe prep fine: w13 {tuple(layer.w13_weight.shape)} w2 {tuple(layer.w2_weight.shape)}"
print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}  stages={stages}", flush=True)
for s in stages:
    stage(s, {"A": A, "B": B, "C": C, "D": D, "F": F}[s])
print("ALL REQUESTED STAGES PASSED")
