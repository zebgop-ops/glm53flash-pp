import os, sys, torch, time, itertools
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import triton
from vllm.v1.attention.ops import mqa_logits_triton as mod
kern = mod._fp8_mqa_logits_kernel           # Autotuner object
fn = kern.fn                                 # underlying JITFunction
configs = kern.configs
print("configs:", [(c.kwargs, c.num_warps, c.num_stages) for c in configs], flush=True)
dev = torch.device("cuda:0"); M, H, D = 2048, 32, 128
def run(N, cfg, causal=True):
    q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N, D, device=dev, dtype=torch.bfloat16)
    scales = torch.rand(N, device=dev); w = torch.rand(M, H, device=dev)
    base = N - M // 4 - 8
    ks = torch.zeros(M, dtype=torch.int32, device=dev)
    ke = (torch.arange(M, device=dev, dtype=torch.int32) // 4 + base + 1).clamp(max=N) if causal else torch.full((M,), N, dtype=torch.int32, device=dev)
    logits = torch.empty((M, N), dtype=torch.float32, device=dev)
    BLOCK_H = max(16, triton.next_power_of_2(H)); BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = cfg.kwargs["BLOCK_N"]
    grid = (M, triton.cdiv(N, BLOCK_N))
    fn[grid](q, k, scales, w, ks, ke, logits, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
             w.stride(0), w.stride(1), logits.stride(0), logits.stride(1),
             num_heads=H, head_dim=D, N=N, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
             num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    torch.cuda.synchronize()
    return logits
for N in [int(x) for x in (sys.argv[1:] or ["30000", "50000", "65536"])]:
    for cfg in configs:
        try:
            out = run(N, cfg)
            print(f"N={N} cfg={cfg.kwargs} w{cfg.num_warps} s{cfg.num_stages}: OK", flush=True)
        except Exception as e:
            print(f"N={N} cfg={cfg.kwargs} w{cfg.num_warps} s{cfg.num_stages}: FAULT {str(e)[:100]}", flush=True)
            sys.exit(1)
