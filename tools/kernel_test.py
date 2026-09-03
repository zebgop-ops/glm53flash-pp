import os, sys, torch, time
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton
dev = torch.device("cuda:0")
M, H, D = 2048, 32, 128
for N in [int(x) for x in (sys.argv[1:] or ["20000", "32768", "33000", "40000", "50000", "65536"])]:
    q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(N, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    ks = torch.zeros(M, dtype=torch.int32, device=dev)
    ke = torch.full((M,), N, dtype=torch.int32, device=dev)
    scales = torch.rand(N, device=dev, dtype=torch.float32)
    w = torch.rand(M, H, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = fp8_mqa_logits_triton(q, (k, scales), w, ks, ke, clean_logits=False)
        torch.cuda.synchronize()
        print(f"N={N}: OK  out {tuple(out.shape)} finite={bool(torch.isfinite(out).all())} {time.time()-t0:.2f}s", flush=True)
    except Exception as e:
        print(f"N={N}: FAULT {type(e).__name__}: {str(e)[:120]}", flush=True); break
