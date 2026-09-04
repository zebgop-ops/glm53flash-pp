import torch, os, sys, time
print("alloc conf:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "| total:", torch.cuda.get_device_properties(0).total_memory/2**30, "GiB", flush=True)
chunks=[]; t=time.time()
try:
    for i in range(70):
        c=torch.empty(2**30, dtype=torch.uint8, device="cuda"); c.fill_(i & 0xFF); torch.cuda.synchronize(); chunks.append(c)
        if (i+1) % 8 == 0: print(f"  {i+1} GiB written ok", flush=True)
except torch.OutOfMemoryError as e:
    print(f"clean OOM after {len(chunks)} GiB (expected near the top)", flush=True)
except Exception as e:
    print(f"FAULT after {len(chunks)} GiB: {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True); sys.exit(1)
# verify the data survived (read back a sample of each chunk)
bad=[i for i,c in enumerate(chunks) if int(c[::2**20].to(torch.int64).sum()) != (i & 0xFF)*1024]
print(f"readback: {len(chunks)} chunks, {len(bad)} corrupted {bad[:5]}; {time.time()-t:.0f}s", flush=True)
