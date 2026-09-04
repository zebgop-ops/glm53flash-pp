"""Find where allocations near the top of memory start faulting instead of OOM-ing.
Base filler B GiB (written), then alloc/write/free cycles of a 2.4 GiB + 0.6 GiB pair (the per-layer
prep transient) while raising B by 0.5 GiB steps. FRACTION (optional) caps PyTorch's allocator."""
import torch, os, sys, time
frac = os.environ.get("FRACTION")
if frac: torch.cuda.set_per_process_memory_fraction(float(frac)); print("allocator fraction", frac, flush=True)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
start = float(os.environ.get("START_GB", "54"))
base = [torch.ones(2**30, dtype=torch.uint8, device="cuda") for _ in range(int(start))]
torch.cuda.synchronize(); half = None
b = start
try:
    while True:
        for _ in range(3):
            big = torch.empty(int(2.4 * 2**30), dtype=torch.uint8, device="cuda"); big.fill_(1)
            sm = torch.empty(int(0.6 * 2**30), dtype=torch.uint8, device="cuda"); sm.fill_(2)
            torch.cuda.synchronize(); del big, sm
        print(f"  base {b:.1f} GiB + 3.0 GiB transient ok (peak {torch.cuda.max_memory_allocated()/2**30:.1f}, reserved {torch.cuda.memory_reserved()/2**30:.1f}, total {total:.1f})", flush=True)
        base.append(torch.ones(2**29, dtype=torch.uint8, device="cuda")); torch.cuda.synchronize(); b += 0.5
except torch.OutOfMemoryError:
    print(f"CLEAN OOM at base {b:.1f} GiB (+3 GiB transient) -> safe behaviour", flush=True)
except Exception as e:
    print(f"FAULT at base {b:.1f} GiB (+3 GiB transient): {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True); sys.exit(1)
