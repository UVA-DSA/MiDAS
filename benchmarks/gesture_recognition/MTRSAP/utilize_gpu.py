# gpu_sanity_check.py
import time
import torch

def main(seconds=45):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Are you on a GPU node and did you request --gres=gpu:1?")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("Using device:", torch.cuda.get_device_name(device))

    # create big tensors on GPU once (to avoid PCIe overhead each iteration)
    N = 8192  # big enough to light up the GPU; adjust if OOM
    a = torch.randn(N, N, device=device, dtype=torch.float16)
    b = torch.randn(N, N, device=device, dtype=torch.float16)

    # warmup
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t_end = time.time() + seconds
    iters = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    try:
        start.record()
        while time.time() < t_end:
            torch.matmul(a, b)  # compute on GPU
            iters += 1
            if iters % 100 == 0:
                allocated = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                print(f"Iter {iters}: Allocated={allocated:.1f} MiB | Reserved={reserved:.1f} MiB")
        end.record()
        torch.cuda.synchronize()

    except KeyboardInterrupt:
        print("Stopped by user.")

    ms = start.elapsed_time(end)
    print(f"Completed {iters} matmuls in {ms/1000.0:.2f}s (~{iters/(ms/1000.0):.2f} it/s).")
    mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
    mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
    print(f"GPU memory: allocated={mem_alloc:.1f} MiB, reserved={mem_reserved:.1f} MiB")

if __name__ == "__main__":
    time_hr = 3
    seconds = time_hr * 3600
    main(seconds=seconds)
