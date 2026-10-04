"""Kernel-only timing for the W4A16/W8A16 GEMMs, per model shape and M.

No model, no server: loads radiance_mxfp4_fp8.so and times launch_w4a16/launch_w8a16 directly.
Used to tune the decode/prefill tile without paying a model load per iteration.
"""

import argparse
import importlib.util
import json

import torch

SHAPES = [
    (17408, 5120),
    (8192, 5120),
    (7168, 5120),
    (5120, 8704),
    (5120, 3072),
    (48, 5120),
]


def load_ext(path):
    spec = importlib.util.spec_from_file_location("radiance_mxfp4_fp8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permute_w16(packed, n, k):
    nt, ks = n // 16, k // 16
    p = packed.view(nt, 16, ks, 2, 2, 2)
    out = torch.empty(nt, ks, 2, 16, 4, dtype=torch.uint8, device=packed.device)
    out[:, :, :, :, 0:2] = p[:, :, :, 0, :, :].permute(0, 2, 3, 1, 4)
    out[:, :, :, :, 2:4] = p[:, :, :, 1, :, :].permute(0, 2, 3, 1, 4)
    return out.contiguous().view(n, k // 2)


def time_call(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1000.0   # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("so")
    parser.add_argument("--ms", default="1,8,2048")
    parser.add_argument("--wperm", default="0,1")
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    ext = load_ext(args.so)
    torch.manual_seed(0)
    scratch = torch.empty(4 * 16 * 32768, dtype=torch.float32, device="cuda")
    shapes = SHAPES if args.shapes == "all" else [
        tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    ms = [int(v) for v in args.ms.split(",")]
    wperms = [int(v) for v in args.wperm.split(",")]

    out = {}
    for n, k in shapes:
        a = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        scale = torch.randint(120, 136, (k // 32, n), dtype=torch.uint8, device="cuda")
        w8 = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        ws8 = (torch.rand(n // 128 + 1, k // 128 + 1, device="cuda") * 0.05 + 0.01).float()
        for m in ms:
            av = a[:m].contiguous()
            for wperm in wperms:
                w = permute_w16(codes, n, k) if wperm else codes
                y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                us = time_call(lambda: ext.launch_w4a16(
                    av.data_ptr(), w.data_ptr(), scale.data_ptr(), y.data_ptr(),
                    scratch.data_ptr(), m, n, k, wperm,
                    torch.cuda.current_stream().cuda_stream), reps=args.reps)
                key = "w4a16 n=%d k=%d m=%d wperm=%d" % (n, k, m, wperm)
                out[key] = us
                print("%-40s %9.1f us" % (key, us))
            y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            us = time_call(lambda: ext.launch_w8a16(
                av.data_ptr(), w8.data_ptr(), ws8.data_ptr(), y.data_ptr(),
                scratch.data_ptr(), m, n, k,
                torch.cuda.current_stream().cuda_stream), reps=args.reps)
            key = "w8a16 n=%d k=%d m=%d" % (n, k, m)
            out[key] = us
            print("%-40s %9.1f us" % (key, us))
        del codes, scale, w8, ws8, a
        torch.cuda.empty_cache()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
