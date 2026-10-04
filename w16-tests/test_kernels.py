"""Isolated unit test for the native W4A16 / W8A16 bf16-WMMA kernels.

Run on the R9700 box with the vLLM python env and a freshly compiled
radiance_mxfp4_fp8.so on PYTHONPATH (or passed as argv[1]).  For every linear
shape of the 27B model and M in {1, 4, 32, 256, 2048} it compares the kernel
against a float32 dequantized reference and asserts the norm-relative error is
below 1e-2 (the CHECKALL bar).  Both W4A16 weight layouts are exercised.
"""

import argparse
import importlib.util
import sys

import torch

# Every language-model linear shape in the 27B model at TP=2, read off a real serve's
# "[radiance.mxfp4] W4A16 layer N=.. K=.." / "[radiance.fp8] W8A16 layer" lines. The fp8
# checkpoint leaves the N=48 gate layer unquantized, so it is a W4A16-only shape.
SHAPES = [
    (17408, 5120),
    (8192, 5120),
    (7168, 5120),
    (5120, 8704),
    (5120, 3072),
    (48, 5120),
]
MS = [1, 4, 32, 256, 2048]
TOL = 1e-2


def load_ext(path):
    spec = importlib.util.spec_from_file_location("radiance_mxfp4_fp8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permute_w16(packed, n, k):
    """Checkpoint [N,K/2] uint8 -> bf16 fragment slots, [N,K/2] uint8.

    Slot (nt, ks, lane=half*16+row) holds bytes {2h,2h+1} of the k-step's first
    8 elements and {2h,2h+1} of the second, i.e. exactly the four bytes the
    kernel expands into lane l's two 4-element k runs."""
    nt, ks = n // 16, k // 16
    p = packed.view(nt, 16, ks, 2, 2, 2)      # [nt][row][ks][group][pair][byte]
    out = torch.empty(nt, ks, 2, 16, 4, dtype=torch.uint8, device=packed.device)
    out[:, :, :, :, 0:2] = p[:, :, :, 0, :, :].permute(0, 2, 3, 1, 4)
    out[:, :, :, :, 2:4] = p[:, :, :, 1, :, :].permute(0, 2, 3, 1, 4)
    return out.contiguous().view(n, k // 2)


E2M1 = None


def dequant_mxfp4(weight, scale):
    global E2M1
    if E2M1 is None or E2M1.device != weight.device:
        E2M1 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=weight.device)
    n, kh = weight.shape
    k = kh * 2
    codes = torch.empty(n, k, dtype=torch.long, device=weight.device)
    codes[:, 0::2] = weight & 0x0F
    codes[:, 1::2] = (weight >> 4) & 0x0F
    values = E2M1[codes]
    scales = torch.pow(2.0, scale.to(torch.int32).float() - 127.0)
    return values * scales.T.repeat_interleave(32, dim=1)


def dequant_fp8(weight, scale):
    n, k = weight.shape
    blocks_n, blocks_k = scale.shape
    assert blocks_n * 128 >= n and blocks_k * 128 >= k
    expanded = scale.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)[:, :k]
    return weight.float() * expanded


def launch_w4a16(ext, a, w, ws, wperm, scratch):
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    ext.launch_w4a16(a.data_ptr(), w.data_ptr(), ws.data_ptr(), out.data_ptr(),
                     scratch.data_ptr(), m, n, k, int(wperm),
                     torch.cuda.current_stream().cuda_stream)
    return out


def launch_w8a16(ext, a, w, ws, scratch):
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    ext.launch_w8a16(a.data_ptr(), w.data_ptr(), ws.data_ptr(), out.data_ptr(),
                     scratch.data_ptr(), m, n, k, torch.cuda.current_stream().cuda_stream)
    return out


def rel_error(out, ref):
    out = out.float()
    return ((out - ref).norm() / ref.norm().clamp_min(1e-9)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("so", nargs="?", default="radiance_mxfp4_fp8.so")
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--ms", default="1,4,32,256,2048")
    args = parser.parse_args()

    torch.manual_seed(1234)
    ext = load_ext(args.so)
    shapes = SHAPES if args.shapes == "all" else [
        tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    ms = [int(v) for v in args.ms.split(",")]
    scratch = torch.empty(4 * 16 * 32768, dtype=torch.float32, device="cuda")

    failures = 0
    print("%-13s %6s %8s  %-10s %-10s" % ("shape N x K", "M", "wperm", "w4a16 rel", "tol"))
    for n, k in shapes:
        for m in ms:
            a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
            scale = torch.randint(120, 136, (k // 32, n), dtype=torch.uint8, device="cuda")
            ref = a.float() @ dequant_mxfp4(codes, scale).T
            for wperm in (0, 1):
                w = permute_w16(codes, n, k) if wperm else codes
                out = launch_w4a16(ext, a, w, scale, wperm, scratch)
                rel = rel_error(out, ref)
                ok = rel < TOL
                failures += 0 if ok else 1
                print("%-13s %6d %8d  %-10.3e %s" % (
                    "%d x %d" % (n, k), m, wperm, rel, "ok" if ok else "FAIL"))
        torch.cuda.empty_cache()

    print()
    print("%-13s %6s %8s  %-10s %-10s" % ("shape N x K", "M", "w8a16", "rel", "tol"))
    for n, k in shapes:
        for m in ms:
            a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            ws = (torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda") * 0.05
                  + 0.01).float()
            ref = a.float() @ dequant_fp8(w, ws).T
            out = launch_w8a16(ext, a, w, ws, scratch)
            rel = rel_error(out, ref)
            ok = rel < TOL
            failures += 0 if ok else 1
            print("%-13s %6d %8s  %-10.3e %s" % (
                "%d x %d" % (n, k), m, "fp8", rel, "ok" if ok else "FAIL"))
        torch.cuda.empty_cache()

    print()
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
