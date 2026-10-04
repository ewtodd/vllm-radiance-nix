// Practical bf16 WMMA ceiling for gfx1201 (R9700), with no memory traffic in the timed loop.
//
// The W16 prefill body is built on `wmma_f32_16x16x16_bf16_w32_gfx12`. This kernel issues that
// instruction back to back on register-resident fragments, with 8 independent accumulator chains
// per lane, so the result is the tensor-pipe issue rate and nothing else: no LDS staging, no
// dequant, no scale, no global reads. It is the ceiling the GEMM body is measured against.
//
// Build (same ROCm/hipcc as the launcher):
//   hipcc -O3 --offload-arch=gfx1201 -std=c++17 wmma_peak.hip -o /tmp/wmma_peak
// Run on one card at a time:
//   HIP_VISIBLE_DEVICES=0 /tmp/wmma_peak [blocks] [iterations] [threads]
//
// FLOPs counted per WMMA instruction: 2 * 16 * 16 * 16 = 8192.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8f wmma_bf16(v8bf a, v8bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// 8 accumulators: if the tensor pipe is ~4 cycles/issue with ~20 cycles latency, ILP 8 keeps it
// full. a/b are read from global once so they cannot be constant-folded.
__global__ __launch_bounds__(1024) void wmma_peak_kernel(const v8bf *__restrict__ ab, float *out,
                                                         long iters) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const v8bf a = ab[(idx * 2) & 31];
  const v8bf b = ab[(idx * 2 + 1) & 31];
  v8f c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0}, c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};
  for (long i = 0; i < iters; ++i) {
    c0 = wmma_bf16(a, b, c0);
    c1 = wmma_bf16(a, b, c1);
    c2 = wmma_bf16(a, b, c2);
    c3 = wmma_bf16(a, b, c3);
    c4 = wmma_bf16(a, b, c4);
    c5 = wmma_bf16(a, b, c5);
    c6 = wmma_bf16(a, b, c6);
    c7 = wmma_bf16(a, b, c7);
  }
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) s += c0[e] + c1[e] + c2[e] + c3[e] + c4[e] + c5[e] + c6[e] + c7[e];
  if (s == 123456.789f) out[idx] = s;   // keep the accumulators live without a real store
}

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
  long iters = argc > 2 ? std::atol(argv[2]) : 20000;
  int threads = argc > 3 ? std::atoi(argv[3]) : 256;

  v8bf *ab = nullptr;
  float *out = nullptr;
  check(hipMalloc(&ab, 32 * sizeof(v8bf)), "hipMalloc ab");
  check(hipMalloc(&out, (size_t)blocks * threads * sizeof(float)), "hipMalloc out");
  check(hipMemset(ab, 0x3c, 32 * sizeof(v8bf)), "hipMemset ab");

  auto launch = [&] { wmma_peak_kernel<<<blocks, threads>>>(ab, out, iters); };
  launch();
  check(hipDeviceSynchronize(), "warmup");

  hipEvent_t t0 = nullptr, t1 = nullptr;
  check(hipEventCreate(&t0), "event t0");
  check(hipEventCreate(&t1), "event t1");
  check(hipEventRecord(t0), "record t0");
  const int reps = 20;
  for (int r = 0; r < reps; ++r) launch();
  check(hipEventRecord(t1), "record t1");
  check(hipEventSynchronize(t1), "sync t1");
  float ms = 0.f;
  check(hipEventElapsedTime(&ms, t0, t1), "elapsed");

  const double wmmas = (double)blocks * (threads / 32) * (double)iters * 8.0 * reps;
  const double flops = wmmas * 2.0 * 16.0 * 16.0 * 16.0;
  const double tflops = flops / (ms * 1e-3) / 1e12;
  std::printf("blocks=%d iters=%ld threads=%d reps=%d  %.3f ms/rep  %.1f TFLOP/s\n", blocks, iters,
              threads, reps, ms / reps, tflops);

  check(hipFree(ab), "free ab");
  check(hipFree(out), "free out");
  return 0;
}
