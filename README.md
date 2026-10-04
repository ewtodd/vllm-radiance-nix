> **Attribution.** This repository is a fork of
> [`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4), which itself builds on
> [`StillDeadcode/vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance). **The vast
> majority of the code, kernels, patches and documentation is theirs** — the full fork history is
> preserved here (`git log`), so every upstream commit is intact. This fork adds a self-contained Nix
> flake (below), the native **W4A16 / W8A16** weight-only GEMMs (`W4A16_W8A16.md`) and the DFlash2
> draft-rope patch. The container workflow has been removed in favour of the flake.

# vllm-radiance-nix

A Nix flake that builds a working vLLM for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)** from
source, with the RDNA4 kernels and patches needed to run it. It serves the same checkpoints as
upstream: FP8, Quark MXFP4, and ParoQuant (INT4 W4A8, INT5 W5A8, MXFP6 W6A8).

## What it builds

Everything from source, against the host's `pkgs`:

| component | version |
|---|---|
| ROCm (TheRock) | 10.0, `gfx120X-all` |
| PyTorch / Triton | nixpkgs ROCm torch (2.13) / 3.7 |
| AITER | 0.1.17, built for gfx1201 |
| vLLM | 0.29.0 |
| transformers | 5.14.1 |
| libr4d | `StillDeadcode/libr4d` @ `b9e42ab` |

Plus the HIP kernels in this repo (`radiance_mxfp4_fp8.hip`, `paroquant/radiance_paroquant.hip`) and
the in-repo patch chain (`patch_*.py`), including the W4A16/W8A16 kernels and the DFlash2 draft-rope
patch. Tested on 2 x R9700, TP=2.

## Nix

### From a NixOS host

```nix
# flake.nix
inputs.vllm-radiance.url = "github:ewtodd/vllm-radiance-nix";

# a module
{ pkgs, inputs, ... }:
let
  stack = inputs.vllm-radiance.lib.${pkgs.stdenv.hostPlatform.system}.mkVllmStack {
    inherit pkgs;   # reuse the host's nixpkgs; libr4d comes from the flake
  };
in
{
  environment.systemPackages = [ stack.pythonEnv ];
}
```

`mkVllmStack` returns `pythonEnv`, `vllm`, `aiter`, `torch`, `libr4d`, `rocmSdk`, `rocmSdkCc`,
`runtimeLibs`, `gfxArch`.

### Standalone

```bash
nix build                                # the vLLM python environment (packages.default)
nix develop                              # shell with vllm + the radiance kernels on PATH
nix shell .#pythonEnv -c vllm -- --help  # or: vllm serve <model> ...
```

### Binary cache

The heavy closures (ROCm SDK, PyTorch, AITER, vLLM) are prebuilt on the house cache — `nix-serve-ng`
on e-desktop, behind Caddy at `https://cache.ethanwtodd.com`:

```nix
nix.settings = {
  substituters = [ "https://cache.ethanwtodd.com" ];
  trusted-public-keys = [ "e-desktop:35K0AY3HcDOSHVQ/lklmbvmrXjIspM/LYf7yek5lyVA=" ];
};
```

Cache entries are keyed on the source rev and nixpkgs, so divergent checkouts rebuild only what
changed.

## Serving

The NixOS module sets the `RADIANCE_*` environment and the vLLM flags. A representative target-only
run:

```bash
export RADIANCE_GFX_ARCH=gfx1201 RADIANCE_USE_R4D=1 RADIANCE_USE_R4D_GDN=1 \
       RADIANCE_SKINNY_GEMM=1 RADIANCE_FAST_DRAFT=1 RADIANCE_QUARK_BF16_MTP=1 \
       RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A16=1 RADIANCE_MXFP4_WPERM=1

vllm serve /path/to/checkpoint \
  --tensor-parallel-size 2 --max-model-len 262144 \
  --attention-backend R4D --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --enable-prefix-caching --mamba-cache-mode align
```

## Formats and environment gates

| checkpoint | activation path | gates |
|---|---|---|
| Quark MXFP4, W4A8 | 4-bit weights, fp8 WMMA | `RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1` |
| Quark MXFP4, W4A16 | 4-bit weights, **bf16** | `RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A16=1 RADIANCE_MXFP4_WPERM=1` (weight-only config, no `input_tensors`) |
| FP8, W8A8 | fp8 dynamic per-token | default |
| FP8, W8A16 | **bf16** | `RADIANCE_FP8_W8A16=1` |
| ParoQuant INT4/INT5/MXFP6 | fp8 WMMA W4A8/W5A8/W6A8 | `quant_method` from the checkpoint (`RADIANCE_PQ_*` profile) |

Drop `--quantization` for the MXFP4/FP8 checkpoints: the runtime reads the method from
`config.json`.

## Documentation

- `W4A16_W8A16.md` — the native weight-only kernels, roofline and context measurements
- `PAROQUANT.md` — ParoQuant formats, kernels, knobs
- `PERFORMANCE.md` — the measured optimization ledger
- `MXFP4-NOTES.md`, `TP3_PADDING_PLAN.md`, `kv-profiles.tsv` — deeper notes

## Provenance and license

Forked from [`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) and
[`StillDeadcode/vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance); vLLM itself is
Apache-2.0. See `LICENSE` and the upstream repositories for the terms that apply to their work.
