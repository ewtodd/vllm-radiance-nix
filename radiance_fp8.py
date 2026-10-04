"""RADIANCE gfx1201 W8A16: e4m3 weights x bf16 activations with 128x128 block scales.

vLLM 0.29 routes block-FP8 checkpoints through _POSSIBLE_FP8_BLOCK_KERNELS[ROCM], whose
entries all quantize the activation (W8A8), and _POSSIBLE_WFP8A16_KERNELS[ROCM] is empty.
With RADIANCE_FP8_W8A16=1 the radiance_kernels runtime hook puts RadianceFp8W8A16Kernel at
the head of both lists; the kernel consumes the layer's own bf16 activation and dequantizes
the e4m3 weight tile to bf16 in the native kernel's LDS staging pass. Gated, default off.

Layouts consumed here (what Fp8LinearMethod / CompressedTensorsW8A16Fp8 leave on the layer):
  weight           [N, K]   float8_e4m3fn
  weight_scale_inv [N/128, K/128]  fp32 (the loader casts the checkpoint's bf16)
"""

import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_FP8_W8A16", "0") == "1"

try:
    import radiance_mxfp4_fp8 as _ext
except Exception as _e:                       # extension missing: nothing to register
    _ext = None
    if ENABLED:
        sys.stderr.write(f"[radiance.fp8] W8A16 ext import failed, disabled: {_e!r}\n")

_OPS_REGISTERED = [False]
# Split-K partials [4][16][max N] fp32 for the W8A16 decode tile, allocated at weight load.
_W8A16_SCRATCH = [None, 0]


def _w8a16_ensure_scratch(device, n: int) -> None:
    need = max(int(n), 32768)
    if _W8A16_SCRATCH[0] is None or _W8A16_SCRATCH[1] < need:
        _W8A16_SCRATCH[0] = torch.empty(4 * 16 * need, dtype=torch.float32, device=device)
        _W8A16_SCRATCH[1] = need
        sys.stderr.write(f"[radiance.fp8] W8A16 split-K scratch "
                         f"{_W8A16_SCRATCH[0].numel() * 4 >> 20} MiB (N<={need})\n")


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return bool(on_gfx12x())
    except Exception:
        return False


def _register_ops():
    if _OPS_REGISTERED[0]:
        return

    @torch.library.custom_op("radiance::fp8_w8a16_mm", mutates_args=())
    def fp8_w8a16_mm(a: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        M, K = a.shape
        N = w.shape[0]
        if _W8A16_SCRATCH[0] is None:
            raise RuntimeError("radiance.fp8: W8A16 split-K scratch was never allocated; "
                               "process_weights_after_loading must run before the first forward")
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        _ext.launch_w8a16(a.data_ptr(), w.data_ptr(), ws.data_ptr(), out.data_ptr(),
                          _W8A16_SCRATCH[0].data_ptr(), M, N, K,
                          torch.cuda.current_stream().cuda_stream)
        return out

    @fp8_w8a16_mm.register_fake
    def _(a, w, ws):
        return torch.empty((a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16)

    _OPS_REGISTERED[0] = True


def _make_kernel_class():
    """Built lazily so importing this module never drags in vllm.model_executor.kernels."""
    from vllm.model_executor.kernels.linear.scaled_mm import Fp8BlockScaledMMLinearKernel
    from vllm.model_executor.utils import replace_parameter

    class RadianceFp8W8A16Kernel(Fp8BlockScaledMMLinearKernel):
        """e4m3 weights x bf16 activations, 128x128 block scales, native bf16-WMMA."""

        # The activation stays bf16: apply_weights must not build the fp8 placeholder.
        apply_input_quant = False

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not ENABLED:
                return False, "RADIANCE_FP8_W8A16 is not enabled"
            if _ext is None or not hasattr(_ext, "launch_w8a16"):
                return False, "the W8A16 kernel is missing from radiance_mxfp4_fp8"
            if not _on_gfx12x():
                return False, "the radiance W8A16 kernel is compiled for gfx12x only"
            return True, None

        @classmethod
        def can_implement(cls, config):
            if not ENABLED:
                return False, "RADIANCE_FP8_W8A16 is not enabled"
            weight_key = config.weight_quant_key
            if weight_key is None or weight_key.dtype != torch.float8_e4m3fn:
                return False, "only supports e4m3 weights"
            shape = weight_key.scale.group_shape
            if (int(shape.row), int(shape.col)) != (128, 128):
                return False, f"only supports 128x128 block scales, got ({shape.row}, {shape.col})"
            return True, None

        def __init__(self, config, layer_param_names=None):
            # layer_param_names is passed by init_wfp8_a16_linear_kernel (compressed-tensors)
            # and omitted by init_fp8_linear_kernel (Fp8Config); both reach the same kernel.
            super().__init__(config)
            self.layer_param_names = layer_param_names

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            weight = layer.weight.data
            scale = getattr(layer, "weight_scale_inv", None)
            scale_name = "weight_scale_inv"
            if scale is None:
                scale = layer.weight_scale.data
                scale_name = "weight_scale"
            N, K = int(weight.shape[0]), int(weight.shape[1])
            if weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"[radiance.fp8] W8A16 needs e4m3 weights, got {weight.dtype}")
            if K % 128 or N % 16:
                raise RuntimeError(f"[radiance.fp8] W8A16 shape N={N} K={K} unsupported")
            # Keep the checkpoint's [N, K] and [N/128, K/128] fp32 layouts; skip the base
            # class's VLLM_ROCM_FP8_PADDING view, which would make weight.stride(0) != K.
            replace_parameter(layer, "weight", weight.contiguous())
            replace_parameter(layer, scale_name, scale.float().contiguous())
            _w8a16_ensure_scratch(weight.device, N)
            layer._radiance_w8a16 = True
            self._radiance_w8a16 = True
            sys.stderr.write(f"[radiance.fp8] W8A16 layer N={N} K={K} "
                             f"(scale {scale_name} {tuple(scale.shape)})\n")

        def apply_block_scaled_mm(self, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                                  Bs: torch.Tensor) -> torch.Tensor:
            # A is bf16 (apply_input_quant = False), B e4m3, Bs fp32 block scales.
            return torch.ops.radiance.fp8_w8a16_mm(A.contiguous(), B, Bs)

    _register_ops()
    return RadianceFp8W8A16Kernel


_KERNEL_CLS = None


def kernel_class():
    """The W8A16 kernel class, built once. None when the path is off or unavailable."""
    global _KERNEL_CLS
    if _KERNEL_CLS is None:
        try:
            if not ENABLED or _ext is None or not hasattr(_ext, "launch_w8a16"):
                _KERNEL_CLS = False
            else:
                _KERNEL_CLS = _make_kernel_class()
        except Exception as e:
            sys.stderr.write(f"[radiance.fp8] W8A16 kernel class unavailable, disabled: {e!r}\n")
            _KERNEL_CLS = False
    return _KERNEL_CLS or None
