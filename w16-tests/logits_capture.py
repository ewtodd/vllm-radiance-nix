"""A vLLM engine-level logits processor that snapshots the last generated step's logits.

The offline extraction in logits_serve.py links this class into the engine with
LLM(logits_processors=[LogitsCapture]); the worker instantiates it and apply() runs on the
final sampling step. RADIANCE_LOGITS_OUT selects the destination file.
"""

import os

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor


class LogitsCapture(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory=False):
        # The ABC's __init__ raises NotImplementedError; subclasses self-initialise.
        self.device = device

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        path = os.environ.get("RADIANCE_LOGITS_OUT")
        if path:
            torch.save(logits.detach().to("cpu", torch.float32), path)
        return logits

    def is_argmax_invariant(self) -> bool:
        # False keeps the processor on the path that runs before greedy argmax as well; the
        # argmax-invariant list is skipped entirely for temperature ~ 0.
        return False

    def update_state(self, batch_update) -> None:
        return None
