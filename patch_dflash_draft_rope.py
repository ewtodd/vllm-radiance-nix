#!/usr/bin/env python3
"""Raise a DFlash draft's max_position_embeddings to the target's context.

The DFlash2 drafter's checkpoint declares a 262144-token native context, while
the target is served at 524288. vLLM's dict hf_overrides are deliberately
target-only (compose_draft_hf_overrides), and the EAGLE guard that lifts a
draft's positional ceiling is gated to eagle/eagle3, so the draft's rotary
cos_sin_cache stays at 262144 rows. Once a prefill crosses that bound the rope
kernel reads past the cache and every TP worker dies with
HSA_STATUS_ERROR_MEMORY_FAULT (#48894).

The draft keeps its own rope parameters -- it was trained with those, and
imposing the target's YaRN on it collapses draft acceptance from ~45% to ~2% at
any context -- so only the ceiling moves: RotaryEmbedding sizes cos_sin_cache
from max_position_embeddings, and _get_and_verify_max_len derives the draft's
max_model_len from the same field.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

HELPERS_OLD = """        return functools.partial(
            SpeculativeConfig._apply_composed_hf_override, target_hf_overrides
        )

    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:
"""

HELPERS_NEW = """        return functools.partial(
            SpeculativeConfig._apply_composed_hf_override, target_hf_overrides
        )

    @staticmethod
    def _apply_dflash_max_position_override(
        hf_config: PretrainedConfig,
        target_max_model_len: int,
    ) -> PretrainedConfig:
        # The draft keeps its native rope parameters; only the positional
        # ceiling moves, so its cos_sin_cache can cover the target's context.
        current = getattr(hf_config, "max_position_embeddings", None)
        if current is None or current < target_max_model_len:
            hf_config.max_position_embeddings = target_max_model_len
        return hf_config

    @staticmethod
    def _apply_dflash_max_position_chained_override(
        inner: Callable[[PretrainedConfig], PretrainedConfig],
        target_max_model_len: int,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        return SpeculativeConfig._apply_dflash_max_position_override(
            inner(hf_config), target_max_model_len
        )

    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:
"""

CALL_OLD = """                    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
                        self.target_model_config.hf_overrides
                    )
                self.draft_model_config = ModelConfig(
"""

CALL_NEW = """                    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
                        self.target_model_config.hf_overrides
                    )
                if self.method == "dflash":
                    # The draft must cover the target's context even when its
                    # checkpoint declares a smaller native window.
                    draft_hf_overrides = functools.partial(
                        SpeculativeConfig._apply_dflash_max_position_chained_override,
                        draft_hf_overrides,
                        self.target_model_config.max_model_len,
                    )
                self.draft_model_config = ModelConfig(
"""


def main() -> None:
    config = SP / "vllm" / "config" / "speculative.py"
    apply(
        config,
        HELPERS_OLD,
        HELPERS_NEW,
        "def _apply_dflash_max_position_override(",
        "dflash: draft max position helper",
    )
    apply(
        config,
        CALL_OLD,
        CALL_NEW,
        "# The draft must cover the target's context even when its",
        "dflash: lift draft positional ceiling",
    )


if __name__ == "__main__":
    main()
