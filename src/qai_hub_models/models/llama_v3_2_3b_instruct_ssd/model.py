# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from qai_hub_models import Precision

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.models._shared.llama3.model import (
    LlamaPartBase,
    LlamaPreSplitBase,
    LlamaPreSplitCollectionBase,
    LlamaQuantizablePreSplitBase,
)
from qai_hub_models.models._shared.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import SplitForwardMixin
from qai_hub_models.models._shared.llm_ssd.model import (
    LLMDynamic_SSD_AIMETOnnx,
    append_ssd_forecast_embeddings,
)
from qai_hub_models.models._shared.lm_driver.generator import (
    HubCompatibleGenerator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
# SSD uses a smaller "token" sequence length (32) than the standard models so
# the speculative-decoding forecast tokens fit in one prompt-processor graph.
DEFAULT_EXPORT_SEQUENCE_LENGTHS = [128, 32]

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4

# Model architecture constants (from Llama 3.2 3B)
NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 14
HIDDEN_SIZE = 3072
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 24

# Hugging Face repo
HF_REPO_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 80

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_2_3B_SSD_1_of_4.onnx)
SPLIT_MODEL_NAME = "Llama3_2_3B_SSD"


class Llama3_2_3B_SSD_PreSplit(LlamaPreSplitBase):
    """FP PreSplit for Llama 3.2 3B with SSD forecast embeddings."""

    model_id = MODEL_ID
    GeneratorClass = HubCompatibleGenerator
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    hf_repo_name = HF_REPO_NAME
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = True
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    @classmethod
    def _ssd_forecast_ckpt(cls) -> Path | None:
        """Fetch the SSD self-speculative-decoding forecast module checkpoint.

        Defined as a classmethod (rather than a module-level helper) so the
        shared ``LLMDynamic_SSD_AIMETOnnx.prepare_genie_assets`` can reach it via
        ``cls.FPModel._ssd_forecast_ckpt()``.
        """
        return CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "forecast_module_state_dict.pt"
        ).fetch()

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint, *args, **kwargs)
        # Extend the embedding table with the SSD forecast token embeddings.
        append_ssd_forecast_embeddings(self.model, self._ssd_forecast_ckpt())


class Llama3_2_3B_SSD_QuantizablePreSplit(  # type: ignore[misc]
    LLMDynamic_SSD_AIMETOnnx,
    LlamaQuantizablePreSplitBase[Llama3_2_3B_SSD_PreSplit],
):
    """Quantizable PreSplit for Llama 3.2 3B with SSD genie-asset support."""

    FPModel = Llama3_2_3B_SSD_PreSplit
    GeneratorClass = HubCompatibleGenerator

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    supported_precisions = SUPPORTED_PRECISIONS
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = True
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


class Llama3_2_3B_SSD_PartBase(LlamaPartBase):
    """Unified Part base for Llama 3.2 3B SSD."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    fp_presplit_cls = Llama3_2_3B_SSD_PreSplit
    quant_presplit_cls = Llama3_2_3B_SSD_QuantizablePreSplit
    default_precision = DEFAULT_PRECISION


class Llama3_2_3B_SSD_Part1_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Llama3_2_3B_SSD_Part2_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Llama3_2_3B_SSD_Part3_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 3: Middle layers."""

    part_id = 3


class Llama3_2_3B_SSD_Part4_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 4: Final layers + LM head."""

    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Llama3_2_3B_SSD_Part1_Of_4,
    Llama3_2_3B_SSD_Part2_Of_4,
    Llama3_2_3B_SSD_Part3_Of_4,
    Llama3_2_3B_SSD_Part4_Of_4,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Llama3_2_3B_SSD_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Llama3_2_3B_SSD_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Llama3_2_3B_SSD_Collection(LlamaPreSplitCollectionBase):
    """Unified Collection with 4 Parts for Llama 3.2 3B SSD."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Llama3_2_3B_SSD_PreSplit
    part_base_cls = Llama3_2_3B_SSD_PartBase
    parts = {
        "part1_of_4": Llama3_2_3B_SSD_Part1_Of_4,
        "part2_of_4": Llama3_2_3B_SSD_Part2_Of_4,
        "part3_of_4": Llama3_2_3B_SSD_Part3_Of_4,
        "part4_of_4": Llama3_2_3B_SSD_Part4_Of_4,
    }
