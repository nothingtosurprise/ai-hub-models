# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.model import SplitForwardMixin

from .model import (
    DEFAULT_PRECISION,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    FPSplitModelWrapper,
    Llama3_2_3B_SSD_Collection,
    Llama3_2_3B_SSD_Part1_Of_4,
    Llama3_2_3B_SSD_Part2_Of_4,
    Llama3_2_3B_SSD_Part3_Of_4,
    Llama3_2_3B_SSD_Part4_Of_4,
    Llama3_2_3B_SSD_PartBase,
    Llama3_2_3B_SSD_PreSplit,
    Llama3_2_3B_SSD_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)

Model = Llama3_2_3B_SSD_Collection

__all__ = [
    "DEFAULT_PRECISION",
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "MODEL_ID",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "FPSplitModelWrapper",
    "Llama3_2_3B_SSD_Collection",
    "Llama3_2_3B_SSD_Part1_Of_4",
    "Llama3_2_3B_SSD_Part2_Of_4",
    "Llama3_2_3B_SSD_Part3_Of_4",
    "Llama3_2_3B_SSD_Part4_Of_4",
    "Llama3_2_3B_SSD_PartBase",
    "Llama3_2_3B_SSD_PreSplit",
    "Llama3_2_3B_SSD_QuantizablePreSplit",
    "Model",
    "QuantizedSplitModelWrapper",
    "SplitForwardMixin",
]
