# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-4B - PreSplit-Part architecture for LLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching for model + ONNX splitting
- Unified Part classes that handle both FP and Quantizable modes based on precision
- Collection class for deploying the model as 4 splits
"""

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
    QuantizedSplitModelWrapper,
    Qwen3_4B_Collection,
    Qwen3_4B_Part1_Of_4,
    Qwen3_4B_Part2_Of_4,
    Qwen3_4B_Part3_Of_4,
    Qwen3_4B_Part4_Of_4,
    Qwen3_4B_PartBase,
    Qwen3_4B_PreSplit,
    Qwen3_4B_QuantizablePreSplit,
)

Model = Qwen3_4B_Collection

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
    "Model",
    "QuantizedSplitModelWrapper",
    "Qwen3_4B_Collection",
    "Qwen3_4B_Part1_Of_4",
    "Qwen3_4B_Part2_Of_4",
    "Qwen3_4B_Part3_Of_4",
    "Qwen3_4B_Part4_Of_4",
    "Qwen3_4B_PartBase",
    "Qwen3_4B_PreSplit",
    "Qwen3_4B_QuantizablePreSplit",
    "SplitForwardMixin",
]
