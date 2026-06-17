# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.qwen3_4b.model import (
    SUPPORTED_PRECISIONS,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_PreSplit,
    Qwen3_4B_QuantizablePreSplit,
)

if __name__ == "__main__":
    use_presplit = "--use-presplit" in sys.argv
    llm_evaluate(
        quantized_model_cls=Qwen3_4B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper if use_presplit else Qwen3_4B_PreSplit,
        qnn_model_cls=LLM_QNN,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
