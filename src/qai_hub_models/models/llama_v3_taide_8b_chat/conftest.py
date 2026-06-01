# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest

from qai_hub_models.models.llama_v3_taide_8b_chat import Model
from qai_hub_models.scorecard.utils.testing import make_cached_from_pretrained_fixture

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig


# Instantiate the model only once for all tests.
# Mock from_pretrained to always return the initialized model.
# This speeds up tests and limits memory leaks.
cached_from_pretrained = make_cached_from_pretrained_fixture(Model)


@pytest.fixture(scope="module", autouse=True)
def ensure_gc() -> None:
    gc.collect()


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig

    return LLMPerfConfig.from_environment()
