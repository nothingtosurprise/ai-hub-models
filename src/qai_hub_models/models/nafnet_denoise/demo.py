# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.nafnet.demo import nafnet_demo
from qai_hub_models.models.nafnet_denoise.model import (
    IMAGE_ADDRESS,
    MODEL_ID,
    NafNetDeNoise,
)


def main(is_test: bool = False) -> None:
    nafnet_demo(NafNetDeNoise, MODEL_ID, IMAGE_ADDRESS, is_test=is_test, task="denoise")


if __name__ == "__main__":
    main()
