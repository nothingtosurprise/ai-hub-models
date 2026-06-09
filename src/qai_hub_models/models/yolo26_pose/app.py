# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloPoseApp


class Yolo26PoseApp(YoloPoseApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """YOLO26 does not check for spatial dim shapes for input image."""
