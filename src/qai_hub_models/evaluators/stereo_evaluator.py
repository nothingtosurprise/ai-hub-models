# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator


class StereoEvaluator(SuperResolutionOutputEvaluator):
    """Evaluator for comparing a batched stereo image output (left + right views)."""

    def add_batch(self, output: list[torch.Tensor], gt: list[torch.Tensor]) -> None:
        """Evaluate one batch of stereo predictions.

        Concatenates the left and right view tensors along the batch dimension
        and delegates to :meth:`SuperResolutionOutputEvaluator.add_batch` for
        per-image PSNR computation (YUV-space, 8-bit data range).

        Parameters
        ----------
        output
            ``[left_pred, right_pred]`` — each tensor of shape ``(B, C, H, W)``.
        gt
            ``[left_gt, right_gt]`` — each tensor of shape ``(B, C, H, W)``.
        """
        assert gt[0].shape == output[0].shape and gt[1].shape == output[1].shape

        combined_output = torch.cat((output[0], output[1]), dim=0)
        combined_gt = torch.cat((gt[0], gt[1]), dim=0)

        super().add_batch(combined_output, combined_gt)
