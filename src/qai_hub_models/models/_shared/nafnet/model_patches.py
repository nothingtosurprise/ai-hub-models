# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class AutoLayerNorm2d(nn.Module):
    """
    LayerNorm for NCHW tensors backed by ``torch.nn.functional.layer_norm``.

    ``torch.nn.LayerNorm`` normalises over the *last* N dimensions, so it
    cannot be applied directly to NCHW tensors when channel-wise normalisation
    is required.  The standard solution is to permute the tensor to NHWC,
    apply layer-norm over the channel dimension (now the last axis), then
    permute back to NCHW.

    Checkpoint compatibility
    ------------------------
    The learnable parameters are stored as ``weight`` and ``bias`` with shape
    ``(channels,)`` — identical names and shapes to the original NAFNet
    ``LayerNorm2d``, so pretrained weights load without any remapping.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Layer Normalization over a 4D input tensor.

        Permutes the input from NCHW to NHWC so that ``F.layer_norm``
        normalises along the channel dimension, then permutes back.

        Parameters
        ----------
        x
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        normalized : torch.Tensor
            Normalized tensor of the same shape as input.
        """
        # NCHW -> NHWC so that the channel axis is last
        x = x.permute(0, 2, 3, 1)
        # Normalise over the channel dimension using torch's optimised kernel
        x = F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)


class NAFLocal_Base:
    """
    Patched Local_Base that skips layer replacement for on-device compatibility.

    The original NAFNet Local_Base.convert() method replaces all nn.AdaptiveAvgPool2d
    layers with a custom AvgPool2d implementation that uses torch.cumsum for efficient
    computation. However, torch.cumsum is not well-supported on certain target devices
    and can cause deployment issues.

    This patched version intentionally skips the layer replacement step, keeping the
    standard nn.AdaptiveAvgPool2d layers intact for better device compatibility.

    Original behavior (skipped):
    ----------------------------
    The original convert() method would:
    1. Recursively traverse all model layers
    2. Replace each nn.AdaptiveAvgPool2d with custom AvgPool2d
    3. The custom AvgPool2d uses cumsum operations for pooling
    4. Run a forward pass to initialize the custom layers

    Patched behavior:
    -----------------
    The convert() method is now a no-op, preserving the original PyTorch layers
    and avoiding cumsum-related deployment issues on target devices.
    """

    def convert(self, *args: Any, **kwargs: Any) -> None:
        # Intentionally empty - skips layer replacement for on-device compatibility
        # This prevents the replacement of nn.AdaptiveAvgPool2d with custom AvgPool2d
        # that uses torch.cumsum, which causes issues on certain target devices
        pass


def ssrforward(
    self: Any, inp_l: torch.Tensor, inp_r: torch.Tensor
) -> list[torch.Tensor]:
    """
    Forward pass with separate left and right stereo inputs.

    Replaces the original forward method to accept two separate images as inputs
    and process them in stereo fashion, instead of splitting a single (1, 6, H, W)
    concatenated input inside the forward pass.

    Parameters
    ----------
    self
        Instance reference.

    inp_l
        Left view image with shape (B, 3, H, W).
        Pixel values in range [0, 1], RGB color space.
    inp_r
        Right view image with shape (B, 3, H, W).
        Pixel values in range [0, 1], RGB color space.

    Returns
    -------
    outputs : list[torch.Tensor]
        List containing two tensors [out_l, out_r]:
        - out_l: Super-resolved left view, shape (B, 3, H*up_scale, W*up_scale)
        - out_r: Super-resolved right view, shape (B, 3, H*up_scale, W*up_scale)
        Both tensors have pixel values in range [0, 1], RGB color space.
    """
    # Interpolate both views for residual connection
    inp_l_hr = F.interpolate(
        inp_l, scale_factor=self.up_scale, mode="bilinear", align_corners=False
    )
    inp_r_hr = F.interpolate(
        inp_r, scale_factor=self.up_scale, mode="bilinear", align_corners=False
    )

    # Intro convolutions
    feat_l = self.intro(inp_l)
    feat_r = self.intro(inp_r)

    # Body with stereo cross-attention
    feat_l, feat_r = self.body(feat_l, feat_r)

    # Upsample and add residual
    out_l = self.up(feat_l) + inp_l_hr
    out_r = self.up(feat_r) + inp_r_hr

    return [out_l, out_r]
