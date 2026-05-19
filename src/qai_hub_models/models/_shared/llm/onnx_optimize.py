# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc

import onnx

from qai_hub_models.utils.version_helpers import ensure_supported_version

_BASIC_RULES_NODE_THRESHOLD = 10000


def optimize_onnx_model(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Optimize an ONNX model in place, similar to onnxscript optimize() but
    faster for large models.

    For graphs under 10k nodes, all default rewrite rules are included.
    For larger graphs (LLMs), _basic_rules is skipped since it applies zero
    rewrites on transformer architectures but takes O(n^2) time.
    """
    ensure_supported_version("onnx_ir", min_version="0.2.1")
    ensure_supported_version("onnxscript", min_version="0.7.0")
    import onnx_ir as ir
    import onnx_ir.passes.common as onnx_common_passes
    from onnxscript.optimizer import _constant_folding
    from onnxscript.rewriter import (
        RewritePass,
        _basic_rules,
        _broadcast_to_matmul,
        _cast_constant_of_shape,
        _collapse_slices,
        _fuse_batchnorm,
        _fuse_pad_into_conv,
        _fuse_relus_clips,
        _min_max_to_clip,
        _no_op,
        _redundant_scatter_nd,
        _remove_optional_bias,
    )

    fast_rewrite_rules = (
        *_no_op.rules,
        *_broadcast_to_matmul.rules,
        *_cast_constant_of_shape.rules,
        *_collapse_slices.rules,
        *_min_max_to_clip.rules,
        *_fuse_relus_clips.rules,
        *_redundant_scatter_nd.rules,
        *_fuse_pad_into_conv.rules,
        *_fuse_batchnorm.rules,
        *_remove_optional_bias.rules,
    )

    model_ir = ir.serde.deserialize_model(onnx_model)
    del onnx_model
    gc.collect()

    num_nodes = sum(1 for _ in model_ir.graph)
    if num_nodes < _BASIC_RULES_NODE_THRESHOLD:
        rewrite_rules = (*fast_rewrite_rules, *_basic_rules.basic_optimization_rules())
    else:
        rewrite_rules = fast_rewrite_rules

    input_size_limit = _constant_folding.DEFAULT_CONSTANT_FOLD_INPUT_SIZE_LIMIT
    output_size_limit = _constant_folding.DEFAULT_CONSTANT_FOLD_OUTPUT_SIZE_LIMIT

    passes = ir.passes.Sequential(
        onnx_common_passes.InlinePass(),
        ir.passes.PassManager(
            [
                _constant_folding.FoldConstantsPass(
                    shape_inference=True,
                    input_size_limit=input_size_limit,
                    output_size_limit=output_size_limit,
                ),
                RewritePass(rewrite_rules),
                onnx_common_passes.RemoveUnusedNodesPass(),
                onnx_common_passes.RemoveUnusedFunctionsPass(),
                onnx_common_passes.RemoveUnusedOpsetsPass(),
            ],
            steps=2,
            early_stop=True,
        ),
        onnx_common_passes.RemoveUnusedNodesPass(),
        onnx_common_passes.LiftConstantsToInitializersPass(
            lift_all_constants=True,
            size_limit=output_size_limit,
        ),
        onnx_common_passes.LiftSubgraphInitializersToMainGraphPass(),
        onnx_common_passes.DeduplicateInitializersPass(),
        onnx_common_passes.CommonSubexpressionEliminationPass(),
        onnx_common_passes.OutputFixPass(),
        onnx_common_passes.NameFixPass(),
    )
    passes(model_ir)

    proto = ir.serde.serialize_model(model_ir)
    del model_ir
    gc.collect()

    return proto
