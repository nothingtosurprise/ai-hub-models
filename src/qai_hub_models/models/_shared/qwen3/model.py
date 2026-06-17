# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    PositionProcessorBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    LLMDynamicBase,
    LLMDynamic_AIMETOnnx,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    DynamicSplitCollectionBase,
    DynamicSplitPartBase,
    SingleSlotCacheMixin,
    get_onnx_model,
)

# isort: on
import contextlib
import copy
import json
import os
import shutil
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import onnx
import torch

# isort: off
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.common.defs import QuantizationDataType
    from aimet_onnx.quantsim import QuantizationSimModel
# isort: on

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen3 import modeling_qwen3
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models._shared.llama3.model import RopeEmbedding
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models._shared.qwen3.model_adaptations import (
    QcQwen3_apply_rotary_pos_emb,
    QCQwen3ForCausalLM,
    QCQwen3MLP,
    SHAQwen3Attention,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.printing import print_with_box

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_qwen"

DATA_DIR = "data"
USE_CACHED_DATA = True

# Qwen3 uses the same ChatML format as Qwen2
START_HEADER = "<|im_start|>"
END_HEADER = "<|im_end|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
END_TOKENS = {"<|im_end|>", "<|endoftext|>"}


class Qwen3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class Qwen3Base(LLMBase):
    LMClass = modeling_qwen3.Qwen3ForCausalLM
    EmbeddingClass = RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": "",
            "system_prefix": f"{START_HEADER}{SYSTEM_ID}\n",
            "system_suffix": f"{END_HEADER}\n",
            "user_prefix": f"{START_HEADER}{USER_ID}\n",
            "user_suffix": f"{END_HEADER}\n",
            "assistant_prefix": f"{START_HEADER}{ASSISTANT_ID}\n",
            "assistant_suffix": f"{END_HEADER}\n",
            "default_system_prompt": cls.default_system_prompt,
        }

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Qwen3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            # In transformers 4.51.0+, Qwen3 directly instantiates Qwen3Attention
            # instead of using an ATTENTION_CLASSES dict
            modeling_qwen3.Qwen3Attention = SHAQwen3Attention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_qwen3.Qwen3RotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(modeling_qwen3.Qwen3RotaryEmbedding, "_original_forward"):
            modeling_qwen3.Qwen3RotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore [reportAttributeAccessIssue]
                modeling_qwen3.Qwen3RotaryEmbedding.forward
            )
            modeling_qwen3.Qwen3RotaryEmbedding.forward = bypass_RotaryEmbedding
        modeling_qwen3.apply_rotary_pos_emb = QcQwen3_apply_rotary_pos_emb

        modeling_qwen3.Qwen3MLP = QCQwen3MLP  # type: ignore[misc, unused-ignore]
        modeling_qwen3.Qwen3ForCausalLM = QCQwen3ForCausalLM  # type: ignore[misc, unused-ignore]

    def _verify_ckpt(self) -> None:
        if not (
            self.llm_config.architectures[0] == "Qwen3ForCausalLM"  # type: ignore[index, unused-ignore]
            and self.llm_config.model_type == "qwen3"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> list[torch.Tensor]:
        return super().forward(
            input_tokens,
            self.attention_mask_multiplier * attention_mask,
            *rest,
        )


class Qwen3PositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: PretrainedConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len, config=config)  # type: ignore[arg-type, unused-ignore]

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask_converter = AttentionMaskConverter(True)
        attention_mask = attention_mask_converter.to_4d(
            attention_mask_before_processor,
            query_length=position_ids.shape[1],
            key_value_length=attention_mask_before_processor.shape[1],
            dtype=torch.float32,
        )
        attention_mask = attention_mask.clip(-50, 0)
        return attention_mask, position_ids_cos, position_ids_sin


class Qwen3Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = RopeEmbedding
    FPModel = Qwen3Base

    ada_scale_model_type: str | None = "qwen3"

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_lengths: list[int],
        model_list: list[str],
        output_path: Path,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
        model_id: str,
        model_name: str,
    ) -> None:
        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_lengths,
            model_list,
            output_path,
            precision,
            encodings_path,
            input_specs,
            output_specs,
            model_id=model_id,
            model_name=model_name,
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        return super().forward(
            input_tokens,
            self.attention_mask_multiplier * attention_mask,
            *rest,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Make sure AIMET encodings are ready for ONNX split."""
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        assert uses_lists

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # See Llama3Base_AIMETOnnx._adapt_aimet_encodings in
        # _shared/llama3/model.py for why this is needed.
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        # convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


# ---------------------------------------------------------------------------
# Dynamic-shape Qwen3 classes
# ---------------------------------------------------------------------------


class Qwen3DynamicBase(LLMDynamicBase, Qwen3Base):
    """Qwen3 FP base with dynamic-shape ONNX export.

    Provides get_full_onnx_bundle() which exports the torch model to ONNX
    with dynamic shapes, caching the result alongside model.encodings.
    """

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes.

        Caches the exported ONNX to the default checkpoint directory
        (alongside model.encodings) so subsequent runs skip the ~30 min
        export.
        """
        precision_dir = self.default_checkpoint.get(self.default_precision)
        cache_dir = (
            ASSET_CONFIG.get_local_store_model_path(
                self.model_id, self.model_asset_version, precision_dir
            )
            if precision_dir
            else None
        )

        if cache_dir is not None:
            cached_onnx = cache_dir / "model_dynamic.onnx"
            cached_data = cache_dir / "model.data"
            if cached_onnx.exists() and cached_data.exists():
                print(f"\nLoading cached dynamic ONNX from {cache_dir}")
                bundle_dir = temp_path / "full_dynamic"
                bundle_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(cached_onnx, bundle_dir / "model.onnx")
                shutil.copy(cached_data, bundle_dir / "model.data")
                return ONNXBundle.from_bundle_path(bundle_dir, "model")

        print_with_box(
            [
                "Exporting ONNX model with dynamic shapes.",
                "This may take around 30 minutes.",
            ]
        )
        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            use_dynamic_shapes=True,
            quiet=True,
        )

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(onnx_dir / "model.onnx", cache_dir / "model_dynamic.onnx")
            shutil.copy(onnx_dir / "model.data", cache_dir / "model.data")
            print(f"\nCached dynamic ONNX to {cache_dir}")

        return ONNXBundle.from_bundle_path(onnx_dir, "model")


class Qwen3DynamicBase_AIMETOnnx(LLMDynamic_AIMETOnnx, Qwen3Base_AIMETOnnx):
    """Dynamic-shape variant of Qwen3Base_AIMETOnnx."""

    FPModel = Qwen3DynamicBase  # type: ignore[assignment]

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Adapt AIMET encodings for the dynamic-shape ONNX model.

        The dynamic ONNX uses node names from torch.export, so this override
        finds the embedding Gather by op_type rather than hardcoded static
        names. Single pass:
        1. Sets the embedding Gather output encoding (precision-dependent)
        2. Promotes weight activation encodings to param_encodings
        3. Propagates encodings through memory ops for correct splitting
        """
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path, load_external_data=False)

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        if uses_lists:
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        # Find the embedding Gather node and set its output encoding.
        # For w4 (no activation quantization): use fp16.
        # For w4a16: copy the embedding weight's quantization parameters.
        gather_node = next((n for n in model.graph.node if n.op_type == "Gather"), None)
        if gather_node is not None:
            gather_output = gather_node.output[0]
            embed_weight_name = gather_node.input[0]

            if self.precision == Precision.w4:
                encodings["activation_encodings"][gather_output] = {
                    "bw": 16,
                    "dtype": "FLOAT",
                    "enc_type": "PER_TENSOR",
                    "name": gather_output,
                }
            else:
                weight_enc = encodings["activation_encodings"].get(embed_weight_name)
                if weight_enc is not None:
                    embedding_enc = copy.deepcopy(weight_enc)
                    embedding_enc["name"] = gather_output
                    encodings["activation_encodings"][gather_output] = embedding_enc

        # Promote weight entries in activation_encodings to param_encodings
        for key, value in list(encodings["activation_encodings"].items()):
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(value)

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as f:
            json.dump(encodings, f, indent=4, sort_keys=True)


class Qwen3Base_QNN(LLM_QNN):
    FPModel = Qwen3Base
    EmbeddingClass = RopeEmbedding
    num_layers_per_split: int

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names


# ---------------------------------------------------------------------------
# Qwen3 PreSplit / Part / Collection family bases
#
# These mirror the Llama family bases in _shared/llama3/model.py. The generic
# split/inference machinery lives in DynamicSplitPartBase /
# DynamicSplitCollectionBase (_shared/llm/model.py); only the genuinely
# Qwen3-coupled pieces live here:
#   - PreSplit bases bind to Qwen3DynamicBase / Qwen3DynamicBase_AIMETOnnx
#     (RoPE embedding + dynamo encoding adaptation).
#   - Qwen3 has an explicit head_dim that differs from
#     hidden_size // num_attention_heads.
#   - Qwen3 multiplies the attention mask, so it uses distinct clip values.
#   - Qwen3PartBase carries the tied-embedding encoding fix.
# ---------------------------------------------------------------------------

# Since we multiply the attention mask for Qwen3, the default clip value has
# issues, so we use the Genie value for the unquantized variant too.
QWEN3_FP_ATTENTION_MASK_MIN_CLIP = -1000.0


class Qwen3PreSplitBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Qwen3DynamicBase
):
    """FP PreSplit base for Qwen3 models.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes).

    Concrete subclasses must set the architecture constants below.
    """

    # Generator used by the demo / eval / calibration paths (make_generator).
    # Qwen3 is a standard text LLM, so it uses the generic Hub generator.
    GeneratorClass = HubCompatibleGenerator

    # --- per-model configuration (override in subclass) ---
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    # Qwen3 defines an explicit head_dim that differs from
    # hidden_size // num_attention_heads.
    head_dim: int | None = None
    hf_repo_name: str = ""

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    # Asset / cache config
    min_memory_recommended: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    default_precision: Precision = Precision.w4a16

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint or self.hf_repo_name, **kwargs)

    def _verify_ckpt(self) -> None:
        """Verify checkpoint compatibility."""
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == self.num_layers
            and self.llm_config.hidden_size == self.hidden_size
            and self.llm_config.num_attention_heads == self.num_attention_heads
            and self.llm_config.num_key_value_heads == self.num_key_value_heads
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = None,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """
        Load or return a cached FP PreSplit.

        Uses dynamic shapes so sequence_length/context_length are not
        needed at construction time.
        """
        checkpoint = checkpoint or cls.hf_repo_name
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached

        instance = cls(
            checkpoint=checkpoint,
            host_device=host_device,
            load_pretrained=True,
            # Qwen3 multiplies the attention mask, so the FP variant uses the
            # Genie clip value too.
            attention_mask_min_clip=QWEN3_FP_ATTENTION_MASK_MIN_CLIP,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Qwen3Base._get_output_names(self.num_layers)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return self._static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type
        )

    @classmethod
    def _static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Build the full-model input spec from the model's architecture
        constants (or an explicit ``llm_config`` override).

        This is a classmethod (not an instance method) so the Quantizable
        PreSplit can build the spec from the FP class without an instance.
        """
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": cls.num_layers,
                "hidden_size": cls.hidden_size,
                "num_key_value_heads": cls.num_key_value_heads,
                "num_attention_heads": cls.num_attention_heads,
                "head_dim": cls.head_dim,
            }
        return Qwen3Base._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", cls.num_layers),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config.get("hidden_size", cls.hidden_size),
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", cls.num_key_value_heads
            ),
            num_attention_heads=llm_config.get(
                "num_attention_heads", cls.num_attention_heads
            ),
            head_dim=llm_config.get("head_dim", cls.head_dim),
            llm_io_type=llm_io_type,
        )


Qwen3PreSplitT = TypeVar("Qwen3PreSplitT", bound=Qwen3PreSplitBase)


class Qwen3QuantizablePreSplitBase(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin[Qwen3PreSplitT],
    Qwen3DynamicBase_AIMETOnnx,
    Generic[Qwen3PreSplitT],
):
    """Quantizable PreSplit base for Qwen3 models.

    Uses the base DynamicQuantizablePreSplitMixin.resolve_default_checkpoint,
    which downloads the full .zip asset (dynamic ONNX + encodings + weights +
    tokenizer + config). Qwen3 ships the full zip (unlike Llama, which fetches
    encodings only and re-exports ONNX locally).

    Concrete subclasses must set ``FPModel`` and the config attributes.
    """

    # Generator used by the demo / eval / calibration paths (make_generator).
    GeneratorClass = HubCompatibleGenerator

    # Set by subclass.
    FPModel: type[Qwen3PreSplitT]

    # Config
    num_layers: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    # AdaScale config (override in subclass).
    ada_scale_num_rmsnorm_per_blk: int | None = None
    supports_thinking: bool = False

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        # Qwen3 multiplies the attention mask; the quantized variant uses
        # (-100, 1.0).
        return (-100.0, 1.0)

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Qwen3Base._get_output_names(self.num_layers)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        if bundle.aimet_encodings_path is not None:
            self._adapt_aimet_encodings(
                str(bundle.aimet_encodings_path),
                str(bundle.aimet_encodings_path),
                str(bundle.onnx_graph_path),
            )
        return super()._postprocess_full_onnx_bundle(bundle)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return self.FPModel._static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )


class Qwen3PartBase(DynamicSplitPartBase):
    """Unified Qwen3 Part base.

    Adds the Qwen3 tied-embedding encoding fix on top of the generic
    DynamicSplitPartBase. Concrete models supply the architecture constants
    (including the explicit ``head_dim``) and the FP / Quantizable PreSplit
    classes via class attributes.
    """

    def transform_split_encodings(
        self,
        onnx_model: onnx.ModelProto,
        quant_sim: QuantizationSimModel,
        encodings_path: str,
    ) -> str:
        """Adjust QuantSim quantizers / encodings for the tied embedding weight.

        Qwen3 ties lm_head.weight to the embedding table, so the dynamo graph
        names the single tied initializer ``model.lm_head.weight`` and feeds
        it to both the embedding ``Gather`` (this part, when it has one) and
        the lm_head ``MatMul`` (final part). ``_build_quantsim`` makes every
        weight a per-channel int4 param quantizer, but a ``Gather`` input has
        ``tensor_quantizer_params=None`` so per-channel can't be applied and
        ``load_encodings_to_sim`` raises. The embedding is fp16 (not
        weight-quantized) anyway, so relax that quantizer to float16 before
        loading. The final (lm_head) part is unaffected -- there the same
        weight feeds a MatMul and keeps its per-channel encoding.
        AIMET renames quantized tensors with an ``_updated`` / ``_qdq``
        suffix in the QuantSim graph, but the quantizer dict and encodings
        use the original initializer name -- strip the suffix when matching.
        """

        def _strip_suffix(name: str) -> str:
            for suffix in ("_updated", "_qdq"):
                if name.endswith(suffix):
                    return name[: -len(suffix)]
            return name

        gather_weight_names = {
            _strip_suffix(node.input[0])
            for node in onnx_model.graph.node
            if node.op_type == "Gather" and len(node.input) >= 1
        }
        self._relax_gather_weight_quantizers(quant_sim, gather_weight_names)
        return self._embedding_safe_encodings(encodings_path, gather_weight_names)

    @staticmethod
    def _relax_gather_weight_quantizers(
        quant_sim: QuantizationSimModel, gather_weight_names: set[str]
    ) -> None:
        """Set Gather-fed weight quantizers to float16, per-channel off.

        The default QuantSim config marks every weight a per-channel int param
        (``channelAxis=0``), but a ``Gather`` input cannot carry per-channel
        params (``tensor_quantizer_params`` is ``None``). The tied embedding is
        fp16 / not weight-quantized in the source encodings, so relax that
        quantizer here. See transform_split_encodings for the full
        tied-embedding rationale.
        """
        for name in gather_weight_names:
            qc_op = quant_sim.qc_quantize_op_dict.get(name)
            if qc_op is None:
                continue
            qc_op.quant_info.usePerChannelMode = False
            qc_op.reset_encoding_stats()
            qc_op.data_type = QuantizationDataType.float
            qc_op.bitwidth = 16

    @staticmethod
    def _embedding_safe_encodings(
        encodings_path: str, gather_weight_names: set[str]
    ) -> str:
        """Drop per-channel param encodings on Gather-fed weights.

        ``load_encodings_to_sim`` would otherwise try to apply the migrated
        per-channel lm_head encoding to the embedding part's Gather-fed weight
        (which has no ``tensor_quantizer_params``) and raise. Returns the path
        to a rewritten encodings file, or the original path when nothing
        changes (e.g. the lm_head part, where the weight feeds a MatMul).
        """
        if not gather_weight_names:
            return encodings_path
        with open(encodings_path) as f:
            encodings = json.load(f)
        param_encodings = encodings.get("param_encodings", [])
        kept = [
            e
            for e in param_encodings
            if not (
                e.get("name") in gather_weight_names
                and e.get("enc_type") == "PER_CHANNEL"
            )
        ]
        if len(kept) == len(param_encodings):
            return encodings_path
        encodings["param_encodings"] = kept
        safe_path = str(Path(encodings_path).with_suffix(".embedding_safe.encodings"))
        with open(safe_path, "w") as f:
            json.dump(encodings, f, indent=4, sort_keys=True)
        return safe_path


class Qwen3PreSplitCollectionBase(DynamicSplitCollectionBase):
    """Unified Collection base with N Parts for a Qwen3 model.

    All deployment machinery is generic and lives in
    :class:`DynamicSplitCollectionBase`. Concrete models register their Part
    classes via the ``parts`` mapping and set ``hf_repo_name`` /
    ``fp_presplit_cls`` / ``part_base_cls`` / ``supports_thinking``.
    """
