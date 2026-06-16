# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    LLMDynamicBase,
    LLMDynamic_AIMETOnnx,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_EXPORT_CONTEXT_LENGTHS,
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    FPModelT,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    LLMPartBase,
    SingleSlotCacheMixin,
    get_onnx_model,
)

# isort: on
import contextlib

# This verifies aimet is installed; load_encodings_to_sim is used by the
# PreSplit Part layer to load encodings into a per-part QuantSim.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.quantsim import load_encodings_to_sim

import copy
import itertools
import json
import os
import shutil
from collections.abc import Collection
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import onnx
import onnxruntime
import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
import transformers
from packaging.version import Version
from qai_hub.client import Device
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    ModelMetadata,
)
from qai_hub_models.models._shared.llama3.model_adaptations import (
    QcLlama_apply_rotary_pos_emb,
    QCLlamaForCausalLM,
    QCLlamaMLP,
    SHALlamaAttention,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.llm_helpers import (
    create_genie_config,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.models._shared.llm.model import (
    Embedding,
    PositionProcessorBase,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, CachedWebModelAsset
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphCollectionModel,
    MultiGraphWorkbenchModel,
)
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
    mock_torch_onnx_inference,
)
from qai_hub_models.utils.printing import print_with_box

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

DATA_DIR = "data"
USE_CACHED_DATA = True

## Ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|end_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
EOT_ID = "<|eot_id|>"
END_TOKENS = {"<|eot_id|>", "<|end_of_text|>"}


@unique
class Llama3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class RopeEmbedding(Embedding):
    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: LlamaConfig | None = None,
    ) -> None:
        if config is None:
            config = LlamaConfig()
        head_dim = head_dim or (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: LlamaConfig
    ) -> list[torch.Tensor]:
        kwargs: dict[str, Any] = {
            "config": config,
        }
        if Version(transformers.__version__) < Version("4.48"):
            kwargs |= {
                "max_position_embeddings": config.max_position_embeddings,
                "base": config.rope_theta,
                "dim": head_dim,
            }

        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = None

        rope = modeling_llama.LlamaRotaryEmbedding(**kwargs)
        dummy_x = torch.tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward") and callable(rope._original_forward):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        return [emb.unsqueeze(0) for emb in embeddings]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class LlamaPositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: LlamaConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len, config=config)

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


class Llama3Base(LLMBase):
    LMClass = QCLlamaForCausalLM
    EmbeddingClass = RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What do llamas eat? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": BEGIN_TEXT,
            "system_prefix": f"{START_HEADER}{SYSTEM_ID}{END_HEADER}\n\n",
            "system_suffix": EOT_ID,
            "user_prefix": f"{START_HEADER}{USER_ID}{END_HEADER}\n\n",
            "user_suffix": EOT_ID,
            "assistant_prefix": f"{START_HEADER}{ASSISTANT_ID}{END_HEADER}\n\n",
            "assistant_suffix": EOT_ID,
            "default_system_prompt": cls.default_system_prompt,
        }

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Llama3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        elif hasattr(modeling_llama, "LLAMA_ATTENTION_CLASSES"):
            modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = SHALlamaAttention
        else:
            modeling_llama.LlamaAttention = SHALlamaAttention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_llama.LlamaRotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(modeling_llama.LlamaRotaryEmbedding, "_original_forward"):
            modeling_llama.LlamaRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_llama.LlamaRotaryEmbedding.forward
            )
            modeling_llama.LlamaRotaryEmbedding.forward = bypass_RotaryEmbedding
        modeling_llama.apply_rotary_pos_emb = QcLlama_apply_rotary_pos_emb

        def LlamaRMSNorm_forward(
            self: modeling_llama.LlamaRMSNorm, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            # Raise to rank 4
            hidden_states = hidden_states.unsqueeze(0)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return (hidden_states * self.weight).squeeze(0)

        if (
            skip_optimizations
            and Llama3_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            modeling_llama.LlamaRMSNorm.forward = LlamaRMSNorm_forward

        modeling_llama.LlamaMLP = QCLlamaMLP  # type: ignore[misc, unused-ignore]
        modeling_llama.LlamaForCausalLM = QCLlamaForCausalLM  # type: ignore[misc, unused-ignore]

    def _verify_ckpt(self) -> None:
        if (
            not (
                self.llm_config.architectures
                and self.llm_config.architectures[0] == "LlamaForCausalLM"
                and self.llm_config.model_type == "llama"
            )
            and self.llm_config.rope_scaling is not None
            and self.llm_config.rope_scaling["rope_type"] != "llama3"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )


class Llama3Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = RopeEmbedding
    FPModel = Llama3Base

    ada_scale_model_type: str | None = "llama"

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

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

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

        if uses_lists:
            # Convert encodings to dictionaries for faster look-ups
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        if self.llm_io_type in {
            LLMIOType.genie_input_ids,
            LLMIOType.huggingface_input_ids,
        }:
            # See _shared/llama3/model.py for why this is needed.
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

        if uses_lists and self.llm_io_type in {
            LLMIOType.genie_input_ids,
            LLMIOType.huggingface_input_ids,
        }:
            encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # convert back
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Llama3Base_QNN(LLM_QNN):
    FPModel = Llama3Base
    EmbeddingClass = RopeEmbedding
    num_layers_per_split: int


# ---------------------------------------------------------------------------
# Dynamic-shape Llama3 classes
# ---------------------------------------------------------------------------


class Llama3DynamicBase(LLMDynamicBase, Llama3Base):
    """Llama3 FP base with dynamic-shape ONNX export.

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


class Llama3DynamicBase_AIMETOnnx(LLMDynamic_AIMETOnnx, Llama3Base_AIMETOnnx):
    """Dynamic-shape variant of Llama3Base_AIMETOnnx."""

    FPModel = Llama3DynamicBase  # type: ignore[assignment]

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


class LlamaDynamicQuantizablePreSplitMixin(DynamicQuantizablePreSplitMixin[FPModelT]):
    """Llama-specific DynamicQuantizablePreSplit that exports ONNX from torch.

    Overrides resolve_default_checkpoint to download only the quantization
    encodings from the asset store, then export the ONNX model locally
    from the FP torch model. This avoids storing large ONNX files in the
    asset store at the cost of a longer first-load time (~30 min).
    """

    @classmethod
    def resolve_default_checkpoint(
        cls,
        precision: Precision,
        host_device: torch.device,
        fp_model: FPModelT | None,
    ) -> tuple[str, FPModelT | None]:
        """Fetch encodings only and export ONNX from the FP torch model.

        Parameters
        ----------
        precision
            Quantization precision (already validated).
        host_device
            Device for computation.
        fp_model
            Optional FP model passed by the evaluate framework.

        Returns
        -------
        tuple[str, FPModelT | None]
            (resolved_checkpoint_path, fp_model).
        """
        from qai_hub_models.utils.printing import print_with_box

        precision_checkpoint = cls.default_checkpoint[precision]
        encodings_path = str(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                f"{precision_checkpoint}/model.encodings",
            ).fetch()
        )
        checkpoint = str(Path(encodings_path).parent)

        # Create FP model for ONNX export + tokenizer/config
        if fp_model is None:
            fp_model = cls.FPModel.from_pretrained(  # type: ignore[call-arg]
                host_device=host_device,
            )

        # Export ONNX into checkpoint dir (skips if already exists)
        ckpt_path = Path(checkpoint)
        if (
            not (ckpt_path / "model_dynamic.onnx").exists()
            or not (ckpt_path / "model.data").exists()
        ):
            print_with_box(
                [
                    "Exporting ONNX model with dynamic shapes.",
                    "This may take around 30 minutes.",
                ]
            )
        cls.create_onnx_models(  # type: ignore[attr-defined]
            checkpoint=checkpoint,
            fp_model=fp_model,
            context_length=fp_model.context_length,
            host_device=host_device,
            llm_io_type=fp_model.llm_io_type,
            use_dynamic_shapes=True,
        )
        cls.save_tokenizer_and_config(  # type: ignore[attr-defined]
            checkpoint=checkpoint, fp_model=fp_model
        )
        return checkpoint, fp_model


class LlamaPreSplitBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Llama3DynamicBase
):
    """FP PreSplit base for Llama3 models.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.

    Concrete subclasses must set the architecture constants below.
    """

    # --- per-model configuration (override in subclass) ---
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
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
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_names(self.num_layers)

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
            }
        return Llama3DynamicBase._get_input_spec(
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
            llm_io_type=llm_io_type,
        )


LlamaPreSplitT = TypeVar("LlamaPreSplitT", bound=LlamaPreSplitBase)


class LlamaQuantizablePreSplitBase(  # type: ignore[misc]
    LlamaDynamicQuantizablePreSplitMixin[LlamaPreSplitT],
    Llama3DynamicBase_AIMETOnnx,
    Generic[LlamaPreSplitT],
):
    """Quantizable PreSplit base for Llama3 models.

    Manages QuantSim and calibration. Uses class-level cache keyed by
    checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.

    Concrete subclasses must set ``FPModel`` and the config attributes.
    """

    # Set by subclass.
    FPModel: type[LlamaPreSplitT]

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

    def get_output_names(self) -> list[str]:
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_names(self.num_layers)

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


class LlamaPartBase(LLMPartBase, torch.nn.Module, MultiGraphWorkbenchModel):
    """
    Unified Llama Part base: handles both FP and Quantizable modes based on precision.

    Each Part represents one split of the ONNX model for deployment.
    When precision is float, uses the FP PreSplit (ONNX ModelProto inference).
    When precision is quantized, uses the Quantizable PreSplit (ONNXBundle + encodings).

    Concrete models supply the architecture constants and the concrete FP /
    Quantizable PreSplit classes via class attributes.
    """

    # --- per-model configuration (override in subclass) ---
    part_id: int = 0  # 1-indexed
    num_splits: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    default_precision: Precision = Precision.w4a16
    fp_presplit_cls: type[LlamaPreSplitBase]
    quant_presplit_cls: type[LlamaQuantizablePreSplitBase]

    def __init__(
        self,
        presplit: LlamaPreSplitBase | LlamaQuantizablePreSplitBase,
        precision: Precision | None = None,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision or self.default_precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._sequence_lengths = sequence_lengths
        self._context_lengths = context_lengths
        self._graph_names: dict[str, tuple[int, int]] = {
            f"{'token' if seq_len == 1 else 'prompt'}_ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{self.num_splits}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(
                self._sequence_lengths, self._context_lengths
            )
        }

    @property
    def graph_names(self) -> list[str]:
        return list(self._graph_names.keys())

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: LlamaPreSplitBase | LlamaQuantizablePreSplitBase = (
                cls.fp_presplit_cls.from_pretrained(host_device=host_device)
            )
            precision = Precision.float
        else:
            precision = checkpoint_type.precision(
                cls.default_precision, checkpoint=checkpoint
            )
            presplit = cls.quant_presplit_cls.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(
            presplit,
            precision=precision,
            context_lengths=context_lengths,
            sequence_lengths=sequence_lengths,
        )

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        """Get sample inputs for this specific part only.

        Uses actual ONNX input names read from the split model at runtime.
        When called from the multi-graph sample_inputs path, input_spec
        carries the per-graph shapes so we derive seq_len from it.
        """
        # Derive seq_len from input_spec when available (multi-graph path).
        if input_spec is not None and "input_ids" in input_spec:
            seq_len = input_spec["input_ids"][0][1]  # shape (1, seq_len)
        else:
            seq_len, _ = self._graph_names[graph_name]

        full_inputs = self._presplit._sample_inputs_impl()

        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": [np.zeros((1, seq_len), dtype=np.int32)]}

        # Parts 2+: read actual input names from ONNX and match them
        result: SampleInputsType = {}
        onnx_input_names = self._get_onnx_input_names()

        for name in onnx_input_names:
            if name in full_inputs:
                result[name] = full_inputs[name]
            else:
                # Intermediate hidden state (not in full model inputs)
                # found by process of elimination
                result[name] = [
                    np.zeros((1, seq_len, self.hidden_size), dtype=np.float32)
                ]

        return result

    # -------------------------------------------------------------------
    # Methods that branch on self._is_quantized
    # -------------------------------------------------------------------

    def _get_onnx_input_names(self) -> list[str]:
        """Read actual input names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        """Read actual output names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        """Get ONNXBundle for this Part (works for both FP and quantized)."""
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def _get_quant_sim(self) -> QuantizationSimModel:
        """Get or create QuantSim for this specific part from its ONNXBundle."""
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()

        # Load ONNX model
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Clamp to keep QuantSim compatible.
        onnx_model.ir_version = min(onnx_model.ir_version, 10)

        assert isinstance(self._presplit, self.quant_presplit_cls)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        # Use shared construction + activation config. We skip the
        # full-model _configure_quant_sim (lm_head, KV tying) since
        # those heuristics misfire on split models.
        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        # Load encodings if available
        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim, str(onnx_bundle.aimet_encodings_path), strict=False
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        """Get or create an ORT session for FP inference (cached)."""
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()

        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Patch the file in-place (graph-only
        # load keeps external weight references intact).
        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """Forward pass for this Part (FP or quantized based on precision)."""
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    @property
    def shared_source_model(self) -> bool:
        return True

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Export ONNX model for this Part."""
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        # Include precision in directory name to avoid cache collisions
        # between different precisions sharing the same output_path.
        precision_suffix = f"_{self._precision}" if self._is_quantized else ""
        out_dir = Path(output_dir) / f"{model_name}{precision_suffix}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return out_dir

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super().get_graph_hub_compile_options(
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " --quantize_full_type w8a16 --quantize_io",
            device,
        )

    def get_graph_hub_profile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        if self._is_quantized:
            return self._presplit.get_hub_profile_options(
                target_runtime=target_runtime,
                other_profile_options=other_profile_options,
                context_graph_name=graph_name,
            )
        return super().get_graph_hub_profile_options(
            graph_name,
            target_runtime=target_runtime,
            other_profile_options=other_profile_options,
        )


class LlamaPreSplitCollectionBase(MultiGraphCollectionModel):
    """
    Unified Collection base with N Parts for a Llama3 model.

    Supports both FP and Quantizable modes based on precision parameter.
    All Parts share the same PreSplit via class-level cache for memory
    efficiency.

    This base is NOT decorated with ``add_component``; concrete models define
    a decorated subclass that registers their Part classes and sets the config
    attributes below.
    """

    hf_repo_name: str = ""
    fp_presplit_cls: type[LlamaPreSplitBase]
    part_base_cls: type[LlamaPartBase]

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
    ) -> Self:
        """
        Create Collection with all Parts.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"``
            to create from HuggingFace.
        host_device
            Device for computation.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).
        sequence_lengths
            Sequence lengths to compile for.
        context_lengths
            Context lengths to compile for.

        Returns
        -------
        Self
            The Collection with all Parts.
        """
        parts = [
            part_cls.from_pretrained(
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
                sequence_lengths=sequence_lengths,
                context_lengths=context_lengths,
            )
            for part_cls in cls.component_classes.values()
        ]
        return cls(*parts)

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        output_path = Path(output_dir)

        # Save tokenizer and config from HuggingFace (skip if already present)
        if not (output_path / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_name)
            tokenizer.save_pretrained(output_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(output_path))
        if not (output_path / "config.json").exists():
            llm_config = AutoConfig.from_pretrained(self.hf_repo_name)
            llm_config.save_pretrained(output_path)
        else:
            llm_config = AutoConfig.from_pretrained(str(output_path))

        # Derive context_length from the first part
        first_part = next(iter(self.components.values()))
        assert isinstance(first_part, self.part_base_cls)
        context_length: int = first_part._presplit.context_length

        # Build genie_config.json
        model_list = list(metadata.model_files.keys())
        config = create_genie_config(context_length, llm_config, "rope", model_list)
        with open(output_path / "genie_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Build htp_backend_ext_config.json from chipset attributes
        device_info: dict[str, str] = {}
        if metadata.chipset_attributes:
            ca = metadata.chipset_attributes
            if ca.htp_version is not None:
                device_info["hexagon"] = f"v{ca.htp_version}"
            if ca.soc_model is not None:
                device_info["soc-model"] = str(ca.soc_model)
        if save_htp_config_for_genie_bundle(device_info, output_path):
            metadata.supplementary_files["htp_backend_ext_config.json"] = (
                "HTP backend configuration for the target device."
            )

        # Write sample_prompt.txt for on-device genie-t2t-run
        sample_prompt = self.fp_presplit_cls.get_input_prompt_with_tags(
            tokenizer=tokenizer
        )
        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

        chat_spec = self.fp_presplit_cls.get_chat_template()
        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**chat_spec)
            if chat_spec
            else GenieChatTemplate(),
            context_lengths=[context_length],
            supports_streaming=True,
            supports_vision=False,
            supports_thinking=False,
        )

        metadata.supplementary_files["genie_config.json"] = (
            "Genie SDK configuration for on-device LLM inference."
        )
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample prompt for on-device inference."
        )
        metadata.supplementary_files["tokenizer.json"] = (
            "Tokenizer for encoding/decoding text."
        )
