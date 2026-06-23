# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from qai_hub_models_cli.proto.info_pb2 import ModelInfo
from qai_hub_models_cli.proto.manifest_pb2 import ManifestModelEntry, ReleaseManifest
from qai_hub_models_cli.proto.numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.versions import CURRENT_VERSION
from tqdm import tqdm

from qai_hub_models._version import __version__
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts.build_release_proto import (
    _build_release_assets_proto,
    _manifest_filter_fields,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, is_internal_repo


def get_manifest_proto() -> ReleaseManifest:
    """Build a ReleaseManifest from local model configs (dev installs)."""

    def _build_entry(model_id: str) -> ManifestModelEntry | None:
        info = QAIHMModelInfo.from_model(model_id)
        info_proto = info.to_proto(__version__)
        if not is_internal_repo() and info.status != MODEL_STATUS.PUBLISHED:
            return None
        release_assets = QAIHMModelReleaseAssets.from_model(
            model_id, not_exists_ok=True
        )
        perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        return ManifestModelEntry(
            id=model_id,
            display_name=info.name,
            domain=info_proto.domain,
            **_manifest_filter_fields(release_assets, perf, info),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(
            tqdm(
                pool.map(_build_entry, MODEL_IDS),
                total=len(MODEL_IDS),
                desc="Building CLI model manifest from current repository ref",
                unit="model",
            )
        )

    return ReleaseManifest(
        version=__version__, models=filter(lambda x: x is not None, entries)
    )


def get_info_proto(model_id: str) -> ModelInfo:
    """Build a ModelInfo proto from local model config (dev installs)."""
    return QAIHMModelInfo.from_model(model_id).to_proto(__version__)


def get_perf_proto(model_id: str) -> ModelPerf:
    """Build a ModelPerf proto from local model config (dev installs)."""
    return QAIHMModelPerf.from_model(model_id, not_exists_ok=True).to_proto(
        __version__, model_id
    )


def get_numerics_proto(model_id: str) -> ModelNumerics:
    """Build a ModelNumerics proto from local model config (dev installs)."""
    numerics = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
    if numerics:
        # Cross-reference SDK/tool versions from perf (same as the release build).
        perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        return numerics.to_proto(__version__, model_id, perf=perf)
    return ModelNumerics()


def get_platform_proto() -> PlatformInfo:
    """Build a PlatformInfo proto from local config (dev installs)."""
    return DevicesAndChipsetsYaml.load().to_proto(__version__)


def get_release_assets_proto(model_id: str) -> ModelReleaseAssets | None:
    """Build a ModelReleaseAssets proto from local model config (dev installs)."""
    return _build_release_assets_proto(
        model_id, str(CURRENT_VERSION), is_internal_repo()
    )
