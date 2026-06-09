# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import shutil
from pathlib import Path

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


def write_labels_file(
    labels_file_name: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
) -> None:
    """
    Copy a labels file from qai_hub_models/labels/ to output_dir and register
    it in metadata.supplementary_files.

    Parameters
    ----------
    labels_file_name
        Name of the labels file in qai_hub_models/labels/ (e.g. "coco_labels.txt").
    output_dir
        Directory where the file should be written.
    metadata
        Model metadata; supplementary_files will be updated.
    """
    out_path = Path(output_dir) / "labels.txt"
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / labels_file_name
    shutil.copyfile(labels_path, out_path)
    metadata.supplementary_files["labels.txt"] = (
        "Mapping of model prediction indices -> string labels."
    )
