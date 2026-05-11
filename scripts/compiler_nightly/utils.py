# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared utilities and constants for compiler nightly scorecard scripts."""

from __future__ import annotations

import csv
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from prettytable import PrettyTable
from qai_hub import Client
from ruamel.yaml import YAML

# Project and configuration constants
AIHW_COMPILER_NIGHTLY_PROJECT = os.environ.get("COMPILER_NIGHTLY_PROJECT_ID", "")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_MAX_WORKERS = 10

# Job status constants
JOB_STATUS_SUCCESS = "SUCCESS"
JOB_STATUS_FAILED = "FAILED"
MAX_JOB_RUNTIME_SECONDS = 3 * 3600  # 3 hours

# Display constants
DISPLAY_SEPARATOR = "=" * 80

# Model name parsing constants
DEVICE_PATTERNS = ("cs_", "samsung_")


def get_date_str() -> str:
    return datetime.now().strftime("%m-%d-%Y")


def extract_tag_and_dir_from_yaml(yaml_path: Path) -> tuple[str, Path]:
    """Extract tag and output directory from YAML file path.

    Expects filenames like: dev-compile-jobs__TAG.yaml
    """
    output_dir = yaml_path.parent
    stem = yaml_path.stem
    tag = stem.split("__", 1)[1] if "__" in stem else get_date_str()
    if not re.fullmatch(r"[\w\-\.]+", tag):
        raise ValueError(f"Unsafe tag extracted from filename: {tag!r}")
    return tag, output_dir


def load_client(profile: str) -> Client:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading client with profile: {profile}")
    return Client(profile=profile)


def load_yaml_safe(yaml_path: Path, return_empty_on_not_found: bool = False) -> dict:
    logger = logging.getLogger(__name__)
    try:
        yaml = YAML(typ="safe", pure=True)
        with open(yaml_path) as f:
            return yaml.load(f) or {}
    except FileNotFoundError:
        if return_empty_on_not_found:
            logger.warning(f"File not found: {yaml_path}")
            return {}
        raise


def save_yaml_results(data: dict, output_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {output_path}")
    yaml = YAML()
    yaml.default_flow_style = False
    with open(output_path, "w") as f:
        yaml.dump(data, f)


def setup_script_logging(
    output_dir: Path, script_name: str, verbose: bool, date_str: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{script_name}-{date_str}.log"
    setup_logging(log_file, verbose)
    return log_file


def setup_logging(log_file: Path, verbose: bool = False) -> None:
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.ERROR)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )


def log_and_print(message: str, logger: logging.Logger) -> None:
    logger.info(message)
    print(message)


def get_aihw_compiler_nightly_project() -> str:
    if not AIHW_COMPILER_NIGHTLY_PROJECT:
        raise RuntimeError(
            "COMPILER_NIGHTLY_PROJECT_ID environment variable is required but not set"
        )
    return AIHW_COMPILER_NIGHTLY_PROJECT


def strip_device_suffix(model_name: str) -> str:
    """Strip device suffix from model name, preserving component suffix.

    Examples
    --------
        "model_name-cs_8_gen_3" -> "model_name"
        "model_name-cs_8_gen_3_encoder_1" -> "model_name_encoder_1"
        "model_name-samsung_s24" -> "model_name"
        "model_name-samsung_s24_decoder" -> "model_name_decoder"
        "model_name" -> "model_name"
    """
    # Match: base-device_suffix_component_suffix
    # Device patterns: cs_\d+(_gen_\d+|_elite|_elite_gen_\d+)? or samsung_\w+
    device_regex = r"^(.+)-(cs_\d+(?:_gen_\d+|_elite(?:_gen_\d+)?)?|samsung_\w+)(_.+)?$"
    match = re.match(device_regex, model_name)
    if match:
        base = match.group(1)
        component = match.group(3) or ""
        return base + component
    return model_name


def merge_job_options(base_options: str, extra_options: str | None) -> str:
    if extra_options:
        return f"{base_options} {extra_options}".strip()
    return base_options


def map_prod_by_model(prod_config: dict) -> dict:
    logger = logging.getLogger(__name__)
    prod_by_model = {}
    for prod_key, prod_info in prod_config.items():
        model_name_only = strip_device_suffix(prod_key)
        if model_name_only in prod_by_model:
            logger.warning(
                f"map_prod_by_model: duplicate model key '{model_name_only}' "
                f"(from '{prod_key}'), overwriting previous entry"
            )
        prod_by_model[model_name_only] = prod_info
    return prod_by_model


def create_results_table(
    models: dict,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = field_names
    for field in field_names:
        table.align[field] = "l"

    sorted_models = sorted(models.items(), key=sort_key) if sort_key else models.items()
    for model_name, info in sorted_models:
        table.add_row(row_extractor(model_name, info))

    return table


def print_results_table(
    models: dict,
    title: str,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
    empty_message: str = "No models to display.",
    print_to_console: bool = True,
) -> None:
    logger = logging.getLogger(__name__)

    if not models:
        if print_to_console:
            log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
            log_and_print(empty_message, logger)
            log_and_print(DISPLAY_SEPARATOR, logger)
        else:
            logger.info(f"\n{DISPLAY_SEPARATOR}")
            logger.info(empty_message)
            logger.info(DISPLAY_SEPARATOR)
        return

    table = create_results_table(models, field_names, row_extractor, sort_key)
    table_str = str(table)

    if print_to_console:
        log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
        log_and_print(title, logger)
        log_and_print(DISPLAY_SEPARATOR, logger)
        log_and_print(table_str, logger)
    else:
        logger.info(f"\n{DISPLAY_SEPARATOR}")
        logger.info(title)
        logger.info(DISPLAY_SEPARATOR)
        logger.info(table_str)


def save_results_csv(
    results: dict,
    output_path: Path,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
) -> Path:
    logger = logging.getLogger(__name__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = (
        sorted(results.items(), key=sort_key) if sort_key else results.items()
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for model_name, info in sorted_results:
            writer.writerow(row_extractor(model_name, info))

    log_and_print(f"Saved results to: {output_path}", logger)
    return output_path
