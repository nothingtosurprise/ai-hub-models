#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DISPLAY_SEPARATOR,
    JOB_STATUS_SUCCESS,
    extract_tag_and_dir_from_yaml,
    load_yaml_safe,
    log_and_print,
    print_results_table,
    save_results_csv,
    setup_script_logging,
)

logger = logging.getLogger(__name__)


def calculate_status_changes(prod_config: dict, dev_config: dict) -> dict:
    status_changes = {}

    for model_name, dev_info in dev_config.items():
        # Direct key lookup - no device suffix stripping needed
        prod_info = prod_config.get(model_name, {})

        # Skip if no prod baseline exists
        if not prod_info:
            continue

        prod_success = prod_info.get("prod_job_status") == JOB_STATUS_SUCCESS
        dev_success = dev_info.get("link_status") == JOB_STATUS_SUCCESS

        prod_job_url = prod_info.get("link_job_url")
        if not prod_job_url:
            prod_job_url = prod_info.get("prod_job_url")

        status_changes[model_name] = {
            "prod_success": prod_success,
            "dev_success": dev_success,
            "prod_status": prod_info.get("prod_job_status", "N/A"),
            "dev_status": dev_info.get("link_status", "N/A"),
            "prod_job_url": prod_job_url,
            "dev_job_url": dev_info.get("link_job_url"),
        }

    return status_changes


def get_regressions(status_changes: dict) -> dict:
    regressions = {}
    for model_name, info in status_changes.items():
        if info["prod_success"] and not info["dev_success"]:
            regressions[model_name] = info
    return regressions


def get_progressions(status_changes: dict) -> dict:
    progressions = {}
    for model_name, info in status_changes.items():
        if not info["prod_success"] and info["dev_success"]:
            progressions[model_name] = info
    return progressions


def _status_row(model_name: str, info: dict, empty_value: str = "N/A") -> list:
    return [
        model_name,
        info["prod_status"],
        info["dev_status"],
        info.get("prod_job_url", empty_value),
        info.get("dev_job_url", empty_value),
    ]


def print_progressions_table(progressions: dict) -> None:
    field_names = ["Model", "Prod Status", "Dev Status", "Prod URL", "Dev URL"]
    print_results_table(
        progressions,
        title=f"FIXES: {len(progressions)} models now succeed in dev",
        field_names=field_names,
        row_extractor=_status_row,
        sort_key=lambda x: x[0],
        empty_message="No progressions found.",
        print_to_console=True,
    )


def print_regressions_table(regressions: dict) -> None:
    field_names = ["Model", "Prod Status", "Dev Status", "Prod URL", "Dev URL"]
    print_results_table(
        regressions,
        title=f"REGRESSIONS: {len(regressions)} models now fail in dev",
        field_names=field_names,
        row_extractor=_status_row,
        sort_key=lambda x: x[0],
        empty_message="No regressions found! 🎉",
        print_to_console=True,
    )


def save_full_table_csv(status_changes: dict, output_dir: Path, tag: str) -> Path:
    field_names = ["Model", "Prod Status", "Dev Status", "Prod Job URL", "Dev Job URL"]
    csv_path = output_dir / f"link-results__{tag}.csv"
    return save_results_csv(
        status_changes,
        csv_path,
        field_names,
        lambda name, info: _status_row(name, info, empty_value=""),
        sort_key=lambda x: x[0],
    )


def print_summary(status_changes: dict, regressions: dict, progressions: dict) -> None:
    total = len(status_changes)
    prod_success = sum(1 for s in status_changes.values() if s["prod_success"])
    dev_success = sum(1 for s in status_changes.values() if s["dev_success"])
    regressions_count = len(regressions)
    progressions_count = len(progressions)
    unchanged = total - regressions_count - progressions_count

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.align["Metric"] = "l"
    summary_table.align["Value"] = "r"

    summary_table.add_row(["Total Models", total])
    summary_table.add_row(["Prod Successes", prod_success])
    summary_table.add_row(["Dev Successes", dev_success])
    summary_table.add_row(
        ["Progressions (prod fail -> dev success)", progressions_count]
    )
    summary_table.add_row(["Unchanged", unchanged])
    summary_table.add_row(["Regressions (prod success -> dev fail)", regressions_count])

    log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
    log_and_print("Link Results Summary", logger)
    log_and_print(DISPLAY_SEPARATOR, logger)
    for line in str(summary_table).split("\n"):
        log_and_print(line, logger)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze link results and identify status changes"
    )
    parser.add_argument(
        "--dev-link-config",
        type=Path,
        required=True,
        help="Path to dev-link-jobs__<tag>.yaml with collected results",
    )
    parser.add_argument(
        "--prod-link-config",
        type=Path,
        required=True,
        help="Path to AIHM link-scorecard.yaml from prod",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    tag, output_dir = extract_tag_and_dir_from_yaml(args.dev_link_config)
    log_file = setup_script_logging(output_dir, "post-link-results", args.verbose, tag)
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        prod_config = load_yaml_safe(args.prod_link_config)
        dev_config = load_yaml_safe(args.dev_link_config)

        log_and_print(f"Loaded {len(prod_config)} prod link jobs", logger)
        log_and_print(f"Loaded {len(dev_config)} dev link jobs", logger)

        status_changes = calculate_status_changes(prod_config, dev_config)
        regressions = get_regressions(status_changes)
        progressions = get_progressions(status_changes)

        print_summary(status_changes, regressions, progressions)
        print_progressions_table(progressions)
        print_regressions_table(regressions)
        save_full_table_csv(status_changes, output_dir, tag)

        if regressions:
            log_and_print(f"✗ Found {len(regressions)} link regressions.", logger)
            return 1

        log_and_print("✓ No link regressions found.", logger)
        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
