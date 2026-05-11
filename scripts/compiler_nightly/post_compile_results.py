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
    create_results_table,
    get_date_str,
    load_yaml_safe,
    log_and_print,
    setup_script_logging,
)

logger = logging.getLogger(__name__)


def _compile_row(model_name: str, info: dict) -> list:
    return [model_name, info["prod_job_url"], info["dev_job_url"]]


def print_summary(
    regressions: dict,
    progressions: dict,
    failures: dict,
    passed: dict,
    job_yaml_tag: str,
) -> None:
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Count"]
    summary_table.align["Metric"] = "l"
    summary_table.align["Count"] = "r"
    summary_table.add_row(["Passed (Both Prod & Dev)", len(passed)])
    summary_table.add_row(["Regressions", len(regressions)])
    summary_table.add_row(["Progressions", len(progressions)])
    summary_table.add_row(["Failures (Both Prod & Dev)", len(failures)])

    log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
    log_and_print(f"Summary for {job_yaml_tag}", logger)
    log_and_print(DISPLAY_SEPARATOR, logger)
    for line in str(summary_table).split("\n"):
        log_and_print(line, logger)

    # Print regressions to console and log
    field_names = ["Model", "Prod Job URL", "Dev Job URL"]
    if regressions:
        table = create_results_table(regressions, field_names, _compile_row)
        log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
        log_and_print("REGRESSIONS: Prod SUCCESS -> Dev FAILED", logger)
        log_and_print(DISPLAY_SEPARATOR, logger)
        for line in str(table).split("\n"):
            log_and_print(line, logger)

    # Log-only sections
    sections = [
        (progressions, "PROGRESSIONS: Prod FAILED -> Dev SUCCESS"),
        (failures, "FAILURES: FAILED in both Prod and Dev"),
    ]

    for data, title in sections:
        if data:
            table = create_results_table(data, field_names, _compile_row)
            logger.info(f"\n{DISPLAY_SEPARATOR}")
            logger.info(title)
            logger.info(DISPLAY_SEPARATOR)
            for line in str(table).split("\n"):
                logger.info(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print compile nightly results summary"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result YAML files (default: results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used for output file identifier (default: current date)",
    )

    args = parser.parse_args()

    job_yaml_tag = args.tag or get_date_str()
    log_file = setup_script_logging(
        args.results_dir, "post-compile-results", args.verbose, job_yaml_tag
    )
    log_and_print(f"Logging to {log_file}", logger)

    try:
        regressions = load_yaml_safe(
            args.results_dir / f"dev-regressions__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        progressions = load_yaml_safe(
            args.results_dir / f"dev-progressions__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        failures = load_yaml_safe(
            args.results_dir / f"failures-dev-and-prod__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        passed = load_yaml_safe(
            args.results_dir / f"passed-dev-and-prod__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )

        print_summary(regressions, progressions, failures, passed, job_yaml_tag)

        if regressions:
            log_and_print(f"✗ Found {len(regressions)} regressions.", logger)
            return 1

        log_and_print("✓ No regressions found.", logger)
        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
