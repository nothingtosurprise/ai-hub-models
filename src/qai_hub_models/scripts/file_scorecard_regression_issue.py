#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Generate a GitHub issue body for 2x+ scorecard regressions.

Reads the JSON files produced by PerformanceDiff.dump_severe_regressions_json()
and NumericsDiff.dump_regressions_json(), then renders a Jinja template.

The actual issue creation is done by the GitHub Action that calls this script.

Usage (from GitHub Actions):
    python3 -m qai_hub_models.scripts.file_scorecard_regression_issue \
        --perf-regressions-json path/to/perf-regressions-2x-*.json \
        --output regression-issue.json \
        --run-url "https://github.com/..." \
        --perf-diff-url "https://..." \
        --numerics-diff-url "https://..."
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"

# GitHub's hard limit on issue bodies is 65536 characters. Leave a small margin
# so the truncation footer we may append still fits.
MAX_ISSUE_BODY_LEN = 65000

# AI Hub job IDs are short alphanumeric tokens. Anything else is rejected to
# avoid markdown injection in the rendered issue body.
_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _job_url(job_id: str, deployment: str = "workbench") -> str:
    """Build an AI Hub job URL from a job ID."""
    if job_id == "null" or not job_id:
        return ""
    return f"https://{deployment}.aihub.qualcomm.com/jobs/{job_id}/"


def _job_link(job_id: str, deployment: str = "workbench") -> str:
    """Build a markdown link for a job ID, or N/A."""
    if job_id == "null" or not job_id or not _JOB_ID_RE.match(job_id):
        return "N/A"
    return f"[{job_id}]({_job_url(job_id, deployment)})"


def _linkify_job_ids(rows: list[dict], deployment: str = "workbench") -> list[dict]:
    """Convert job ID values to markdown links in-place.

    Any column whose name contains "Job ID" gets its value converted
    from a plain ID string to a markdown link.
    """
    result = []
    for row in rows:
        new_row = {}
        for key, val in row.items():
            if "Job ID" in key:
                new_row[key] = _job_link(str(val), deployment)
            else:
                new_row[key] = val
        result.append(new_row)
    return result


def _render(
    today: str,
    perf_regressions: list[dict],
    numerics_regressions: list[dict],
    run_url: str,
    perf_diff_url: str,
    numerics_diff_url: str,
    perf_dropped: int = 0,
    numerics_dropped: int = 0,
) -> str:
    template = _env.get_template("scorecard_regression_issue_template.j2")
    return template.render(
        today=today,
        perf_regressions=perf_regressions,
        numerics_regressions=numerics_regressions,
        run_url=run_url,
        perf_diff_url=perf_diff_url,
        numerics_diff_url=numerics_diff_url,
        perf_dropped=perf_dropped,
        numerics_dropped=numerics_dropped,
    )


def build_issue_body(
    perf_regressions: list[dict],
    numerics_regressions: list[dict],
    run_url: str,
    perf_diff_url: str,
    numerics_diff_url: str,
    deployment: str = "workbench",
) -> str:
    """Build the GitHub issue body with regression tables.

    GitHub rejects issue bodies longer than 65536 characters. If the rendered
    body would exceed that, drop rows from the largest table first until it
    fits, and append a note pointing readers to the linked diff artifacts for
    the full list.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    perf = _linkify_job_ids(perf_regressions, deployment)
    numerics = list(numerics_regressions)

    body = _render(today, perf, numerics, run_url, perf_diff_url, numerics_diff_url)
    perf_dropped = numerics_dropped = 0
    while len(body) > MAX_ISSUE_BODY_LEN and (perf or numerics):
        # Bulk-drop based on current overage so we don't re-render once per row.
        total_rows = len(perf) + len(numerics)
        chars_per_row = max(1, len(body) // max(total_rows, 1))
        rows_to_drop = max(1, (len(body) - MAX_ISSUE_BODY_LEN) // chars_per_row)
        for _ in range(rows_to_drop):
            if not perf and not numerics:
                break
            # Drop from whichever table currently has more rows; ties go to perf.
            if len(perf) >= len(numerics) and perf:
                perf.pop()
                perf_dropped += 1
            elif numerics:
                numerics.pop()
                numerics_dropped += 1
        body = _render(
            today,
            perf,
            numerics,
            run_url,
            perf_diff_url,
            numerics_diff_url,
            perf_dropped=perf_dropped,
            numerics_dropped=numerics_dropped,
        )
    # Defensive hard cap: if the template's fixed overhead alone (headers,
    # URLs, footers) exceeds the limit, GitHub would still 422 us. Truncate.
    if len(body) > MAX_ISSUE_BODY_LEN:
        body = body[: MAX_ISSUE_BODY_LEN - 3] + "..."
    return body


def _resolve_glob(pattern: str) -> str | None:
    """Resolve a glob pattern to a single file path, or None."""
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load_json(path: str | None) -> list[dict]:
    """Load a JSON file and return its contents, or empty list."""
    if not path:
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GitHub issue body for 2x+ scorecard regressions"
    )
    parser.add_argument(
        "--perf-regressions-json",
        required=True,
        help="Path (or glob) to perf-regressions-2x-*.json",
    )
    parser.add_argument(
        "--numerics-regressions-json",
        default="",
        help="Path (or glob) to numerics-regressions-*.json",
    )
    parser.add_argument(
        "--run-url",
        default="N/A",
        help="URL to the scorecard GitHub Actions run",
    )
    parser.add_argument(
        "--perf-diff-url",
        default="N/A",
        help="URL to the performance diff artifact",
    )
    parser.add_argument(
        "--numerics-diff-url",
        default="N/A",
        help="URL to the numerics diff artifact",
    )
    parser.add_argument(
        "--deployment",
        default="workbench",
        help="AI Hub deployment subdomain for job URLs (default: workbench)",
    )
    parser.add_argument(
        "--labels",
        default="p1,scorecard",
        help="Comma-separated labels for the filed issue (default: p1,scorecard)",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional prefix for the issue title (e.g. '[TEST] ')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the issue JSON (title + body)",
    )
    args = parser.parse_args()

    # Load structured regression data
    perf_path = _resolve_glob(args.perf_regressions_json)
    if not perf_path:
        print(f"No perf regressions JSON found matching: {args.perf_regressions_json}")
        return
    perf_regressions = _load_json(perf_path)

    numerics_path = (
        _resolve_glob(args.numerics_regressions_json)
        if args.numerics_regressions_json
        else None
    )
    numerics_regressions = _load_json(numerics_path)

    if not perf_regressions and not numerics_regressions:
        print("No 2x+ regressions found — skipping issue creation.")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    env_label = "Prod" if args.deployment == "workbench" else "Dev"
    title = f"[Scorecard - {env_label}] 2x+ Regressions Detected - {today}"

    body = build_issue_body(
        perf_regressions,
        numerics_regressions,
        args.run_url,
        args.perf_diff_url,
        args.numerics_diff_url,
        deployment=args.deployment,
    )

    perf_count = len(perf_regressions)
    numerics_count = len(numerics_regressions)
    print(
        f"Found {perf_count} perf regression(s) and "
        f"{numerics_count} numerics regression(s)."
    )

    title = f"{args.title_prefix}{title}"
    labels = [l.strip() for l in args.labels.split(",")]
    output = {"title": title, "body": body, "labels": labels}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Issue JSON written to {args.output}")


if __name__ == "__main__":
    main()
