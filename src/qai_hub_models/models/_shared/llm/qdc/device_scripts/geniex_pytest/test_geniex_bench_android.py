# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device geniex-bench scorecard run for QDC Android phones.

Mirrors run_android.py's robustness pattern: network preflight, single
retry per cell, and explicit on-device output check.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest

HOST_ARTIFACT_ROOT = "/qdc/appium"
HOST_QAIRT_BUNDLES = f"{HOST_ARTIFACT_ROOT}/qairt_bundles"
HOST_ROWS = f"{HOST_ARTIFACT_ROOT}/matrix_rows.txt"
HOST_CHIPSET = f"{HOST_ARTIFACT_ROOT}/chipset.txt"
HOST_STAGE = f"{HOST_ARTIFACT_ROOT}/_stage"
HOST_BUNDLE = f"{HOST_STAGE}/bundle"

DEVICE_BUNDLE = "/data/local/tmp/pkg-geniex"
DEVICE_QAIRT_BUNDLES = f"{DEVICE_BUNDLE}/qairt_bundles"
DEVICE_MM_CACHE = "/data/local/tmp/geniex-cache"
DEVICE_QDC_LOGS = "/data/local/tmp/QDC_logs"
DEVICE_RESULTS = f"{DEVICE_QDC_LOGS}/results"

CTXS = tuple(int(c) for c in "{CTX_LIST}".split(","))
ANDROID_BENCH_URL = "{ANDROID_BENCH_URL}"
PLUGIN = "{PLUGIN}"
N_GEN = int("{N_GEN}")


def adb(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    # adb shell drops the remote exit code; recover it via __RC__ trailer.
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    stdout, rc = raw.stdout, raw.returncode
    lines = stdout.rstrip("\n").split("\n") if stdout else []
    if lines and lines[-1].startswith("__RC__:"):
        try:
            rc = int(lines[-1][7:])
            stdout = "\n".join(lines[:-1]) + "\n"
        except ValueError:
            pass
    print(stdout)
    result = subprocess.CompletedProcess(raw.args, rc, stdout=stdout)
    if check:
        assert rc == 0, f"adb command failed (exit {rc}): {cmd}"
    return result


def _preflight_network() -> None:
    # QDC phones occasionally boot with degraded wifi; fail fast instead
    # of letting stage_bundle() hang silently on a stalled fetch. HEAD the
    # actual asset (a bucket-root probe returns 403 on S3 list-bucket).
    preflight = subprocess.run(
        [
            "adb",
            "shell",
            f"curl -sSI -o /dev/null -w '%{{http_code}}' --max-time 15 "
            f"{ANDROID_BENCH_URL}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    http_code = preflight.stdout.strip()
    if preflight.returncode != 0 or not http_code.startswith(("2", "3")):
        pytest.fail(
            f"Device cannot reach {ANDROID_BENCH_URL} (rc={preflight.returncode}, "
            f"http_code={http_code!r}, stderr={preflight.stderr!r}). "
            "Likely QDC device-side wifi failure — file a QDC infra ticket and re-run."
        )


def stage_bundle() -> None:
    if os.path.exists(os.path.join(HOST_BUNDLE, "bin", "geniex-bench")):
        return
    os.makedirs(HOST_STAGE, exist_ok=True)

    print(f"Fetching {ANDROID_BENCH_URL}")
    with urllib.request.urlopen(ANDROID_BENCH_URL) as resp:
        bench_tgz = resp.read()
    with tarfile.open(fileobj=io.BytesIO(bench_tgz), mode="r:gz") as tf:
        members = tf.getmembers()
        top = members[0].name.split("/", 1)[0] if members else ""
        for m in members:
            if not m.name.startswith(top + "/") or m.name == top + "/":
                continue
            rel = m.name[len(top) + 1 :]
            dst = os.path.join(HOST_BUNDLE, rel)
            real_dst = os.path.realpath(dst)
            real_base = os.path.realpath(HOST_BUNDLE)
            if not real_dst.startswith(real_base + os.sep):
                raise ValueError(f"Refusing unsafe tar member path: {m.name!r}")
            if m.isdir():
                os.makedirs(dst, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            os.chmod(dst, m.mode | 0o400)


def push_bundle() -> None:
    # QDC reflashes the phone every session, so always push.
    stage_bundle()
    subprocess.run(["adb", "push", HOST_BUNDLE, DEVICE_BUNDLE], check=True)
    adb(f"find {DEVICE_BUNDLE}/bin -type f -exec chmod 755 {{}} +")
    adb(f"cp {DEVICE_BUNDLE}/lib/qairt/htp-files/*.so {DEVICE_BUNDLE}/lib/")
    adb(f"cp {DEVICE_BUNDLE}/lib/llama_cpp/*.so {DEVICE_BUNDLE}/lib/")


def _run_bench(
    ctx: int, env: str, tsv_path: str, chipset: str, bundle_name: str | None
) -> int:
    if PLUGIN == "qairt":
        assert bundle_name is not None
        size_flags = (
            f"-c {ctx} -n {N_GEN} "
            f"--prompt-file {DEVICE_QAIRT_BUNDLES}/{bundle_name}/sample_prompt.txt"
        )
    else:
        size_flags = f"-c {ctx + N_GEN} -p {ctx} -n {N_GEN}"
    cmd = (
        f"cd {DEVICE_BUNDLE} && {env} ./bin/geniex-bench "
        f"--matrix-file {tsv_path} --output-json-dir {DEVICE_RESULTS} -r 3 "
        f"{size_flags} "
        f"--mm-data-dir {DEVICE_MM_CACHE} --chipset '{chipset}' "
        f"2>>{DEVICE_QDC_LOGS}/geniex_bench_stderr.log"
    )
    res = adb(cmd, check=False)
    if res.returncode == 0:
        return 0
    print(f"geniex-bench ctx={ctx} failed (rc={res.returncode}); retrying once.")
    return adb(cmd, check=False).returncode


def test_scorecard() -> None:
    _preflight_network()
    push_bundle()
    adb(f"mkdir -p {DEVICE_MM_CACHE} {DEVICE_RESULTS}")
    bundle_name: str | None = None
    if PLUGIN == "qairt" and os.path.isdir(HOST_QAIRT_BUNDLES):
        adb(f"mkdir -p {DEVICE_QAIRT_BUNDLES}")
        subprocess.run(
            ["adb", "push", f"{HOST_QAIRT_BUNDLES}/.", DEVICE_QAIRT_BUNDLES],
            check=True,
        )
        names = [
            d
            for d in os.listdir(HOST_QAIRT_BUNDLES)
            if os.path.isdir(os.path.join(HOST_QAIRT_BUNDLES, d))
        ]
        assert len(names) == 1, f"expected one qairt bundle, got {names}"
        bundle_name = names[0]

    chipset = Path(HOST_CHIPSET).read_text().strip()
    rows = [r for r in Path(HOST_ROWS).read_text().splitlines() if r.strip()]
    tsv_by_ctx: dict[int, list[str]] = {ctx: [] for ctx in CTXS}
    for row in rows:
        name, plugin, devs, model_id, vlm, _image = row.split("|")
        for d in devs.split(","):
            for ctx in CTXS:
                tsv_by_ctx[ctx].append(
                    f"{name}-{plugin}-{d}-c{ctx}\t{plugin}\t{d}\t{model_id}"
                    f"\t\t\t\t{vlm}"
                )
    assert any(tsv_by_ctx.values()), "no model rows produced"

    lib = f"{DEVICE_BUNDLE}/lib"
    env = (
        f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
        f"ADSP_LIBRARY_PATH={lib} "
        f"GENIEX_PLUGIN_PATH={lib}"
    )
    failures = []
    for ctx in CTXS:
        tsv_path = f"/data/local/tmp/matrix-{ctx}.tsv"
        adb(
            "printf '%s\\n' "
            + " ".join(f"'{ln}'" for ln in tsv_by_ctx[ctx])
            + f" > {tsv_path}"
        )
        if _run_bench(ctx, env, tsv_path, chipset, bundle_name) != 0:
            failures.append(ctx)

    # Confirm cell JSONs exist; adb hides on-device exit codes.
    ls = adb(f"ls -l {DEVICE_RESULTS}", check=False)
    count_proc = adb(f"ls {DEVICE_RESULTS} | wc -l", check=False)
    count = (
        int(count_proc.stdout.strip().split()[-1]) if count_proc.stdout.strip() else 0
    )
    if failures or count == 0:
        pytest.fail(
            f"geniex-bench produced no usable output (failed ctxs={failures}, "
            f"cell_json_count={count}).\n--- {DEVICE_RESULTS} ---\n{ls.stdout}"
        )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    )
