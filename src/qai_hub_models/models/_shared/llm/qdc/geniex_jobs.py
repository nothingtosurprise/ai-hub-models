# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from qualcomm_device_cloud_sdk.models import ArtifactType

from qai_hub_models.models._shared.llm.qdc.qdc_jobs import (
    QDCDevice,
    QDCJobs,
)

GENIEX_BENCH_JOB_TIMEOUT = 7200  # 2 hours
_QDC_EXECUTION_MAX_ATTEMPTS = 2

# Versioned URLs follow the geniex release workflow's flat S3 layout
# (<stem>-<vX.Y.Z>.<ext>); the unversioned mirror is refreshed on every
# stable tag and is used when no version is pinned.
_S3_BASE = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex"


def _bench_url(platform_stem: str, ext: str, version: str | None) -> str:
    suffix = f"-{version}" if version else ""
    return f"{_S3_BASE}/geniex-bench-{platform_stem}{suffix}.{ext}"


DEFAULT_CONTEXT_LENGTHS = [512, 1024, 4096]

_N_GEN = 128


def _create_zip(zip_path: str, source_dir: os.PathLike | str) -> None:
    source_dir_str = str(source_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(source_dir_str):
            for fn in files:
                abs_path = os.path.join(root, fn)
                zf.write(abs_path, os.path.relpath(abs_path, source_dir_str))


@dataclass
class GenieXBenchMetrics:
    cell_id: str
    plugin: str
    device_alias: str
    context_length: int
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    gen_tokens: int


class GenieXBenchArtifactHandler(ABC):
    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
    ) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError

    @staticmethod
    def _rewrite_matrix_for_qairt_bundles(
        matrix_rows: list[str],
        qairt_bundles: dict[str, str],
        device_root: str,
    ) -> list[str]:
        # Replace matrix col-4 with the on-device bundle path so qairt plugin
        # skips the model-manager fetch.
        sep = "\\" if "\\" in device_root else "/"
        out: list[str] = []
        for row in matrix_rows:
            parts = row.split("|")
            name = parts[0]
            if name in qairt_bundles:
                parts[3] = f"{device_root}{sep}qairt_bundles{sep}{name}"
            out.append("|".join(parts))
        return out

    @staticmethod
    def _stage_qairt_bundles(
        dest_dir: os.PathLike | str, qairt_bundles: dict[str, str]
    ) -> None:
        base = os.path.join(dest_dir, "qairt_bundles")
        for model_id, bundle_dir in qairt_bundles.items():
            if not os.path.isdir(bundle_dir):
                raise FileNotFoundError(
                    f"QAIRT bundle for {model_id!r} is not a directory: {bundle_dir!r}"
                )
            shutil.copytree(
                bundle_dir, os.path.join(base, model_id), dirs_exist_ok=True
            )


class GenieXBenchAndroidArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/pkg-geniex"

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        pytest_dir = os.path.join(ds_dir, "geniex_pytest")
        ctx_list_str = ",".join(str(c) for c in context_lengths)

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        bench_url = _bench_url("android-arm64", "tar.gz", geniex_version)
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)
        for fn in os.listdir(pytest_dir):
            if fn.endswith(".pyc"):
                continue
            src = os.path.join(pytest_dir, fn)
            if not os.path.isfile(src):
                continue
            with open(src, encoding="utf-8") as f:
                content = f.read()
            if fn.endswith(".py"):
                content = (
                    content.replace("{CTX_LIST}", ctx_list_str)
                    .replace("{ANDROID_BENCH_URL}", bench_url)
                    .replace("{PLUGIN}", plugin)
                    .replace("{N_GEN}", str(_N_GEN))
                )
            out_path = (
                os.path.join(dest_dir, fn)
                if fn == "requirements.txt"
                else os.path.join(test_folder, fn)
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        with open(
            os.path.join(dest_dir, "matrix_rows.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(matrix_rows) + "\n")
        with open(os.path.join(dest_dir, "chipset.txt"), "w", encoding="utf-8") as f:
            f.write(chipset + "\n")

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        _create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchLinuxArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/TestContent"

    @property
    def entry_script(self) -> str:
        return f"/bin/bash {self.DEVICE_ROOT}/run_geniex_bench_linux.sh"

    def _bench_size_flags(
        self, plugin: str, qairt_bundles: dict[str, str] | None
    ) -> str:
        if plugin == "qairt":
            assert qairt_bundles and len(qairt_bundles) == 1
            (name,) = qairt_bundles
            prompt_path = f"{self.DEVICE_ROOT}/qairt_bundles/{name}/sample_prompt.txt"
            return f'-c "$ctx" -n {_N_GEN} --prompt-file "{prompt_path}"'
        return f'-c "$((ctx + {_N_GEN}))" -p "$ctx" -n {_N_GEN}'

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        sh_src = os.path.join(ds_dir, "run_geniex_bench_linux.sh")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(sh_src, encoding="utf-8") as f:
            script = f.read()
        script = (
            script.replace(
                "{LINUX_BENCH_URL}",
                _bench_url("linux-arm64", "tar.gz", geniex_version),
            )
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace("{CTX_LIST}", " ".join(str(c) for c in context_lengths))
            .replace(
                "{BENCH_SIZE_FLAGS}",
                self._bench_size_flags(plugin, qairt_bundles),
            )
        )
        sh_dest = os.path.join(dest_dir, "run_geniex_bench_linux.sh")
        with open(sh_dest, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(sh_dest, 0o755)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        _create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchWindowsArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "C:\\Temp\\TestContent"

    @property
    def entry_script(self) -> str:
        return f"{self.DEVICE_ROOT}\\run_geniex_bench_windows.ps1"

    def _bench_size_flags_args(
        self, plugin: str, qairt_bundles: dict[str, str] | None
    ) -> str:
        if plugin == "qairt":
            assert qairt_bundles and len(qairt_bundles) == 1
            (name,) = qairt_bundles
            path = f"{self.DEVICE_ROOT}\\qairt_bundles\\{name}\\sample_prompt.txt"
            return f'"-c", "$ctx", "-n", "{_N_GEN}", "--prompt-file", "{path}",'
        return f'"-c", "$($ctx + {_N_GEN})", "-p", "$ctx", "-n", "{_N_GEN}",'

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        ps1_src = os.path.join(ds_dir, "run_geniex_bench_windows.ps1")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(ps1_src, encoding="utf-8") as f:
            script = f.read()
        script = (
            script.replace(
                "{WINDOWS_BENCH_URL}",
                _bench_url("windows-arm64", "zip", geniex_version),
            )
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace("{CTX_LIST}", ",".join(str(c) for c in context_lengths))
            .replace(
                "{BENCH_SIZE_FLAGS_ARGS}",
                self._bench_size_flags_args(plugin, qairt_bundles),
            )
        )
        with open(
            os.path.join(dest_dir, "run_geniex_bench_windows.ps1"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(script)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        _create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchQDCJobs(QDCJobs):
    def _get_artifact_handler(
        self, qdc_device: QDCDevice
    ) -> GenieXBenchArtifactHandler:
        if qdc_device.windows_platform:
            return GenieXBenchWindowsArtifactHandler()
        if qdc_device.iot_platform:
            return GenieXBenchLinuxArtifactHandler()
        if qdc_device.mobile_platform:
            return GenieXBenchAndroidArtifactHandler()
        raise NotImplementedError(
            "geniex-bench currently supports Windows (Snapdragon X / X2 "
            "Elite), IoT Linux (Dragonwing IQ-9075 EVK), and Android "
            "(Snapdragon 8 Elite QRD / Gen 5 QRD). "
            f"Device {qdc_device.device.name!r} is none of these."
        )

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        chipset: str,
        matrix_rows: list[str],
        plugin: str,
        context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
        qairt_bundles: dict[str, str] | None = None,
        geniex_version: str | None = None,
    ) -> tuple[list[str], str | None]:
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        handler = self._get_artifact_handler(qdc_device)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = handler.create_artifact(
                curr_dirname,
                tmpdir,
                chipset,
                matrix_rows,
                context_lengths,
                plugin,
                qairt_bundles,
                geniex_version,
            )
            artifact = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
        return [artifact], handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
        save_results_dir: str | None = None,
    ) -> list[GenieXBenchMetrics]:
        metrics: list[GenieXBenchMetrics] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for job_log in job_log_files:
                target = os.path.join(tmpdir, "logs", f"{job_log.filename}.zip")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                self.download_job_log_files(job_log.filename, target)
                try:
                    shutil.unpack_archive(target, tmpdir, "zip")
                except shutil.ReadError:
                    continue

            for root, _, files in os.walk(tmpdir):
                for fn in sorted(files):
                    if not fn.endswith(".json"):
                        continue
                    path = os.path.join(root, fn)
                    parsed = self._parse_cell_metrics(path)
                    if parsed is None:
                        continue
                    metrics.append(parsed)
                    if save_results_dir:
                        rel = os.path.relpath(path, tmpdir)
                        dest = os.path.join(save_results_dir, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        shutil.copy(path, dest)

        if metrics:
            print(f"Parsed {len(metrics)} geniex-bench cells:")
            for m in metrics:
                print(
                    f"  [{m.cell_id} ctx={m.context_length}] "
                    f"decode={m.decode_tps:.2f} tok/s, prefill={m.prefill_tps:.2f} tok/s, "
                    f"TTFT={m.ttft_ms:.1f} ms"
                )
        else:
            print("Warning: no geniex-bench schema_v3 results found in logs.")

        return metrics

    @staticmethod
    def _parse_cell_metrics(path: str) -> GenieXBenchMetrics | None:
        try:
            with open(path, encoding="utf-8") as f:
                cell = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(cell, dict) or cell.get("schema_version") != "3":
            return None
        agg = cell.get("agg") or {}
        params = cell.get("params") or {}

        def med(key: str) -> float | None:
            entry = agg.get(key) or {}
            return entry.get("median")

        ttft = med("ttft_ms")
        prefill = med("prefill_tps")
        decode = med("decode_tps")
        if ttft is None or prefill is None or decode is None:
            return None

        cell_id = cell.get("cell_id") or ""
        _, sep, suffix = cell_id.rpartition("-c")
        ctx = int(suffix) if sep and suffix.isdigit() else int(params.get("n_ctx") or 0)
        if ctx == 0:
            return None

        return GenieXBenchMetrics(
            cell_id=cell_id,
            plugin=cell.get("plugin") or "",
            device_alias=cell.get("device") or "",
            context_length=ctx,
            ttft_ms=float(ttft),
            prefill_tps=float(prefill),
            decode_tps=float(decode),
            prompt_tokens=int((agg.get("prompt_tokens") or {}).get("median") or 0),
            gen_tokens=int((agg.get("gen_tokens") or {}).get("median") or 0),
        )


def _hf_repo(model_url: str) -> str:
    if "huggingface.co/" not in model_url:
        raise ValueError(f"Only HuggingFace URLs are supported: {model_url}")
    parts = model_url.split("huggingface.co/")[1].split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse HF repo from: {model_url}")
    if not model_url.endswith(".gguf"):
        raise ValueError(f"Expected .gguf URL, got: {model_url}")
    return f"{parts[0]}/{parts[1]}"


def submit_geniex_bench_to_qdc_device(
    api_token: str,
    hub_device_name: str,
    chipset: str,
    model_rows: list[tuple[str, str]],
    context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
    plugin: str = "llama_cpp",
    device_alias: str = "hybrid",
    job_name: str = "geniex-bench",
    save_results_dir: str | None = None,
    geniex_version: str | None = None,
    llamacpp_quant: str | None = None,
) -> list[GenieXBenchMetrics]:
    # plugin="qairt": model_ref is a local genie bundle dir (uploaded
    # under qairt_bundles/). plugin="llama_cpp": model_ref is a HF GGUF URL
    # and llamacpp_quant is the GGUF quant token (e.g. "q4_0", "mxfp4"),
    # taken from release-assets.yaml's precision key.
    if plugin == "llama_cpp" and not llamacpp_quant:
        raise ValueError("llamacpp_quant is required when plugin='llama_cpp'.")
    qdc_device = QDCDevice(hub_device_name)

    matrix_rows: list[str] = []
    qairt_bundles: dict[str, str] = {}
    for name, ref in model_rows:
        if plugin == "qairt":
            qairt_bundles[name] = ref
            model_id = name
        else:
            model_id = f"{_hf_repo(ref)}:{llamacpp_quant}"
        matrix_rows.append(f"{name}|{plugin}|{device_alias}|{model_id}||")

    geniex_job = GenieXBenchQDCJobs(
        api_key=api_token,
        app_name_header="GenieXBenchQDCJobApp",
    )

    job_artifacts, entry_script = geniex_job.add_job_artifacts(
        qdc_device,
        chipset,
        matrix_rows,
        plugin,
        context_lengths=context_lengths,
        qairt_bundles=qairt_bundles or None,
        geniex_version=geniex_version,
    )

    last_failure_reason: str | None = None
    for attempt in range(1, _QDC_EXECUTION_MAX_ATTEMPTS + 1):
        job_id = geniex_job.submit_automated_job(
            qdc_device, job_artifacts, entry_script, job_name=job_name
        )
        if job_id is None:
            raise RuntimeError("Job submission failed.")

        print(f"Submitted QDC job with ID: {job_id}")
        job_status = geniex_job.status(job_id, timeout=GENIEX_BENCH_JOB_TIMEOUT)
        job_result = geniex_job.result(job_id)
        print(
            f"QDC job {job_id} completed with status: {job_status}, "
            f"result: {job_result}"
        )

        if job_result is not None and job_result != "Successful":
            last_failure_reason = (
                f"QDC job {job_id} on device '{hub_device_name}' finished with "
                f"status='{job_status}', result='{job_result}'"
            )
            print(
                f"[attempt {attempt}/{_QDC_EXECUTION_MAX_ATTEMPTS}] "
                f"{last_failure_reason}"
            )
            if attempt < _QDC_EXECUTION_MAX_ATTEMPTS:
                print("Retrying QDC job execution...")
                continue
            raise RuntimeError(
                f"{last_failure_reason} after {_QDC_EXECUTION_MAX_ATTEMPTS} "
                f"attempt(s). The device-side job did not complete successfully; "
                f"check the QDC job logs for details."
            )

        geniex_job.log_upload_status(job_id)
        # The file listing lags log-upload-status on the QDC backend, so wait
        # for it to populate -- otherwise a successful job yields no metrics.
        job_log_files = geniex_job.get_job_log_files(job_id, wait_for_logs=True)

        # An empty listing here means the job succeeded but its logs never
        # became retrievable; retry the whole job (transient) rather than
        # returning empty metrics.
        if not job_log_files:
            last_failure_reason = (
                f"QDC job {job_id} on device '{hub_device_name}' reported result="
                f"'{job_result}' but produced no retrievable log files"
            )
            print(
                f"[attempt {attempt}/{_QDC_EXECUTION_MAX_ATTEMPTS}] "
                f"{last_failure_reason}"
            )
            if attempt < _QDC_EXECUTION_MAX_ATTEMPTS:
                print("Retrying QDC job execution...")
                continue
            raise RuntimeError(
                f"{last_failure_reason} after {_QDC_EXECUTION_MAX_ATTEMPTS} "
                f"attempt(s). Check the QDC job logs for details."
            )

        return geniex_job.compute_metrics(
            job_log_files, save_results_dir=save_results_dir
        )

    raise RuntimeError(
        f"QDC job execution failed for device '{hub_device_name}': "
        f"{last_failure_reason}"
    )
