# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelFileMetadata, ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.export import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    _ensure_int_list,
    _parse_comma_separated_ints,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.qwen2_5_vl_7b_instruct import MODEL_ID, Model
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import Qwen2_5_VL_7B_PartBase
from qai_hub_models.utils.args import (
    export_parser,
    get_export_model_name,
    get_model_kwargs,
)
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    check_unpublished_model_warning,
)
from qai_hub_models.utils.base_model import PretrainedCollectionModel
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.export_result import (
    ComponentGroup,
    ExportResult,
    LegacyCollectionExportResult,
)
from qai_hub_models.utils.export_without_hub_access import export_without_hub_access
from qai_hub_models.utils.onnx.helpers import download_and_unzip_workbench_onnx_model
from qai_hub_models.utils.path_helpers import get_next_free_path
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
    print_tool_versions,
)
from qai_hub_models.utils.qai_hub_helpers import (
    assert_success_and_get_target_models,
    can_access_qualcomm_ai_hub,
)

DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048]


def compile_model(
    model: PretrainedCollectionModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    output_path: Path,
    components: list[str] | None = None,
    extra_options: str = "",
) -> dict[str, list[hub.client.CompileJob]]:
    compile_jobs: dict[str, list[hub.client.CompileJob]] = {}
    for component_name in components or Model.component_class_names:
        component = model.components[component_name]

        input_spec = component.get_input_spec()
        # Trace the model
        model_to_compile = component.convert_to_hub_source_model(
            target_runtime, output_path, input_spec
        )

        compile_jobs[component_name] = []

        # Upload model once, reuse for all compile specs (token + prompt graphs)
        uploaded_model = hub.upload_model(model_to_compile)  # type: ignore[arg-type]

        if isinstance(component, Qwen2_5_VL_7B_PartBase):
            compile_specs = component.get_compile_specs()
        else:
            # Vision encoder: single compile spec, no graph name
            compile_specs = [(input_spec, None)]

        for input_spec, graph_name in compile_specs:
            context_graph_name = graph_name or f"{MODEL_ID}_{component_name.lower()}"
            model_compile_options = component.get_hub_compile_options(
                target_runtime,
                Precision.w4a16,
                extra_options,
                device,
                context_graph_name=context_graph_name,
            )
            print(f"Optimizing model {component_name} to run on-device")
            submitted_compile_job = hub.submit_compile_job(
                model=uploaded_model,
                input_specs=input_spec,  # type: ignore[arg-type]
                device=device,
                name=f"{model_name}_{component_name}",
                options=model_compile_options,
            )
            compile_jobs[component_name].append(
                cast(hub.client.CompileJob, submitted_compile_job)
            )
    return compile_jobs


def link_model(
    compiled_models: dict[str, list[hub.Model]],
    device: hub.Device,
    model_name: str,
    model: PretrainedCollectionModel,
    target_runtime: TargetRuntime,
) -> dict[str, hub.client.LinkJob]:
    """Link compiled DLCs to context binary for AOT."""
    assert target_runtime.is_aot_compiled, (
        f"link_model() requires an AOT runtime, got {target_runtime}"
    )
    link_jobs: dict[str, hub.client.LinkJob] = {}
    for component_name, model_list in compiled_models.items():
        component = model.components[component_name]

        link_options = component.get_hub_link_options(target_runtime)
        print(f"Linking {component_name} to context binary")
        link_jobs[component_name] = hub.submit_link_job(
            cast(list, model_list),
            device=device,
            name=f"{model_name}_{component_name}",
            options=link_options,
        )
    return link_jobs


def profile_model(
    model_name: str,
    device: hub.Device,
    options: dict[str, list[tuple[str, str | None]]],
    target_models: dict[str, hub.Model],
    components: list[str] | None = None,
) -> dict[str, list[hub.client.ProfileJob]]:
    profile_jobs: dict[str, list[hub.client.ProfileJob]] = {}
    for component_name in components or Model.component_class_names:
        profile_jobs[component_name] = []
        for opts, graph_name in options.get(component_name, []):
            job_name = (
                f"{model_name}_{component_name}"
                if graph_name is None
                else f"{model_name}_{component_name}_{graph_name}"
            )
            print(f"Profiling model {component_name} on a hosted device.")
            submitted_profile_job = hub.submit_profile_job(
                model=target_models[component_name],
                device=device,
                name=job_name,
                options=opts,
            )
            profile_jobs[component_name].append(
                cast(hub.client.ProfileJob, submitted_profile_job)
            )
    return profile_jobs


def inference_model(
    inputs: ComponentGroup[SampleInputsType],
    model_name: str,
    device: hub.Device,
    options: ComponentGroup[str],
    target_models: ComponentGroup[hub.Model],
) -> ComponentGroup[hub.client.InferenceJob]:
    inference_jobs: dict[str, hub.client.InferenceJob] = {}
    for component_name in target_models:
        print(
            f"Running inference for {component_name} on a hosted device with example inputs."
        )
        submitted_inference_job = hub.submit_inference_job(
            model=target_models[component_name],
            inputs=inputs[component_name],
            device=device,
            name=f"{model_name}_{component_name}",
            options=options.get(component_name, ""),
        )
        inference_jobs[component_name] = cast(
            hub.client.InferenceJob, submitted_inference_job
        )
    return ComponentGroup(inference_jobs)


def download_model(
    output_dir: os.PathLike | str,
    model: PretrainedCollectionModel,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_models: dict[str, hub.Model],
    zip_assets: bool,
) -> Path:
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        # Download models and capture filenames, then generate metadata
        model_file_metadata = {}
        for component_name, target_model in target_models.items():
            if target_model.model_type == hub.SourceModelType.ONNX:
                onnx_result = download_and_unzip_workbench_onnx_model(
                    target_model, dst_path, component_name
                )
                model_file_name = onnx_result.onnx_graph_name
            else:
                downloaded_path = target_model.download(
                    os.path.join(dst_path, component_name)
                )
                model_file_name = os.path.basename(downloaded_path)

            # Generate metadata using the actual downloaded filename
            model_file_metadata[model_file_name] = ModelFileMetadata.from_hub_model(
                target_model
            )

        # Extract and save metadata alongside downloaded model
        metadata_path = dst_path / "metadata.json"
        model_metadata = ModelMetadata(
            model_id=MODEL_ID,
            model_name="Qwen2.5-VL-7B-Instruct",
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files=model_file_metadata,
        )

        # Dump supplementary files into the model folder
        model.write_supplementary_files(dst_path, model_metadata)

        model_metadata.to_json(metadata_path)
        if zip_assets:
            output_path = Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        else:
            shutil.move(dst_path, output_path)

    return output_path


def export_model(
    device: hub.Device,
    components: list[str] | None = None,
    precision: Precision = Precision.w4a16,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    compile_options: str = "",
    profile_options: str = "",
    fetch_static_assets: str | None = None,
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> LegacyCollectionExportResult:
    """
    This function executes the following recipe:

        1. Instantiates a PyTorch model and converts it to a traced TorchScript format
        2. Compiles the model to an asset that can be run on device
        3. Profiles the model performance on a real device
        4. Inferences the model on sample inputs
        5. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
        6. Downloads the model asset to the local directory
        7. Summarizes the results from profiling and inference

    Each of the last 5 steps can be optionally skipped using the input options.

    Parameters
    ----------
    device
        Device for which to export the model (e.g., hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    components
        List of sub-components of the model that will be exported.
        Each component is compiled and profiled separately.
        Defaults to all components of the CollectionModel if not specified.
    precision
        The precision to which this model should be quantized.
        Quantization is skipped if the precision is float.
    skip_profiling
        If set, skips profiling of compiled model on real devices.
    skip_inferencing
        If set, skips computing on-device outputs from sample data.
    skip_downloading
        If set, skips downloading of compiled model.
    skip_summary
        If set, skips waiting for and summarizing results
        from profiling and inference.
    output_dir
        Directory to store generated assets (e.g. compiled model).
        Defaults to `<cwd>/export_assets`.
    target_runtime
        Which on-device runtime to target. Default is TFLite.
    compile_options
        Additional options to pass when submitting the compile job.
    profile_options
        Additional options to pass when submitting the profile job.
    fetch_static_assets
        If set, known assets are fetched from the given version rather than re-computing them. Can be passed as "latest" or "v<version>".
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_pretrained`

    Returns
    -------
    LegacyCollectionExportResult
        A Mapping from component_name to:
            * A CompileJob object containing metadata about the compile job submitted to hub.
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
        * The path to the downloaded model folder (or zip), or None if one or more of: skip_downloading is True, fetch_static_assets is set, or AI Hub Workbench is not accessible
    """
    model_name = get_export_model_name(
        Model, MODEL_ID, precision, additional_model_kwargs
    )

    output_path = Path(output_dir or Path.cwd() / "export_assets")
    assert precision in [
        Precision.w4a16,
    ], f"Precision {precision!s} is not supported by {model_name}"
    component_arg = components
    components = components or Model.component_class_names
    for component_name in components:
        if component_name not in Model.component_class_names:
            raise ValueError(f"Invalid component {component_name}.")
    if fetch_static_assets or not can_access_qualcomm_ai_hub():
        static_model_path = export_without_hub_access(
            MODEL_ID,
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            precision,
            compile_options + profile_options,
            component_arg,
            qaihm_version_tag=fetch_static_assets,
        )
        return LegacyCollectionExportResult(
            components={
                component_name: ExportResult() for component_name in components
            },
            download_path=static_model_path,
        )

    hub_device = hub.get_devices(
        name=device.name, attributes=device.attributes, os=device.os
    )[-1]
    chipset_attr = next(
        (attr for attr in hub_device.attributes if "chipset" in attr), None
    )
    chipset = chipset_attr.split(":")[-1] if chipset_attr else None

    # 1. Instantiates a PyTorch model and converts it to a traced TorchScript format
    model_kwargs = dict(**additional_model_kwargs, precision=precision)

    # Normalize sequence_length / context_length to lists
    sequence_lengths = _ensure_int_list(
        model_kwargs.pop("sequence_length", DEFAULT_EXPORT_SEQUENCE_LENGTHS)
    )
    model_kwargs["sequence_length"] = sequence_lengths[0]

    context_lengths = _ensure_int_list(model_kwargs.pop("context_length", [4096]))
    model_kwargs["context_length"] = context_lengths[0]
    model_kwargs["context_lengths"] = context_lengths

    model = Model.from_pretrained(**get_model_kwargs(Model, model_kwargs))
    model._hub_device = hub_device  # Store for write_supplementary_files

    # 2. Compiles the model to an asset that can be run on device
    compile_jobs: dict[str, list[hub.client.CompileJob]] = compile_model(
        model,
        model_name,
        device,
        target_runtime,
        output_path=output_path,
        components=components,
        extra_options=compile_options,
    )

    link_jobs: dict[str, hub.client.LinkJob] | None = None
    target_models: dict[str, hub.Model]
    if target_runtime.uses_hub_link:
        compiled_models: dict[str, list[hub.Model]] = {}
        for comp_name, jobs in compile_jobs.items():
            compiled_models[comp_name] = []
            for job in jobs:
                target_model = job.get_target_model()
                assert target_model is not None, f"Compile job failed: {job}"
                compiled_models[comp_name].append(target_model)
        link_jobs = link_model(
            compiled_models,
            device,
            model_name,
            model,
            target_runtime,
        )
        # Extract target models from link jobs for profile/inference
        target_models = assert_success_and_get_target_models(ComponentGroup(link_jobs))
    else:
        # For JIT runtimes, extract models from compile jobs
        flat_jobs = {k: v[0] for k, v in compile_jobs.items()}
        target_models = assert_success_and_get_target_models(ComponentGroup(flat_jobs))

    # Build profile options; one entry per context graph so each gets its own profile job
    per_component_profile_options: dict[str, list[tuple[str, str | None]]] = {}
    per_component_inference_options: dict[str, str] = {}
    for component_name in components:
        component = model.components[component_name]

        base_opts = component.get_hub_profile_options(
            target_runtime=target_runtime,
            other_profile_options=profile_options,
        )
        per_component_inference_options[component_name] = base_opts
        if isinstance(component, Qwen2_5_VL_7B_PartBase):
            compile_specs = component.get_compile_specs()
            graph_names = [name for _, name in compile_specs if name is not None]
        else:
            graph_names = []
        if graph_names:
            per_component_profile_options[component_name] = [
                (base_opts + f" --qnn_options context_enable_graphs={name}", name)
                for name in graph_names
            ]
        else:
            per_component_profile_options[component_name] = [(base_opts, None)]

    # 3. Profiles the model performance on a real device
    profile_jobs: dict[str, list[hub.client.ProfileJob]] = {}
    if not skip_profiling:
        profile_jobs = profile_model(
            model_name,
            device,
            per_component_profile_options,
            target_models,
            components,
        )

    # 4. Inferences the model on sample inputs
    inference_result: ComponentGroup[hub.client.InferenceJob] | None = None
    if not skip_inferencing:
        inference_result = inference_model(
            model.sample_inputs(
                use_channel_last_format=target_runtime.channel_last_native_execution
            ),
            model_name,
            device,
            ComponentGroup(per_component_inference_options),
            target_models,
        )

    # 5. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
    tool_versions: ToolVersions | None = None
    tool_versions_are_from_device_job = False
    if not skip_summary or not skip_downloading:
        first_profile_jobs_list = (
            next(iter(profile_jobs.values()), []) if profile_jobs else []
        )
        first_profile_job = (
            first_profile_jobs_list[0] if first_profile_jobs_list else None
        )
        inference_job = (
            next(iter(inference_result.values())) if inference_result else None
        )
        first_compile_jobs = next(iter(compile_jobs.values()), [])
        compile_job = first_compile_jobs[0] if first_compile_jobs else None
        if first_profile_job is not None and first_profile_job.wait():
            tool_versions = ToolVersions.from_job(first_profile_job)
            tool_versions_are_from_device_job = True
        elif inference_job is not None and inference_job.wait():
            tool_versions = ToolVersions.from_job(inference_job)
            tool_versions_are_from_device_job = True
        elif compile_job and compile_job.wait():
            tool_versions = ToolVersions.from_job(compile_job)

    # 6. Downloads the model asset to the local directory
    downloaded_model_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        model_directory = output_path / ASSET_CONFIG.get_release_asset_name(
            MODEL_ID, target_runtime, precision, chipset
        )
        downloaded_model_path = download_model(
            model_directory,
            model,
            target_runtime,
            precision,
            tool_versions,
            target_models,
            zip_assets,
        )

    # 7. Summarizes the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for component_name in components:
            for pj in profile_jobs[component_name]:
                assert pj.wait().success, "Job failed: " + pj.url
                profile_data: dict[str, Any] = pj.download_profile()
                print_profile_metrics_from_job(pj, profile_data)

    if not skip_summary and not skip_inferencing and inference_result is not None:
        for component_name in components:
            component = model.components[component_name]

            ij = inference_result[component_name]

            # Skip torch inference comparison for components loaded from
            # quantized checkpoints (ONNX-only, no PyTorch forward).
            try:
                sample_inputs = component.sample_inputs(use_channel_last_format=False)
                torch_out = torch_inference(
                    component,
                    sample_inputs,
                    return_channel_last_output=target_runtime.channel_last_native_execution,
                )
            except RuntimeError:
                print(
                    f"Skipping torch inference comparison for {component_name} "
                    f"(no PyTorch forward available)."
                )
                continue

            assert ij.wait().success, "Job failed: " + ij.url
            ij_output = ij.download_output_data()
            assert ij_output is not None
            print_inference_metrics(
                ij, ij_output, torch_out, component.get_output_names()
            )

    if not skip_summary:
        print_tool_versions(tool_versions, tool_versions_are_from_device_job)

    # Clean up intermediate .aimet/.onnx staging directories
    for entry in output_path.iterdir():
        if entry.is_dir() and entry.suffix in (".aimet", ".onnx"):
            shutil.rmtree(entry)

    if downloaded_model_path:
        print(f"{model_name} was saved to {downloaded_model_path}\n")

    return LegacyCollectionExportResult(
        components={
            component_name: ExportResult(
                compile_job=compile_jobs[component_name][0],
                link_job=link_jobs.get(component_name)
                if (target_runtime.uses_hub_link and link_jobs)
                else None,
                inference_job=inference_result.get(component_name)
                if inference_result
                else None,
                profile_job=profile_jobs.get(component_name, [None])[0],
            )
            for component_name in components
        },
        download_path=downloaded_model_path,
        tool_versions=tool_versions,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    if not check_unpublished_model_warning():
        return
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.w4a16: [
            TargetRuntime.GENIE,
        ],
    }

    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device="Samsung Galaxy S25 (Family)",
    )

    # Override --context-length to accept comma-separated values
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == "context_length":
            action.type = _parse_comma_separated_ints
            action.default = DEFAULT_CONTEXT_LENGTHS
            action.help = (
                "Context length(s) for the model. "
                "Pass a single value (e.g. 4096) or a comma-separated list "
                "(e.g. 512,1024,2048,3072,4096) to export models for "
                "multiple context lengths."
            )

    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
