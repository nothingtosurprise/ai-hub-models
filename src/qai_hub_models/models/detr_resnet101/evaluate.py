# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.detr_resnet101 import MODEL_ID, Model
from qai_hub_models.models.detr_resnet101.export import export_model
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.args import evaluate_parser, get_model_kwargs
from qai_hub_models.utils.evaluate import _load_quant_cpu_onnx, evaluate_on_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import filter_kwargs


def main() -> None:
    warnings.filterwarnings("ignore")
    eval_dataset_classes = Model.get_eval_dataset_classes()
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a16_mixed_int16: [
            TargetRuntime.ONNX,
        ],
    }

    parser = evaluate_parser(
        model_cls=Model,
        supported_dataset_classes=eval_dataset_classes,
        supported_precision_runtimes=supported_precision_runtimes,
        default_device="Samsung Galaxy S25 (Family)",
    )
    args = parser.parse_args()

    model_kwargs = get_model_kwargs(Model, vars(args))
    input_spec_kwargs = filter_kwargs(Model.get_input_spec, vars(args))

    if len(eval_dataset_classes) == 0:
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_model(
            device=args.device,
            target_runtime=args.target_runtime,
            skip_downloading=True,
            skip_profiling=True,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
            **{**model_kwargs, **input_spec_kwargs},
        )
        return

    input_spec: InputSpec | None = None
    torch_model = Model.from_pretrained(**model_kwargs)
    model_executors: dict[str, ExecutableModelProtocol] = {}
    if not args.skip_torch_accuracy:
        model_executors["torch"] = torch_model
        input_spec = torch_model.get_input_spec(**input_spec_kwargs)

    if not args.skip_device_accuracy or args.compute_quant_cpu_accuracy:
        if args.hub_model_id is not None:
            compiled_model: hub.Model = hub.get_model(args.hub_model_id)
        else:
            compiled_result = compile_model_from_args(
                MODEL_ID, args, {**model_kwargs, **input_spec_kwargs}
            )
            assert isinstance(compiled_result, hub.Model)
            compiled_model = compiled_result
        if compiled_model.get_producer() is None:
            raise ValueError(
                "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
            )
        on_device_model = AsyncOnDeviceModel(
            model=compiled_model,
            input_names=list(input_spec) if input_spec else None,
            device=args.device,
            inference_options=args.profile_options,
        )
        if not args.skip_device_accuracy:
            model_executors["on-device"] = on_device_model
        if args.compute_quant_cpu_accuracy and args.precision != Precision.float:
            model_executors["quant cpu"] = _load_quant_cpu_onnx(compiled_model)
        input_spec = on_device_model.get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    evaluate_on_dataset(
        evaluator_func=torch_model.get_evaluator,
        dataset_cls=args.dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        use_cache=args.use_dataset_cache,
    )


if __name__ == "__main__":
    main()
