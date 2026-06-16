# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Protocol, runtime_checkable

import torch
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_model import PretrainedCollectionModel
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec

RUN_MODEL_RETURN_TYPE = list[torch.Tensor] | torch.Tensor

# Generator type for multi-component collection model evaluation pipelines.
# Yields intermediate results per pipeline component; returns the final result.
CollectionModelEvalGenerator = Generator[
    tuple[torch.Tensor, ...] | AsyncOnDeviceResult,
    None,
    tuple[torch.Tensor, ...] | AsyncOnDeviceResult,
]


class BaseCollectionApp(ABC):
    @abstractmethod
    def run_model(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[RUN_MODEL_RETURN_TYPE, ...] | RUN_MODEL_RETURN_TYPE:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model: PretrainedCollectionModel) -> BaseCollectionApp:
        pass


@runtime_checkable
class CollectionAppQuantizeProtocol(Protocol):
    """Protocol for apps that provide calibration data for CollectionModels."""

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: PretrainedCollectionModel,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Produces a numpy dataset to be used for calibration data of a quantize job.

        Parameters
        ----------
        collection_model
            The parent collection model.
        component_name
            The name of the component being calibrated.
        input_specs
            Per-component input specs. If None, uses each component's defaults.
        num_samples
            Number of data samples to use. If not specified, uses
            default specified on dataset.

        Returns
        -------
        DatasetEntries
            Dataset compatible with the format expected by AI Hub Workbench.
        """
        ...


@runtime_checkable
class CollectionAppEvaluateProtocol(Protocol):
    @property
    def uses_ondevice_model(self) -> bool:
        """True if any component is an AsyncOnDeviceModel; False if all are local."""
        return any(isinstance(v, AsyncOnDeviceModel) for v in vars(self).values())

    @classmethod
    def from_components(
        cls, models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel]
    ) -> CollectionAppEvaluateProtocol:
        """
        Create a collection app instance from a list of model components.

        Parameters
        ----------
        models
            List of model components to use in the collection app. Can be either:
            - list[ExecutableModelProtocol]: Models that can be executed locally
            - list[AsyncOnDeviceModel]: Models compiled for on-device execution

        Returns
        -------
        CollectionAppEvaluateProtocol
            A collection app instance initialized with the provided models.
        """
        ...

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """
        Run model(s) for evaluation, handling multi-component pipelines.

        This method is used during evaluation to execute one or more models in sequence,
        potentially with intermediate processing between models.

        Parameters
        ----------
        model_input
            Input data for the first model. Can be either:
            - A generator yielding AsyncOnDeviceResult for on-device execution
            - A tuple of torch.Tensor for local execution
        model_batch_size
            Batch size to use when splitting inputs for model execution.

        Returns
        -------
        CollectionModelEvalGenerator
            - Yields: Output from each model in the pipeline
            - Returns: The final pipeline output
        """
        ...
