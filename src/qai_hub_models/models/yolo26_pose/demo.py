# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.yolo.demo import yolo_pose_estimation_demo
from qai_hub_models.models.yolo26_pose.app import Yolo26PoseApp
from qai_hub_models.models.yolo26_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Yolo26PoseDetector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolo26_pose_demo.jpg"
)


def main(is_test: bool = False) -> None:
    yolo_pose_estimation_demo(
        model_type=Yolo26PoseDetector,
        model_id=MODEL_ID,
        app_type=Yolo26PoseApp,
        default_image=IMAGE_ADDRESS,
        output_filename="yolo26_pose_demo_output.png",
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
