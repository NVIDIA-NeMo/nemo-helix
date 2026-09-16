# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

from nhx.automodel.entities.values import FinetuningType
from nhx.automodel.tasks.training.backends.checkpoints import ModelType, process_checkpoint
from pytest_mock import MockerFixture


def test_process_checkpoint_merges_cross_encoder_lora_without_onnx(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    customizer_config = MagicMock()
    customizer_config.training.finetuning_type = FinetuningType.LORA_MERGED
    customizer_config.model.path = str(tmp_path / "base")
    customizer_config.model.precision = None

    merge_cross = mocker.patch("nhx.automodel.tasks.training.backends.checkpoints.merge_lora_cross_encoder_adapter")
    merge_embed = mocker.patch("nhx.automodel.tasks.training.backends.checkpoints.merge_lora_embedding_adapter")
    merge_llm = mocker.patch("nhx.automodel.tasks.training.backends.checkpoints.merge_lora_adapter")
    mocker.patch("nhx.automodel.tasks.training.backends.checkpoints.fix_fsdp2_architecture")
    export_onnx = mocker.patch("nhx.automodel.tasks.training.backends.checkpoints.export_onnx")
    mocker.patch("nhx.automodel.tasks.training.backends.checkpoints._restructure_embedding_output")
    mocker.patch(
        "nhx.automodel.tasks.training.backends.checkpoints.extract_precision_from_model_config",
        return_value=None,
    )

    checkpoint_path = tmp_path / "adapter"
    output_path = tmp_path / "output"
    process_checkpoint(
        checkpoint_path,
        output_path,
        customizer_config,
        model_type=ModelType.CROSS_ENCODER,
    )

    merge_cross.assert_called_once_with(
        adapter_path=checkpoint_path,
        base_model_path=str(tmp_path / "base"),
        output_path=output_path,
    )
    merge_embed.assert_not_called()
    merge_llm.assert_not_called()
    export_onnx.assert_not_called()
