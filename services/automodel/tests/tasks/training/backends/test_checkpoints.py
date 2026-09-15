# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""process_checkpoint merge, ONNX dispatch, and fileset layout."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nmp.automodel.entities.values import FinetuningType
from nmp.automodel.tasks.training.backends.checkpoints import (
    ModelType,
    _restructure_encoder_output,
    process_checkpoint,
)
from nmp.automodel.tasks.training.schemas import ExportConfig, RetrievalConfig
from pytest_mock import MockerFixture

CHECKPOINTS = "nmp.automodel.tasks.training.backends.checkpoints"


@pytest.fixture
def merged_lora_config(tmp_path: Path) -> MagicMock:
    customizer_config = MagicMock()
    customizer_config.training.finetuning_type = FinetuningType.LORA_MERGED
    customizer_config.model.path = str(tmp_path / "base")
    customizer_config.model.precision = None
    return customizer_config


@pytest.fixture
def patched_checkpoint_io(mocker: MockerFixture) -> dict:
    mocker.patch(f"{CHECKPOINTS}.fix_fsdp2_architecture")
    mocker.patch(f"{CHECKPOINTS}.extract_precision_from_model_config", return_value=None)
    return {
        "merge_cross": mocker.patch(f"{CHECKPOINTS}.merge_lora_cross_encoder_adapter"),
        "merge_embed": mocker.patch(f"{CHECKPOINTS}.merge_lora_embedding_adapter"),
        "export_onnx": mocker.patch(f"{CHECKPOINTS}.export_onnx"),
        "restructure": mocker.patch(f"{CHECKPOINTS}._restructure_encoder_output"),
        "copytree": mocker.patch(f"{CHECKPOINTS}.shutil.copytree"),
    }


class TestCrossEncoderCheckpoint:
    def test_merges_cross_encoder_lora_and_exports_onnx(
        self, merged_lora_config: MagicMock, patched_checkpoint_io: dict, tmp_path: Path
    ) -> None:
        checkpoint_path = tmp_path / "adapter"
        output_path = tmp_path / "output"
        export = ExportConfig(primary="hf", opset=18)
        merged_lora_config.retrieval = RetrievalConfig(export=export)

        process_checkpoint(
            checkpoint_path,
            output_path,
            merged_lora_config,
            model_type=ModelType.CROSS_ENCODER,
        )

        patched_checkpoint_io["merge_cross"].assert_called_once_with(
            adapter_path=checkpoint_path,
            base_model_path=str(tmp_path / "base"),
            output_path=output_path,
        )
        patched_checkpoint_io["merge_embed"].assert_not_called()

        # The reranker must be traced as a cross-encoder, not pooled embeddings.
        call = patched_checkpoint_io["export_onnx"].call_args.kwargs
        assert call["model_type"] == ModelType.CROSS_ENCODER
        assert call["model_path"] == output_path
        assert call["output_path"] == output_path
        assert call["tokenizer_path"] == str(tmp_path / "base")
        assert call["cfg"] is export
        patched_checkpoint_io["restructure"].assert_called_once_with(output_path, "hf")


class TestRestructureEncoderOutput:
    @staticmethod
    def _checkpoint(tmp_path: Path) -> Path:
        output = tmp_path / "output"
        (output / "tokenizer").mkdir(parents=True)
        for name in ("model.onnx", "model.safetensors", "config.json"):
            (output / name).write_text(name)
        return output

    def test_onnx_primary_moves_hf_weights_to_alternates(self, tmp_path: Path) -> None:
        output = self._checkpoint(tmp_path)

        _restructure_encoder_output(output, "onnx")

        assert {e.name for e in output.iterdir()} == {"model.onnx", "tokenizer", "alternates"}
        assert {e.name for e in (output / "alternates" / "hf").iterdir()} == {
            "model.safetensors",
            "config.json",
        }

    def test_hf_primary_moves_onnx_to_alternates(self, tmp_path: Path) -> None:
        output = self._checkpoint(tmp_path)

        _restructure_encoder_output(output, "hf")

        assert {e.name for e in output.iterdir()} == {"model.safetensors", "config.json", "tokenizer", "alternates"}
        assert {e.name for e in (output / "alternates" / "onnx").iterdir()} == {"model.onnx"}
