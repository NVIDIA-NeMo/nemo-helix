# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AutomodelBackend embedding model type selection."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

# Mock nemo_automodel before importing backend/config modules
# (nemo_automodel is only available in the training container)
sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()

from nmp.automodel.tasks.training.backends.backend import AutomodelBackend  # noqa: E402
from nmp.automodel.tasks.training.backends.checkpoints import ModelType  # noqa: E402
from nmp.automodel.tasks.training.schemas import TrainingRecipe  # noqa: E402


class TestAutomodelBackend:
    """Tests for AutomodelBackend."""

    def test_find_checkpoints_uses_model_embedding_flag(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """ModelType should be EMBEDDING when model.is_embedding_model is True."""
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.model.checkpoint_head_type = "unknown"
        customizer_config.model.is_embedding_model = True
        customizer_config.model.name = "meta/llama-3.1-8b-instruct"

        expected = {"best": tmp_path / "best.ckpt"}
        mock_find_selected = mocker.patch(
            "nmp.automodel.tasks.training.backends.backend.find_selected_checkpoints",
            return_value=expected,
        )

        result = backend.find_checkpoints(tmp_path, customizer_config)

        assert result == expected
        mock_find_selected.assert_called_once_with(
            tmp_path,
            customizer_config,
            model_type=ModelType.EMBEDDING,
        )

    def test_process_checkpoints_uses_model_embedding_flag_not_model_name(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """ModelType should stay LLM when model.is_embedding_model is False."""
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.model.checkpoint_head_type = "unknown"
        customizer_config.model.is_embedding_model = False
        customizer_config.model.name = "nvidia/llama-nemotron-embed-1b-v2"

        checkpoint_info = MagicMock()
        mock_process = mocker.patch(
            "nmp.automodel.tasks.training.backends.backend.process_selected_checkpoints",
            return_value=checkpoint_info,
        )

        checkpoints = {"best": tmp_path / "checkpoint"}
        output_path = tmp_path / "output_model"
        result = backend.process_checkpoints(
            checkpoints=checkpoints,
            output_path=output_path,
            workspace_dir=tmp_path,
            customizer_config=customizer_config,
            library_config=None,
        )

        assert result == checkpoint_info
        mock_process.assert_called_once_with(
            checkpoints,
            output_path,
            tmp_path,
            customizer_config,
            model_type=ModelType.LLM,
            resolved_chat_template=None,
        )

    def test_cross_encoder_recipe_uses_cross_encoder_checkpoint_processing(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.training.recipe = TrainingRecipe.CROSS_ENCODER

        expected = {"best": tmp_path / "best.ckpt"}
        mock_find_selected = mocker.patch(
            "nmp.automodel.tasks.training.backends.backend.find_selected_checkpoints",
            return_value=expected,
        )

        assert backend.find_checkpoints(tmp_path, customizer_config) == expected
        mock_find_selected.assert_called_once_with(
            tmp_path,
            customizer_config,
            model_type=ModelType.CROSS_ENCODER,
        )

    def test_auto_recipe_cross_encoder_head_uses_cross_encoder_checkpoint_processing(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        backend = AutomodelBackend(job_ctx=MagicMock())
        customizer_config = MagicMock()
        customizer_config.training.recipe = TrainingRecipe.AUTO
        customizer_config.model.is_embedding_model = False
        customizer_config.model.checkpoint_head_type = "cross_encoder"

        mock_process = mocker.patch(
            "nmp.automodel.tasks.training.backends.backend.process_selected_checkpoints",
            return_value=MagicMock(),
        )

        backend.process_checkpoints(
            checkpoints={"best": tmp_path / "checkpoint"},
            output_path=tmp_path / "output_model",
            workspace_dir=tmp_path,
            customizer_config=customizer_config,
            library_config=None,
        )

        assert mock_process.call_args.kwargs["model_type"] == ModelType.CROSS_ENCODER
