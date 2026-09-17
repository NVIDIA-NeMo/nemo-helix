# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""process_checkpoint merge, ONNX dispatch, and fileset layout."""

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from nmp.automodel.entities.values import FinetuningType
from nmp.automodel.tasks.training.backends.checkpoints import (
    ModelType,
    _build_export_module,
    _probe_bidirectional_mask,
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
        assert call["output_path"] == output_path / "alternates" / "onnx"
        assert call["tokenizer_path"] == str(tmp_path / "base")
        assert call["cfg"] is export
        patched_checkpoint_io["restructure"].assert_called_once_with(output_path, "hf")


class TestRestructureEncoderOutput:
    HF_NAMES = frozenset(
        {
            "model.safetensors",
            "config.json",
            "tokenizer_config.json",
            "modeling_nemotron.py",
            "configuration_nemotron.py",
            "custom_code",
        }
    )
    ONNX_NAMES = frozenset(
        {
            "model.onnx",
            "model.onnx.data",
            "onnx__MatMul_4456",
            "model.embed_tokens.weight",
            "model.layers.1.input_layernorm.weight",
            "encoder.layer.0.attention.self.query.bias",
            "tokenizer",
        }
    )

    @classmethod
    def _checkpoint(cls, tmp_path: Path) -> Path:
        """The layout ``process_checkpoint`` produces: HF at the root, ONNX in its own dir."""
        output = tmp_path / "output"
        output.mkdir()
        (output / "custom_code").mkdir()
        (output / "custom_code" / "modeling.py").write_text("code")
        for name in cls.HF_NAMES - {"custom_code"}:
            (output / name).write_text(name)

        onnx_dir = output / "alternates" / "onnx"
        (onnx_dir / "tokenizer").mkdir(parents=True)
        (onnx_dir / "tokenizer" / "tokenizer.json").write_text("tok")
        for name in cls.ONNX_NAMES - {"tokenizer"}:
            (onnx_dir / name).write_text(name)
        return output

    def test_onnx_primary_promotes_onnx_and_nests_hf(self, tmp_path: Path) -> None:
        output = self._checkpoint(tmp_path)

        _restructure_encoder_output(output, "onnx")

        assert {e.name for e in output.iterdir()} == self.ONNX_NAMES | {"alternates"}
        assert {e.name for e in (output / "alternates" / "hf").iterdir()} == self.HF_NAMES
        assert not (output / "alternates" / "onnx").exists()
        assert (output / "alternates" / "hf" / "custom_code" / "modeling.py").is_file()

    def test_hf_primary_leaves_hf_at_root_and_onnx_nested(self, tmp_path: Path) -> None:
        output = self._checkpoint(tmp_path)

        _restructure_encoder_output(output, "hf")

        assert {e.name for e in output.iterdir()} == self.HF_NAMES | {"alternates"}
        assert {e.name for e in (output / "alternates" / "onnx").iterdir()} == self.ONNX_NAMES


def test_probe_bidirectional_mask_survives_lazy_loader_modules() -> None:
    """A PEP 562 loader that raises ModuleNotFoundError must read as "no mask"."""
    import types

    mod = types.ModuleType("fake_lazy_pkg")

    def _raising_getattr(name: str):
        raise ModuleNotFoundError(f"no submodule {name!r}")

    setattr(mod, "__getattr__", _raising_getattr)

    assert _probe_bidirectional_mask(mod) is None


def test_probe_bidirectional_mask_returns_the_function() -> None:
    import types

    mod = types.ModuleType("fake_pkg")

    def _mask() -> str:
        return "original"

    setattr(mod, "create_bidirectional_mask", _mask)

    assert _probe_bidirectional_mask(mod) is _mask


def test_probe_bidirectional_mask_missing_attribute_is_none() -> None:
    import types

    assert _probe_bidirectional_mask(types.ModuleType("bare_pkg")) is None


@pytest.mark.parametrize(
    ("pooling", "padding_side", "expected"),
    [
        ("avg", "right", [1.0, 10.5]),
        ("cls", "right", [0.0, 10.0]),
        ("last", "right", [2.0, 11.0]),
        ("avg", "left", [2.0, 12.5]),
        ("cls", "left", [1.0, 12.0]),
        ("last", "left", [3.0, 13.0]),
    ],
)
def test_embedding_pooling_supports_both_padding_sides(
    pooling: Literal["avg", "cls", "last"], padding_side: str, expected: list[float]
) -> None:
    torch = pytest.importorskip("torch")
    mask = {
        "right": [[1, 1, 1, 0], [1, 1, 0, 0]],
        "left": [[0, 1, 1, 1], [0, 0, 1, 1]],
    }[padding_side]
    hidden = torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]])
    export_model = _build_export_module(
        torch.nn.Identity(),
        ModelType.EMBEDDING,
        ExportConfig(pooling=pooling, normalize=False),
    )

    actual = export_model._pool(hidden, torch.tensor(mask))

    torch.testing.assert_close(actual[:, 0], torch.tensor(expected))
