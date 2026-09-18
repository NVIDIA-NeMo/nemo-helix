# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""process_checkpoint merge, ONNX dispatch, and fileset layout."""

import json
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
    find_selected_checkpoints,
    process_checkpoint,
    process_selected_checkpoints,
    sanitize_encoder_hf,
)
from nmp.automodel.tasks.training.schemas import (
    CheckpointFormat,
    CheckpointInfo,
    CheckpointSelection,
    ExportConfig,
    RetrievalConfig,
)
from pytest_mock import MockerFixture

CHECKPOINTS = "nmp.automodel.tasks.training.backends.checkpoints"


@pytest.fixture
def merged_lora_config(tmp_path: Path) -> MagicMock:
    customizer_config = MagicMock()
    customizer_config.training.finetuning_type = FinetuningType.LORA_MERGED
    customizer_config.model.path = str(tmp_path / "base")
    customizer_config.model.precision = None
    customizer_config.model.trust_remote_code = False
    return customizer_config


@pytest.fixture
def patched_checkpoint_io(mocker: MockerFixture) -> dict:
    mocker.patch(f"{CHECKPOINTS}.fix_fsdp2_architecture")
    mocker.patch(f"{CHECKPOINTS}.extract_precision_from_model_config", return_value=None)
    order: list[str] = []

    def _record(name: str):
        def _side_effect(*_args, **_kwargs):
            order.append(name)

        return _side_effect

    return {
        "order": order,
        "merge_cross": mocker.patch(f"{CHECKPOINTS}.merge_lora_cross_encoder_adapter"),
        "merge_embed": mocker.patch(f"{CHECKPOINTS}.merge_lora_embedding_adapter"),
        "export_onnx": mocker.patch(f"{CHECKPOINTS}.export_onnx", side_effect=_record("export")),
        "sanitize": mocker.patch(f"{CHECKPOINTS}.sanitize_encoder_hf", side_effect=_record("sanitize")),
        "restructure": mocker.patch(f"{CHECKPOINTS}._restructure_encoder_output", side_effect=_record("restructure")),
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
            trust_remote_code=False,
        )
        patched_checkpoint_io["merge_embed"].assert_not_called()

        # The reranker must be traced as a cross-encoder, not pooled embeddings.
        call = patched_checkpoint_io["export_onnx"].call_args.kwargs
        assert call["model_type"] == ModelType.CROSS_ENCODER
        assert call["model_path"] == output_path
        assert call["output_path"] == output_path / "alternates" / "onnx"
        assert call["tokenizer_path"] == str(tmp_path / "base")
        assert call["cfg"] is export
        assert call["trust_remote_code"] is False
        assert call["load_trust_remote_code"] is False
        patched_checkpoint_io["sanitize"].assert_called_once_with(output_path, tmp_path / "base")
        assert patched_checkpoint_io["order"] == ["export", "sanitize", "restructure"]


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


class TestCheckpointSelection:
    @staticmethod
    def _checkpoint(base: Path, name: str) -> Path:
        model = base / name / "model" / "consolidated"
        model.mkdir(parents=True)
        return model

    def test_both_resolves_lowest_val_and_latest(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        best = self._checkpoint(checkpoints, "epoch_1_step_10")
        last = self._checkpoint(checkpoints, "epoch_2_step_20")
        (checkpoints / "LOWEST_VAL").symlink_to(best.parent.parent)
        (checkpoints / "LATEST").symlink_to(last.parent.parent)
        config = MagicMock()
        config.training.finetuning_type = FinetuningType.ALL_WEIGHTS
        config.schedule.checkpoint_selection = CheckpointSelection.BOTH

        selected = find_selected_checkpoints(tmp_path, config)

        assert selected == {"best": best.resolve(), "last": last.resolve()}

    def test_last_does_not_prefer_lowest_val(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        best = self._checkpoint(checkpoints, "epoch_1_step_10")
        last = self._checkpoint(checkpoints, "epoch_2_step_20")
        (checkpoints / "LOWEST_VAL").symlink_to(best.parent.parent)
        (checkpoints / "LATEST").symlink_to(last.parent.parent)
        config = MagicMock()
        config.training.finetuning_type = FinetuningType.ALL_WEIGHTS
        config.schedule.checkpoint_selection = CheckpointSelection.LAST

        assert find_selected_checkpoints(tmp_path, config) == {"last": last.resolve()}


@pytest.mark.parametrize("primary", ["hf", "onnx"])
def test_both_embedding_checkpoints_publish_all_formats_and_stats(
    primary: Literal["hf", "onnx"], tmp_path: Path, mocker: MockerFixture
) -> None:
    output = tmp_path / "output"
    workspace = tmp_path / "workspace"
    stats = workspace / "checkpoints" / "checkpoint_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text('{"checkpoints": [{"step": 1, "val_loss": {"default": 0.5}}]}\n')
    config = MagicMock()
    config.retrieval = RetrievalConfig(export=ExportConfig(primary=primary))

    def fake_process(checkpoint_path: Path, destination: Path, *_args, **_kwargs) -> CheckpointInfo:
        label = checkpoint_path.name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{label}-{primary}").write_text(primary)
        alternate = "onnx" if primary == "hf" else "hf"
        alternate_path = destination / "alternates" / alternate
        alternate_path.mkdir(parents=True)
        (alternate_path / f"{label}-{alternate}").write_text(alternate)
        return CheckpointInfo(path=str(destination), format=CheckpointFormat.HF)

    mocker.patch(f"{CHECKPOINTS}.process_checkpoint", side_effect=fake_process)

    process_selected_checkpoints(
        {"best": tmp_path / "best", "last": tmp_path / "last"},
        output,
        workspace,
        config,
        model_type=ModelType.EMBEDDING,
    )

    alternate = "onnx" if primary == "hf" else "hf"
    # The best checkpoint's own alternate format and the last checkpoint are siblings.
    assert (output / f"best-{primary}").is_file()
    assert (output / "alternates" / alternate / f"best-{alternate}").is_file()
    assert (output / "alternates" / "last" / f"last-{primary}").is_file()
    assert (output / "alternates" / "last" / "alternates" / alternate / f"last-{alternate}").is_file()
    assert (output / "checkpoint_stats.json").read_text() == stats.read_text()
    assert not (output / "best").exists(), "the staging directories must not survive in the fileset"
    assert not (output / "last").exists()


@pytest.mark.parametrize("label", ["best", "last"])
def test_a_single_selected_checkpoint_lands_at_the_fileset_root(
    label: Literal["best", "last"], tmp_path: Path, mocker: MockerFixture
) -> None:
    """Staging is an implementation detail: one checkpoint still publishes unnested."""
    output = tmp_path / "output"
    config = MagicMock()
    config.retrieval = RetrievalConfig(export=ExportConfig(primary="onnx"))

    def fake_process(_checkpoint_path: Path, destination: Path, *_args, **_kwargs) -> CheckpointInfo:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.onnx").write_text("onnx")
        (destination / "alternates" / "hf").mkdir(parents=True)
        return CheckpointInfo(path=str(destination), format=CheckpointFormat.HF)

    mocker.patch(f"{CHECKPOINTS}.process_checkpoint", side_effect=fake_process)

    info = process_selected_checkpoints(
        {label: tmp_path / label},
        output,
        tmp_path / "workspace",
        config,
        model_type=ModelType.EMBEDDING,
    )

    assert info.path == str(output), "the reported path must be the fileset root, not the staging directory"
    assert (output / "model.onnx").is_file()
    assert (output / "alternates" / "hf").is_dir()
    assert not (output / label).exists()
    assert not (output / "alternates" / "last").exists()


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


class TestSanitizeEncoderHf:
    def _write_configs(self, tmp_path: Path) -> tuple[Path, Path]:
        base = tmp_path / "base"
        output = tmp_path / "hf"
        base.mkdir()
        output.mkdir()
        (base / "config.json").write_text(
            '{"model_type": "ministral3", "architectures": ["Ministral3Model"], "is_causal": true}'
        )
        (output / "config.json").write_text(
            '{"model_type": "ministral3_bidirec", "architectures": ["Ministral3BidirectionalModel"],'
            ' "auto_map": {"AutoModel": "model.Ministral3BidirectionalModel"},'
            ' "is_causal": false, "pooling": "avg", "temperature": 0.02}'
        )
        (output / "model.py").write_text("raise RuntimeError('custom')")
        (base / "shared.py").write_text("ok")
        (output / "shared.py").write_text("ok")
        return output, base

    def test_restores_base_architecture_and_drops_auto_map(self, tmp_path: Path) -> None:
        output, base = self._write_configs(tmp_path)

        sanitize_encoder_hf(output, base)

        config = json.loads((output / "config.json").read_text())
        assert config["model_type"] == "ministral3"
        assert config["architectures"] == ["Ministral3Model"]
        assert "auto_map" not in config
        assert config["is_causal"] is False
        assert config["pooling"] == "avg"
        assert config["temperature"] == 0.02
        assert not (output / "model.py").exists()
        assert (output / "shared.py").is_file()

    def test_skips_when_config_missing(self, tmp_path: Path) -> None:
        output = tmp_path / "hf"
        output.mkdir()
        (output / "model.py").write_text("x")
        sanitize_encoder_hf(output, tmp_path / "missing-base")
        assert (output / "model.py").is_file()


class TestProcessCheckpointSanitize:
    def _full_sft_config(self, tmp_path: Path, *, trust_remote_code: bool) -> MagicMock:
        customizer_config = MagicMock()
        customizer_config.training.finetuning_type = FinetuningType.ALL_WEIGHTS
        customizer_config.model.path = str(tmp_path / "base")
        customizer_config.model.precision = None
        customizer_config.model.trust_remote_code = trust_remote_code
        customizer_config.retrieval = RetrievalConfig(export=ExportConfig(primary="onnx"))
        return customizer_config

    def test_full_sft_untrusted_exports_then_sanitizes(self, patched_checkpoint_io: dict, tmp_path: Path) -> None:
        config = self._full_sft_config(tmp_path, trust_remote_code=False)
        checkpoint_path = tmp_path / "ckpt"
        output_path = tmp_path / "output"

        process_checkpoint(checkpoint_path, output_path, config, model_type=ModelType.EMBEDDING)

        patched_checkpoint_io["copytree"].assert_called_once()
        export_call = patched_checkpoint_io["export_onnx"]
        sanitize_call = patched_checkpoint_io["sanitize"]
        assert export_call.call_args.kwargs["load_trust_remote_code"] is True
        assert export_call.call_args.kwargs["trust_remote_code"] is False
        assert sanitize_call.call_args.args == (output_path, tmp_path / "base")
        assert patched_checkpoint_io["order"] == ["export", "sanitize", "restructure"]

    def test_full_sft_trusted_skips_sanitize(self, patched_checkpoint_io: dict, tmp_path: Path) -> None:
        config = self._full_sft_config(tmp_path, trust_remote_code=True)
        process_checkpoint(tmp_path / "ckpt", tmp_path / "output", config, model_type=ModelType.EMBEDDING)
        patched_checkpoint_io["sanitize"].assert_not_called()
        assert patched_checkpoint_io["export_onnx"].call_args.kwargs["load_trust_remote_code"] is True
        assert patched_checkpoint_io["export_onnx"].call_args.kwargs["trust_remote_code"] is True

    def test_unmerged_lora_skips_sanitize_and_onnx(self, patched_checkpoint_io: dict, tmp_path: Path) -> None:
        config = MagicMock()
        config.training.finetuning_type = FinetuningType.LORA
        config.model.path = str(tmp_path / "base")
        config.model.precision = None
        config.model.trust_remote_code = False
        process_checkpoint(tmp_path / "adapter", tmp_path / "output", config, model_type=ModelType.EMBEDDING)
        patched_checkpoint_io["sanitize"].assert_not_called()
        patched_checkpoint_io["export_onnx"].assert_not_called()

    def test_merged_lora_untrusted_sanitizes_and_loads_native(
        self, merged_lora_config: MagicMock, patched_checkpoint_io: dict, tmp_path: Path
    ) -> None:
        merged_lora_config.retrieval = RetrievalConfig(export=ExportConfig(primary="hf"))
        process_checkpoint(
            tmp_path / "adapter",
            tmp_path / "output",
            merged_lora_config,
            model_type=ModelType.EMBEDDING,
        )
        patched_checkpoint_io["merge_embed"].assert_called_once_with(
            adapter_path=tmp_path / "adapter",
            base_model_path=str(tmp_path / "base"),
            output_path=tmp_path / "output",
            trust_remote_code=False,
        )
        call = patched_checkpoint_io["export_onnx"].call_args.kwargs
        assert call["load_trust_remote_code"] is False
        assert call["trust_remote_code"] is False
        patched_checkpoint_io["sanitize"].assert_called_once()
        assert patched_checkpoint_io["order"] == ["export", "sanitize", "restructure"]
