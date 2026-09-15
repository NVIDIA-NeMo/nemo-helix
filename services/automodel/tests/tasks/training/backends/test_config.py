# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automodel config compilation functions."""

import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

# Mock nemo_automodel before importing the config module
# (nemo_automodel is only available in the training container)
sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()


@pytest.fixture(autouse=True)
def _transformers_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply `transformers` for the AutoConfig patches without leaking one.

    config.py imports transformers inside the functions that need it, so it is
    only wanted while a test runs -- and only when the real package is absent,
    which is the training image's arrangement, not necessarily the test env's.

    Installed at module scope with setdefault it won permanently whenever this
    file happened to import before anything had loaded the real package, and the
    MagicMock then leaked across the whole xdist worker. services/unsloth does
    `from transformers import TrainerCallback` at call time, so its
    HfTrainerProgressCallback became a mock subclass whose hooks did nothing --
    two of its tests passed vacuously or failed depending on collection order.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "transformers", MagicMock())


from nmp.automodel.tasks.training.backends.config import (  # noqa: E402
    _configure_chat_dataset,
    _configure_moe_backend,
    _configure_retrieval_dataset,
    _configure_sft_dataset,
    _resolve_optimizer_target,
    compile_automodel_config,
    estimate_steps_per_epoch,
    resolve_compiled_recipe,
    resolve_warmup_steps,
)
from nmp.automodel.tasks.training.datasets.preparation import PreparedDataset  # noqa: E402
from nmp.automodel.tasks.training.schemas import RetrievalConfig, TrainingRecipe, TrainingStepConfig  # noqa: E402

CONFIG_MODULE = "nmp.automodel.tasks.training.backends.config"
AUTOCONFIG_PATCH = "transformers.AutoConfig"
MODEL_REGISTRY_PATCH = f"{CONFIG_MODULE}.ModelRegistry"


@pytest.fixture
def mock_customizer_config() -> MagicMock:
    """Create a mock TrainingStepConfig for testing."""
    config = MagicMock()
    config.model.path = "/models/test-model"
    config.model.name = "test/model"
    return config


@pytest.fixture
def temp_dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary dataset files for testing."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')
    val_file.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')
    return train_file, val_file


class TestConfigureChatDataset:
    """Tests for _configure_chat_dataset function."""

    def test_chat_dataset_includes_split_attribute(
        self,
        mock_customizer_config: MagicMock,
        temp_dataset_files: tuple[Path, Path],
        mocker,
    ) -> None:
        train_file, val_file = temp_dataset_files
        cfg: dict[str, Any] = {}

        mocker.patch(f"{CONFIG_MODULE}.resolve_chat_template", return_value="mock_template")
        mock_customizer_config.parallelism.pipeline_parallel_size = 1
        _configure_chat_dataset(cfg, mock_customizer_config, train_file, val_file, seq_length=2048)

        assert "dataset" in cfg
        assert "validation_dataset" in cfg
        assert cfg["dataset"]["split"] == "train"
        assert cfg["validation_dataset"]["split"] == "validation"

    def test_chat_dataset_includes_required_fields(
        self,
        mock_customizer_config: MagicMock,
        temp_dataset_files: tuple[Path, Path],
        mocker,
    ) -> None:
        train_file, val_file = temp_dataset_files
        cfg: dict[str, Any] = {}

        mocker.patch(f"{CONFIG_MODULE}.resolve_chat_template", return_value="mock_template")
        mock_customizer_config.parallelism.pipeline_parallel_size = 1
        _configure_chat_dataset(cfg, mock_customizer_config, train_file, val_file, seq_length=2048)

        assert cfg["dataset"]["_target_"] == "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset"
        assert cfg["dataset"]["path_or_dataset_id"] == str(train_file)
        assert cfg["dataset"]["seq_length"] == 2048
        assert cfg["dataset"]["chat_template"] == "mock_template"


class TestConfigureSftDataset:
    """Tests for _configure_sft_dataset function."""

    def test_sft_dataset_includes_split_attribute(
        self,
        temp_dataset_files: tuple[Path, Path],
        mock_customizer_config: MagicMock,
    ) -> None:
        train_file, val_file = temp_dataset_files
        cfg: dict[str, Any] = {}

        mock_customizer_config.parallelism.pipeline_parallel_size = 1
        _configure_sft_dataset(
            cfg,
            mock_customizer_config,
            train_file,
            val_file,
            question_col="prompt",
            answer_col="completion",
            seq_length=2048,
        )

        assert "dataset" in cfg
        assert "validation_dataset" in cfg
        assert cfg["dataset"]["split"] == "train"
        assert cfg["validation_dataset"]["split"] == "validation"

    def test_sft_dataset_includes_required_fields(
        self,
        temp_dataset_files: tuple[Path, Path],
        mock_customizer_config: MagicMock,
    ) -> None:
        train_file, val_file = temp_dataset_files
        cfg: dict[str, Any] = {}

        mock_customizer_config.parallelism.pipeline_parallel_size = 1
        _configure_sft_dataset(
            cfg,
            mock_customizer_config,
            train_file,
            val_file,
            question_col="prompt",
            answer_col="completion",
            seq_length=2048,
        )

        assert (
            cfg["dataset"]["_target_"]
            == "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset"
        )
        assert cfg["dataset"]["path_or_dataset_id"] == str(train_file)
        assert cfg["dataset"]["seq_length"] == 2048
        assert cfg["dataset"]["column_mapping"]["question"] == "prompt"
        assert cfg["dataset"]["column_mapping"]["answer"] == "completion"
        assert cfg["dataset"]["answer_only_loss_mask"] is True
        assert cfg["dataset"]["padding"] == "do_not_pad"
        assert cfg["dataset"]["truncation"] == "longest_first"


class TestConfigureRetrievalDataset:
    def test_bi_encoder_uses_bi_encoder_collator(
        self,
        tmp_path: Path,
        mock_customizer_config: MagicMock,
    ) -> None:
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "validation.jsonl"
        train_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        val_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        cfg: dict[str, Any] = {}

        _configure_retrieval_dataset(
            cfg,
            mock_customizer_config,
            train_file,
            val_file,
            seed=42,
            retrieval_config=RetrievalConfig(),
            recipe=TrainingRecipe.BI_ENCODER,
        )

        assert cfg["dataloader"]["dataset"]["model_type"] == "bi_encoder"
        assert cfg["dataloader"]["collate_fn"]["_target_"].endswith("BiEncoderCollator")

    def test_cross_encoder_uses_cross_encoder_collator(
        self,
        tmp_path: Path,
        mock_customizer_config: MagicMock,
    ) -> None:
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "validation.jsonl"
        train_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        val_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        cfg: dict[str, Any] = {}

        _configure_retrieval_dataset(
            cfg,
            mock_customizer_config,
            train_file,
            val_file,
            seed=42,
            retrieval_config=RetrievalConfig(),
            recipe=TrainingRecipe.CROSS_ENCODER,
        )

        assert cfg["dataloader"]["dataset"]["model_type"] == "cross_encoder"
        assert cfg["dataloader"]["collate_fn"]["_target_"].endswith("CrossEncoderCollator")
        assert cfg["dataloader"]["collate_fn"]["rerank_max_length"] == 512

    def test_bi_encoder_strips_trailing_space_from_nemotron_prefixes(
        self,
        tmp_path: Path,
        mock_customizer_config: MagicMock,
    ) -> None:
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "validation.jsonl"
        train_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        val_file.write_text('{"query":"q","pos_doc":"p","neg_doc":["n"]}\n')
        cfg: dict[str, Any] = {}

        _configure_retrieval_dataset(
            cfg,
            mock_customizer_config,
            train_file,
            val_file,
            seed=42,
            retrieval_config=RetrievalConfig(
                train_n_passages=7,
                query_max_length=256,
                passage_max_length=384,
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
            recipe=TrainingRecipe.BI_ENCODER,
        )

        collator = cfg["dataloader"]["collate_fn"]
        assert collator["query_prefix"] == "query:"
        assert collator["passage_prefix"] == "passage:"
        assert collator["q_max_len"] == 256
        assert collator["p_max_len"] == 384
        assert cfg["dataloader"]["dataset"]["n_passages"] == 7


class TestConfigureMoeBackend:
    """Tests for _configure_moe_backend function."""

    def _make_config(
        self,
        model_path: str = "/models/test-model",
        num_nodes: int = 1,
        num_gpus_per_node: int = 1,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int | None = None,
    ) -> MagicMock:
        config = MagicMock()
        config.model.path = model_path
        config.parallelism.num_nodes = num_nodes
        config.parallelism.num_gpus_per_node = num_gpus_per_node
        config.parallelism.tensor_parallel_size = tensor_parallel_size
        config.parallelism.expert_parallel_size = expert_parallel_size
        return config

    def _make_hf_config(
        self,
        architectures: list[str],
        num_local_experts: int | None = None,
        num_experts: int | None = None,
    ) -> MagicMock:
        hf_config = MagicMock()
        hf_config.architectures = architectures

        original_getattr = type(hf_config).__getattr__

        def _controlled_getattr(self: Any, name: str) -> Any:
            if name == "num_local_experts":
                return num_local_experts
            if name == "num_experts":
                return num_experts
            return original_getattr(self, name)

        type(hf_config).__getattr__ = cast(Any, _controlled_getattr)
        return hf_config

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_model_gets_backend_and_parallelizer(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        _configure_moe_backend(cfg, self._make_config(num_gpus_per_node=8, expert_parallel_size=8))

        assert cfg["model"]["backend"] == {
            "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
            "enable_deepep": False,
        }

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_multi_gpu_tp_gt1_raises(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        with pytest.raises(ValueError, match=r"Tensor parallelism.*not supported.*MoE"):
            _configure_moe_backend(
                cfg,
                self._make_config(num_gpus_per_node=8, tensor_parallel_size=2, expert_parallel_size=4),
            )

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_multi_gpu_ep_not_set_raises(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        with pytest.raises(ValueError, match=r"expert_parallel_size.*not set.*requires expert_parallel_size > 1"):
            _configure_moe_backend(
                cfg,
                self._make_config(num_gpus_per_node=8, expert_parallel_size=None),
            )

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_multi_gpu_ep_eq1_raises(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        with pytest.raises(ValueError, match=r"expert_parallel_size is 1.*requires expert_parallel_size > 1"):
            _configure_moe_backend(
                cfg,
                self._make_config(num_gpus_per_node=8, expert_parallel_size=1),
            )

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_single_gpu_skips_multi_gpu_validation(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        _configure_moe_backend(cfg, self._make_config(num_gpus_per_node=1, expert_parallel_size=None))

        assert cfg["model"]["backend"]["_target_"] == "nemo_automodel.components.models.common.utils.BackendConfig"

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_dense_custom_model_no_moe_config(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["LlamaForCausalLM"],
        )
        mock_registry.model_arch_name_to_cls = {"LlamaForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {"model": {}}
        _configure_moe_backend(cfg, self._make_config())

        assert "backend" not in cfg["model"]
        assert "parallelizer" not in cfg

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_standard_hf_model_no_custom_config(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["LlamaForCausalLM"],
        )
        mock_registry.model_arch_name_to_cls = {}

        cfg: dict[str, Any] = {"model": {}}
        _configure_moe_backend(cfg, self._make_config())

        assert "backend" not in cfg["model"]
        assert "parallelizer" not in cfg

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_autoconfig_exception_handled_gracefully(self, mock_autoconfig_cls, _mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.side_effect = OSError("Model not found")

        cfg: dict[str, Any] = {"model": {}}
        _configure_moe_backend(cfg, self._make_config())

        assert "backend" not in cfg["model"]
        assert "parallelizer" not in cfg

    @patch(MODEL_REGISTRY_PATCH)
    @patch(AUTOCONFIG_PATCH)
    def test_moe_validation_error_propagates(self, mock_autoconfig_cls, mock_registry) -> None:
        mock_autoconfig_cls.from_pretrained.return_value = self._make_hf_config(
            architectures=["NemotronHForCausalLM"],
            num_local_experts=8,
        )
        mock_registry.model_arch_name_to_cls = {"NemotronHForCausalLM": MagicMock()}

        cfg: dict[str, Any] = {}
        with pytest.raises(ValueError, match="Tensor parallelism"):
            _configure_moe_backend(
                cfg,
                self._make_config(num_gpus_per_node=8, tensor_parallel_size=4, expert_parallel_size=2),
            )


class TestResolveWarmupSteps:
    """Tests for resolve_warmup_steps function."""

    @staticmethod
    def _resolve(**overrides: int) -> int:
        kwargs: dict[str, int] = {
            "warmup_steps": 50,
            "max_steps": 10,
        }
        kwargs.update(overrides)
        return resolve_warmup_steps(**kwargs)

    def test_warmup_below_decay_steps_is_unchanged(self) -> None:
        assert self._resolve(warmup_steps=5, max_steps=39) == 5

    def test_warmup_clamped_to_one_below_decay_steps(self) -> None:
        # max_steps=10 is the shorter schedule, so warmup must land at 9.
        assert self._resolve(warmup_steps=50) == 9

    def test_warmup_equal_to_decay_steps_is_clamped(self) -> None:
        assert self._resolve(warmup_steps=10) == 9

    def test_single_step_schedule_disables_warmup(self) -> None:
        assert self._resolve(warmup_steps=50, max_steps=1) == 0

    def test_zero_warmup_is_left_alone(self) -> None:
        assert self._resolve(warmup_steps=0) == 0

    def test_clamping_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            self._resolve(warmup_steps=50)
        assert "clamping warmup_steps to 9" in caplog.text


class TestEstimateStepsPerEpoch:
    """Tests for sequence-packing-aware step estimation."""

    def test_without_packing_uses_raw_sample_count(self) -> None:
        assert estimate_steps_per_epoch(train_samples=9741, batch_size=1024) == 10

    def test_packing_factor_reduces_steps(self) -> None:
        assert estimate_steps_per_epoch(train_samples=9741, batch_size=1024, packing_factor=1.6) == 6

    @pytest.mark.parametrize("packing_factor", [None, 0.0, 1.0])
    def test_non_effective_packing_factor_is_ignored(self, packing_factor: float | None) -> None:
        assert (
            estimate_steps_per_epoch(
                train_samples=9741,
                batch_size=1024,
                packing_factor=packing_factor,
            )
            == 10
        )


def test_compile_uses_fallback_packing_factor_for_schedule(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).parents[3] / "contract" / "input_configs" / "llama-3.2-1b" / "llama_3_2_1b_lora_packing.json"
    )
    raw = json.loads(fixture.read_text())
    raw.pop("backend")
    config = TrainingStepConfig.model_validate(raw)
    config.optimizer.warmup_steps = 50

    prepared = PreparedDataset(
        merged_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        train_samples=100,
        validation_samples=10,
    )

    with (
        patch(f"{CONFIG_MODULE}.prepare_dataset", return_value=prepared),
        patch(f"{CONFIG_MODULE}.DatasetValidator"),
        patch(f"{CONFIG_MODULE}.estimate_dataset_sequence_lengths", return_value=None),
        patch(f"{CONFIG_MODULE}._configure_datasets"),
        patch(f"{CONFIG_MODULE}._configure_moe_backend"),
        patch(f"{CONFIG_MODULE}.build_wandb_config", return_value=None),
        patch(f"{CONFIG_MODULE}.build_mlflow_config", return_value=None),
    ):
        compiled = compile_automodel_config(config, tmp_path, MagicMock())

    assert compiled["packed_sequence"]["packed_sequence_size"] == 2048
    assert compiled["step_scheduler"]["max_steps"] == 2
    assert compiled["step_scheduler"]["val_every_steps"] == 1
    assert compiled["lr_scheduler"]["lr_warmup_steps"] == 1


def test_the_reporting_block_reaches_the_recipe_config(tmp_path: Path) -> None:
    """The compiler writes what AutomodelRecipeWrapper reads, under the same key.

    The recipe config file is the only channel from the compiled job to the
    training process, and `_progress_reporting` is ours rather than the recipe's,
    so nothing upstream validates it. Both sides were separately covered --
    test_finetune.py exercises the reader against a hand-built block -- and that
    is exactly the arrangement where a rename passes every test and the run
    quietly reports at the default forever. This is the test that fails.
    """
    from nmp.customization_common.training.reporting import ProgressReportingConfig

    fixture = (
        Path(__file__).parents[3] / "contract" / "input_configs" / "llama-3.2-1b" / "llama_3_2_1b_lora_packing.json"
    )
    raw = json.loads(fixture.read_text())
    raw.pop("backend")
    config = TrainingStepConfig.model_validate(raw)
    config.schedule.progress_reporting = ProgressReportingConfig(
        time_series_metrics=["*_loss"], min_report_interval_seconds=30
    )

    prepared = PreparedDataset(
        merged_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        train_samples=100,
        validation_samples=10,
    )

    with (
        patch(f"{CONFIG_MODULE}.prepare_dataset", return_value=prepared),
        patch(f"{CONFIG_MODULE}.DatasetValidator"),
        patch(f"{CONFIG_MODULE}.estimate_dataset_sequence_lengths", return_value=None),
        patch(f"{CONFIG_MODULE}._configure_datasets"),
        patch(f"{CONFIG_MODULE}._configure_moe_backend"),
        patch(f"{CONFIG_MODULE}.build_wandb_config", return_value=None),
        patch(f"{CONFIG_MODULE}.build_mlflow_config", return_value=None),
    ):
        compiled = compile_automodel_config(config, tmp_path, MagicMock())

    # Exact equality, not a subset check: the reader in test_finetune.py is written
    # against this literal shape, so a key renamed or dropped on either side breaks
    # one of the two. A subset check here would let the writer grow a key the reader
    # never looks at, which is the same silence in a different place.
    assert compiled["_progress_reporting"] == {
        "time_series_metrics": ["*_loss"],
        "min_report_interval_seconds": 30,
    }


def test_compile_cross_encoder_recipe_selects_cross_encoder_model(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[3] / "contract" / "input_configs" / "embed-1b" / "embed_1b_full_sft.json"
    raw = json.loads(fixture.read_text())
    raw.pop("backend")
    config = TrainingStepConfig.model_validate(raw)
    config.training.recipe = TrainingRecipe.CROSS_ENCODER

    prepared = PreparedDataset(
        merged_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        train_samples=100,
        validation_samples=10,
    )

    with (
        patch(f"{CONFIG_MODULE}.prepare_dataset", return_value=prepared),
        patch(f"{CONFIG_MODULE}.DatasetValidator"),
        patch(f"{CONFIG_MODULE}._configure_datasets"),
        patch(f"{CONFIG_MODULE}.build_wandb_config", return_value=None),
        patch(f"{CONFIG_MODULE}.build_mlflow_config", return_value=None),
    ):
        compiled = compile_automodel_config(config, tmp_path, MagicMock())

    assert compiled["model"]["_target_"].endswith("NeMoAutoModelCrossEncoder.from_pretrained")
    assert compiled["model"]["num_labels"] == 1
    assert "loss_fn" not in compiled


def _embed_training_config(tmp_path: Path, **overrides: Any) -> tuple[TrainingStepConfig, PreparedDataset]:
    fixture = Path(__file__).parents[3] / "contract" / "input_configs" / "embed-1b" / "embed_1b_full_sft.json"
    raw = json.loads(fixture.read_text())
    raw.pop("backend")
    config = TrainingStepConfig.model_validate(raw)
    for key, value in overrides.items():
        setattr(config, key, value)
    prepared = PreparedDataset(
        merged_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        train_samples=100,
        validation_samples=10,
    )
    return config, prepared


def _compile_retrieval(config: TrainingStepConfig, tmp_path: Path, prepared: PreparedDataset) -> dict[str, Any]:
    with (
        patch(f"{CONFIG_MODULE}.prepare_dataset", return_value=prepared),
        patch(f"{CONFIG_MODULE}.DatasetValidator"),
        patch(f"{CONFIG_MODULE}._configure_datasets"),
        patch(f"{CONFIG_MODULE}.build_wandb_config", return_value=None),
        patch(f"{CONFIG_MODULE}.build_mlflow_config", return_value=None),
    ):
        return compile_automodel_config(config, tmp_path, MagicMock())


def test_auto_recipe_maps_cross_encoder_head_to_cross_encoder_model(tmp_path: Path) -> None:
    config, prepared = _embed_training_config(tmp_path)
    config.training.recipe = TrainingRecipe.AUTO
    config.model.is_embedding_model = False
    config.model.checkpoint_head_type = "cross_encoder"

    compiled = _compile_retrieval(config, tmp_path, prepared)

    assert resolve_compiled_recipe(config) == TrainingRecipe.CROSS_ENCODER
    assert compiled["model"]["_target_"].endswith("NeMoAutoModelCrossEncoder.from_pretrained")
    assert compiled["optimizer"]["_target_"] == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
    assert compiled["model"]["attn_implementation"] == "sdpa"


def test_auto_recipe_causal_lm_head_ignores_stale_embedding_alias(tmp_path: Path) -> None:
    config, _ = _embed_training_config(tmp_path)
    config.training.recipe = TrainingRecipe.AUTO
    config.model.is_embedding_model = True
    config.model.checkpoint_head_type = "causal_lm"

    assert resolve_compiled_recipe(config) == TrainingRecipe.SFT


def test_auto_recipe_prefers_cross_encoder_head_over_stale_embedding_alias(tmp_path: Path) -> None:
    config, prepared = _embed_training_config(tmp_path)
    config.training.recipe = TrainingRecipe.AUTO
    config.model.is_embedding_model = True
    config.model.checkpoint_head_type = "cross_encoder"

    compiled = _compile_retrieval(config, tmp_path, prepared)

    assert resolve_compiled_recipe(config) == TrainingRecipe.CROSS_ENCODER
    assert compiled["model"]["_target_"].endswith("NeMoAutoModelCrossEncoder.from_pretrained")


def test_bi_encoder_compile_uses_fused_adam_and_job_retrieval_config(tmp_path: Path) -> None:
    config, prepared = _embed_training_config(
        tmp_path,
        retrieval=RetrievalConfig(query_prefix="query: ", passage_prefix="passage: ", train_n_passages=6),
    )
    config.training.recipe = TrainingRecipe.BI_ENCODER

    compiled = _compile_retrieval(config, tmp_path, prepared)

    assert compiled["optimizer"]["_target_"] == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
    assert compiled["model"]["attn_implementation"] == "sdpa"
    assert compiled["model"]["_target_"].endswith("NeMoAutoModelBiEncoder.from_pretrained")


@pytest.mark.parametrize(
    ("optimizer_name", "recipe", "expected_target"),
    [
        ("auto", TrainingRecipe.SFT, "torch.optim.Adam"),
        (
            "auto",
            TrainingRecipe.BI_ENCODER,
            "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
        ),
        ("Adam", TrainingRecipe.BI_ENCODER, "torch.optim.Adam"),
        ("AdamW", TrainingRecipe.CROSS_ENCODER, "torch.optim.AdamW"),
        (
            "FusedAdam",
            TrainingRecipe.SFT,
            "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
        ),
    ],
)
def test_optimizer_selection_is_explicit_after_auto_resolution(
    optimizer_name: str,
    recipe: TrainingRecipe,
    expected_target: str,
) -> None:
    config = MagicMock()
    config.optimizer.optimizer_name = optimizer_name

    assert _resolve_optimizer_target(config, recipe) == expected_target
