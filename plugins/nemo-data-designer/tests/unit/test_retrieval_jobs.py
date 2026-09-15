# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import data_designer.config as dd
import pandas as pd
import pytest
from nemo_data_designer_plugin.jobs.create import CreateJob
from nemo_data_designer_plugin.jobs.retrieval_generate import RetrievalGenerateJob
from nemo_data_designer_plugin.jobs.retrieval_prepare import RetrievalPrepareJob, _materialize_input
from nemo_data_designer_plugin.jobs.retrieval_run import RetrievalRunJob
from nemo_data_designer_plugin.jobs.retrieval_spec import (
    RetrievalGenerateJobConfig,
    RetrievalGenerateStepConfig,
    RetrievalMiningOptions,
    RetrievalPrepareJobConfig,
    RetrievalPrepareStepConfig,
    RetrievalPreviewSpec,
    RetrievalRunJobConfig,
)
from nemo_data_designer_plugin.jobs.spec import DataDesignerJobConfig
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec, PlatformJobStep
from pydantic import ValidationError


def _steps(compiled: PlatformJobSpec) -> list[PlatformJobStep]:
    return list(compiled["steps"])


def _executor(step: PlatformJobStep) -> dict[str, Any]:
    return cast(dict[str, Any], step["executor"])


def _generate_config(**overrides: object) -> RetrievalGenerateJobConfig:
    payload = {
        "corpus": "default/docs",
        "provider": "default/nvidia-build",
        "artifact_extraction_model": "nvidia/nemotron-3-nano-30b-a3b",
        "qa_generation_model": "nvidia/nemotron-3-nano-30b-a3b",
        "quality_judge_model": "nvidia/nemotron-3-nano-30b-a3b",
        "embed_model": "nvidia/nemotron-3-embed-1b",
    }
    payload.update(overrides)
    return RetrievalGenerateJobConfig.model_validate(payload)


def _ctx(tmp_path: Path) -> Mock:
    ephemeral = tmp_path / "ephemeral"
    persistent = tmp_path / "persistent"
    ephemeral.mkdir()
    persistent.mkdir()
    ctx = Mock()
    ctx.workspace = "default"
    ctx.storage.ephemeral = ephemeral
    ctx.storage.persistent = persistent
    ctx.results.save.return_value = SimpleNamespace(model_dump=lambda: {"name": "artifacts"})
    return ctx


@pytest.mark.asyncio
async def test_retrieval_generate_compile_is_cpu() -> None:
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    compiled = await RetrievalGenerateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    steps = _steps(compiled)
    assert len(steps) == 1
    executor = _executor(steps[0])
    assert executor["provider"] == "cpu"
    assert "nmp-cpu-tasks" in executor["container"]["image"]


@pytest.mark.asyncio
async def test_retrieval_generate_compile_ignores_subprocess_profiles() -> None:
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    compiled = await RetrievalGenerateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    executor = _executor(_steps(compiled)[0])
    assert executor["provider"] == "cpu"
    assert executor["container"]["command"] == ["nemo_data_designer_plugin.jobs.retrieval_generate"]


@pytest.mark.asyncio
async def test_retrieval_prepare_compile_uses_one_container_profile() -> None:
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="default/stage0", enable_mining=True),
        phase="convert",
        model_fileset="default/retrieval-model",
        model_trust_remote_code=True,
    )
    compiled = await RetrievalPrepareJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
        profile="gpu",
    )
    steps = _steps(compiled)
    assert [_executor(step)["provider"] for step in steps] == ["cpu", "cpu", "gpu"]
    assert [_executor(step)["profile"] for step in steps] == ["gpu", "gpu", "gpu"]
    assert "nmp-automodel-training" in _executor(steps[2])["container"]["image"]
    assert _executor(steps[2])["container"]["command"] == ["nmp.automodel.tasks.retrieval_mine"]
    assert _executor(steps[1])["container"]["command"] == [
        "nmp.customization_common.tasks.file_io",
        "--service-source",
        "automodel",
        "--service-name",
        "customizer",
    ]
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    with patch(
        "nemo_data_designer_plugin.jobs.retrieval_generate.create_validation_context",
        return_value=dd_ctx,
    ):
        step = await RetrievalGenerateJob.to_spec(
            _generate_config(),
            workspace="default",
            entity_client=Mock(),
            async_sdk=AsyncMock(),
            is_local=False,
        )
    assert isinstance(step, RetrievalGenerateStepConfig)
    assert step.model_providers == providers
    assert step.chat_provider_name == "default/nvidia-build"
    assert step.embed_provider_name == "default/nvidia-build"


def test_retrieval_generate_run_writes_artifacts(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    result = SimpleNamespace(dataset_name="retrieval_sdg", num_records=3, output_path=tmp_path / "out.jsonl")
    with (
        patch("nemo_data_designer_plugin.jobs.retrieval_generate.materialize_corpus", return_value=corpus),
        patch(
            "nemo_data_designer_plugin.retrieval.generation.execute_generation",
            return_value=result,
        ) as run_generation,
        patch("nemo_data_designer_plugin.retrieval.generation.build_generation_run_config") as build_cfg,
    ):
        build_cfg.return_value = SimpleNamespace()
        output = RetrievalGenerateJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())
    run_generation.assert_called_once()
    assert output["exit_code"] == 0
    ctx.results.save.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_prepare_convert_only_is_cpu() -> None:
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="default/stage0", enable_mining=False),
        phase="convert",
    )
    compiled = await RetrievalPrepareJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    steps = _steps(compiled)
    assert len(steps) == 1
    assert _executor(steps[0])["provider"] == "cpu"


@pytest.mark.asyncio
async def test_retrieval_prepare_resolves_model_fileset_for_mining() -> None:
    model = SimpleNamespace(
        workspace="nvidia",
        name="Nemotron-3-Embed-1B-BF16",
        fileset="nvidia/nemotron-3-embed-1b-bf16",
        trust_remote_code=True,
    )
    with patch(
        "nemo_data_designer_plugin.jobs.retrieval_prepare.fetch_model_entity",
        new=AsyncMock(return_value=model),
    ) as fetch:
        step = await RetrievalPrepareJob.to_spec(
            RetrievalPrepareJobConfig(sdg_input="default/stage0", enable_mining=True),
            workspace="default",
            entity_client=Mock(),
            async_sdk=AsyncMock(),
            is_local=False,
        )

    fetch.assert_awaited_once()
    assert isinstance(step, RetrievalPrepareStepConfig)
    assert step.model_fileset == model.fileset
    assert step.model_trust_remote_code is True


@pytest.mark.asyncio
async def test_retrieval_prepare_compile_adds_gpu_mining_step() -> None:
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="default/stage0", enable_mining=True),
        phase="convert",
        model_fileset="default/retrieval-model",
        model_trust_remote_code=True,
    )
    compiled = await RetrievalPrepareJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    steps = _steps(compiled)
    assert len(steps) == 3
    assert _executor(steps[0])["provider"] == "cpu"
    assert _executor(steps[1])["provider"] == "cpu"
    assert "nmp-customizer-tasks" in _executor(steps[1])["container"]["image"]
    assert steps[1]["config"]["download"] == [
        {"src": {"workspace": "default", "name": "retrieval-model"}, "dest": "model"}
    ]
    assert _executor(steps[2])["provider"] == "gpu"
    assert "nmp-automodel-training" in _executor(steps[2])["container"]["image"]
    assert _executor(steps[2])["container"]["command"] == ["nmp.automodel.tasks.retrieval_mine"]
    for step in steps:
        environment = {item["name"]: item["value"] for item in step["environment"]}
        assert environment["NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH"] == "/var/run/scratch/job"
    assert {item["name"]: item["value"] for item in steps[2]["environment"]} == {
        "NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH": "/var/run/scratch/job",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_retrieval_prepare_convert_emits_eval_layout(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    sdg = ctx.storage.persistent / "sdg"
    sdg.mkdir()
    jsonl = sdg / "qa.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    train_file = tmp_path / "converted" / "train.json"
    train_file.parent.mkdir()
    train_file.write_text(
        json.dumps({"corpus": {}, "data": [{"question": "q", "pos_doc": ["p"], "neg_doc": []}]}), encoding="utf-8"
    )
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="sdg", enable_mining=False),
        phase="convert",
    )
    conversion = SimpleNamespace(train_file=train_file)

    def _fake_convert(**kwargs: Any) -> SimpleNamespace:
        eval_dir = Path(kwargs["output_dir"]) / "eval_beir"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
        (eval_dir / "queries.jsonl").write_text("{}\n", encoding="utf-8")
        qrels = eval_dir / "qrels"
        qrels.mkdir()
        (qrels / "test.tsv").write_text("qid\tdocid\trel\n", encoding="utf-8")
        return conversion

    with patch(
        "nemo_data_designer_plugin.retrieval.conversion.execute_conversion",
        side_effect=_fake_convert,
    ):
        output = RetrievalPrepareJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())
    assert output["exit_code"] == 0
    staged = ctx.storage.persistent / "stage1_data_prep"
    assert (staged / "eval_beir" / "corpus.jsonl").exists()
    assert (staged / "training.jsonl").exists()
    assert (staged / "train.json").exists()


def test_retrieval_prepare_rejects_mine_phase(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="default/stage0", enable_mining=True),
        phase="mine",
    )
    with pytest.raises(RuntimeError, match="retrieval_mine"):
        RetrievalPrepareJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())


def test_retrieval_prepare_reports_missing_train_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    train_input = ctx.storage.persistent / "empty-train-input"
    train_input.mkdir()
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(train_input_file="empty-train-input"),
        phase="convert",
    )

    with pytest.raises(FileNotFoundError, match="No train.json"):
        RetrievalPrepareJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())


def test_retrieval_prepare_rejects_host_filesystem_paths(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    host_corpus = tmp_path / "host-sdg"
    host_corpus.mkdir()
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input=str(host_corpus), enable_mining=False),
        phase="convert",
    )

    with pytest.raises(ValueError):
        RetrievalPrepareJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())


@pytest.mark.asyncio
async def test_retrieval_run_compile_chains_generate_then_prepare() -> None:
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    spec = RetrievalRunJobConfig(
        generate=_generate_config(),
        prepare=RetrievalPrepareJobConfig(enable_mining=False),
    )
    with patch(
        "nemo_data_designer_plugin.jobs.retrieval_generate.create_validation_context",
        return_value=dd_ctx,
    ):
        compiled = await RetrievalRunJob.compile(
            workspace="default",
            spec=spec,
            entity_client=Mock(),
            job_name=None,
            async_sdk=AsyncMock(),
        )
    names = [step["name"] for step in _steps(compiled)]
    assert names[0] == "retrieval-generate"
    assert "retrieval-prepare-convert" in names
    assert all(_executor(step)["provider"] == "cpu" for step in _steps(compiled))


def test_create_job_schema_rejects_dataframe_seeds() -> None:
    config = DataDesignerJobConfig(
        num_records=1,
        config=dd.DataDesignerConfig(
            columns=[dd.ExpressionColumnConfig(name="value", expr="1")],
            seed_config=dd.SeedConfig(source=dd.DataFrameSeedSource(df=pd.DataFrame({"value": [1]}))),
        ),
    )
    with pytest.raises(ValidationError):
        DataDesignerJobConfig.model_validate(config.model_dump())
    assert CreateJob.input_spec_schema is DataDesignerJobConfig


def test_preview_spec_reuses_generate_config() -> None:
    spec = RetrievalPreviewSpec(generate=_generate_config(), num_records=2)
    assert spec.num_records == 2
    assert spec.generate.embed_model == "nvidia/nemotron-3-embed-1b"


def test_prepare_mining_options_are_typed() -> None:
    spec = RetrievalPrepareJobConfig(sdg_input="default/stage0", mining=RetrievalMiningOptions(corpus_chunk_size=10000))
    assert spec.mining.corpus_chunk_size == 10000
    assert spec.mining.hard_neg_margin_type == "perc"
    with pytest.raises(ValidationError):
        RetrievalPrepareJobConfig.model_validate(
            {"sdg_input": "default/stage0", "mining": {"hard_neg_margin_type": "relative"}}
        )


def test_prepare_rejects_staged_path_that_escapes_job_storage(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    this_job = jobs_root / "this-job"
    other_job = jobs_root / "other-job"
    this_job.mkdir(parents=True)
    other_job.mkdir()
    (other_job / "train.json").write_text("secret", encoding="utf-8")

    ctx = _ctx(this_job)
    with pytest.raises(ValueError, match="escapes job storage"):
        _materialize_input("../../other-job/train.json", ctx.storage.ephemeral / "dest", ctx, Mock())


def test_prepare_uses_contained_staged_input(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    staged = ctx.storage.persistent / "stage0" / "train.json"
    staged.parent.mkdir()
    staged.write_text("{}", encoding="utf-8")
    resolved = _materialize_input("stage0/train.json", ctx.storage.ephemeral / "dest", ctx, Mock())
    assert resolved == staged.resolve()


def test_retrieval_cli_prints_spec() -> None:
    from nemo_data_designer_plugin.cli.retrieval import retrieval_app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        retrieval_app,
        [
            "generate",
            "--corpus",
            "default/docs",
            "--provider",
            "default/nvidia-build",
            "--chat-model",
            "nvidia/nemotron-3-nano-30b-a3b",
            "--embed-model",
            "nvidia/nemotron-3-embed-1b",
            "--print-spec",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["corpus"] == "default/docs"
    assert payload["qa_generation_model"] == "nvidia/nemotron-3-nano-30b-a3b"
    assert payload["embed_model"] == "nvidia/nemotron-3-embed-1b"


def test_retrieval_cli_help_distinguishes_helpers_from_job_commands() -> None:
    from nemo_data_designer_plugin.cli.retrieval import retrieval_app
    from typer.testing import CliRunner

    result = CliRunner().invoke(retrieval_app, ["--help"])

    assert result.exit_code == 0
    assert "Build specs" in result.output
    assert "retrieval-generate" in result.output
    assert "retrieval-prepare" in result.output
    assert "retrieval-preview" in result.output


def test_retrieval_prepare_cli_requires_exactly_one_input() -> None:
    from nemo_data_designer_plugin.cli.retrieval import retrieval_app
    from typer.testing import CliRunner

    runner = CliRunner()
    missing = runner.invoke(retrieval_app, ["prepare"])
    both = runner.invoke(
        retrieval_app,
        ["prepare", "--sdg-input", "default/stage0", "--train-input-file", "default/train"],
    )
    assert missing.exit_code != 0
    assert both.exit_code != 0
    assert "exactly one" in missing.output
    assert "exactly one" in both.output


def test_retrieval_cli_shell_quotes_workspace_and_spec() -> None:
    from nemo_data_designer_plugin.cli.retrieval import retrieval_app
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        retrieval_app,
        [
            "prepare",
            "--sdg-input",
            "default/stage0",
            "--workspace",
            "team's-ws",
        ],
    )
    assert result.exit_code == 0
    assert "retrieval-prepare" in result.output
    quoted = shlex.join(["--workspace", "team's-ws"])
    assert quoted in result.output
    assert "--spec" in result.output


def _secret_env(step: PlatformJobStep) -> dict[str, str]:
    return {
        item["name"]: item["from_secret"]["name"] for item in step["environment"] if item.get("from_secret") is not None
    }


@pytest.mark.asyncio
async def test_retrieval_generate_compile_projects_hf_token_secret() -> None:
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(corpus="hf://example/private-corpus", hf_token_secret="default/hf-token"),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    compiled = await RetrievalGenerateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    step = _steps(compiled)[0]
    assert _secret_env(step) == {"HF_TOKEN": "default/hf-token"}
    # The reference travels as a secret ref, never as a plaintext value in the spec.
    assert "default/hf-token" not in [item.get("value") for item in step["environment"]]


@pytest.mark.asyncio
async def test_retrieval_generate_compile_omits_hf_token_without_secret() -> None:
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    compiled = await RetrievalGenerateJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    assert _secret_env(_steps(compiled)[0]) == {}


@pytest.mark.asyncio
async def test_retrieval_prepare_compile_projects_hf_token_secret_on_convert_only() -> None:
    spec = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(
            sdg_input="hf://example/private-stage0",
            enable_mining=True,
            hf_token_secret="hf-token",
        ),
        phase="convert",
        model_fileset="default/retrieval-model",
    )
    compiled = await RetrievalPrepareJob.compile(
        workspace="default",
        spec=spec,
        entity_client=Mock(),
        job_name=None,
        async_sdk=AsyncMock(),
    )
    steps = _steps(compiled)
    assert _secret_env(steps[0]) == {"HF_TOKEN": "hf-token"}
    # Mining runs offline in the automodel image, so it gets no Hub credential.
    assert _secret_env(steps[2]) == {}


def test_retrieval_generate_run_forwards_hf_token_to_corpus_staging(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    spec = RetrievalGenerateStepConfig(
        job_config=_generate_config(corpus="hf://example/private-corpus", hf_token_secret="default/hf-token"),
        model_providers=[dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")],
        chat_provider_name="default/nvidia-build",
        embed_provider_name="default/nvidia-build",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    result = SimpleNamespace(dataset_name="retrieval_sdg", num_records=3, output_path=tmp_path / "out.jsonl")
    with (
        patch(
            "nemo_data_designer_plugin.jobs.retrieval_generate.materialize_corpus",
            return_value=corpus,
        ) as materialize,
        patch("nemo_data_designer_plugin.retrieval.generation.execute_generation", return_value=result),
        patch("nemo_data_designer_plugin.retrieval.generation.build_generation_run_config") as build_cfg,
    ):
        build_cfg.return_value = SimpleNamespace()
        RetrievalGenerateJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=Mock())

    assert materialize.call_args.kwargs["hf_token"] == "hf_secret_value"


def test_retrieval_prepare_materialize_input_forwards_hf_token(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    with patch(
        "nemo_data_designer_plugin.jobs.retrieval_prepare.materialize_corpus",
        return_value=tmp_path / "staged",
    ) as materialize:
        _materialize_input("hf://example/private-stage0", tmp_path / "dest", ctx, Mock())

    assert materialize.call_args.kwargs["hf_token"] == "hf_secret_value"
