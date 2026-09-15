# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import ClassVar, cast

from nemo_data_designer_plugin.jobs.retrieval_common import (
    RETRIEVAL_MINE_MODULE,
    model_download_step,
    retrieval_step,
    work_dir,
)
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalPrepareJobConfig, RetrievalPrepareStepConfig
from nemo_data_designer_plugin.retrieval.corpus import hf_token_from_env, materialize_corpus
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nmp.customization_common.retrieval.inline import move_aux_files_to_additional, wrapped_to_inline_jsonl
from nmp.customization_common.service.platform_client import fetch_model_entity
from pydantic import BaseModel


class RetrievalPrepareJob(NemoJob):
    name: ClassVar[str] = "retrieval-prepare"
    description: ClassVar[str] = "Convert retrieval SDG output to eval_beir and training JSONL (Nemotron Stage 1)."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    input_spec_schema = RetrievalPrepareJobConfig
    spec_schema = RetrievalPrepareStepConfig

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> BaseModel:
        job_config = cast(RetrievalPrepareJobConfig, input_spec)
        if not job_config.sdg_input and not job_config.train_input_file:
            raise ValueError("One of sdg_input or train_input_file is required.")
        if job_config.sdg_input and job_config.train_input_file:
            raise ValueError("sdg_input and train_input_file are mutually exclusive.")
        if not job_config.enable_mining:
            return RetrievalPrepareStepConfig(job_config=job_config, phase="convert")

        model = await fetch_model_entity(job_config.model, workspace, cast(AsyncNeMoPlatform, async_sdk))
        if not model.fileset:
            raise ValueError(
                f"Model '{model.workspace}/{model.name}' has no fileset. "
                "Attach model weights before enabling retrieval mining."
            )
        return RetrievalPrepareStepConfig(
            job_config=job_config,
            phase="convert",
            model_fileset=model.fileset,
            model_trust_remote_code=model.trust_remote_code or False,
        )

    @classmethod
    async def compile(
        cls,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        spec = cast(RetrievalPrepareStepConfig, spec)
        steps = [
            await retrieval_step(
                "retrieval-prepare-convert",
                "nemo_data_designer_plugin.jobs.retrieval_prepare",
                spec,
                profile=profile,
                async_sdk=async_sdk,
                hf_token_secret=spec.job_config.hf_token_secret,
            )
        ]
        if spec.job_config.enable_mining:
            if not spec.model_fileset:
                raise ValueError("Retrieval mining requires a resolved model fileset")
            mine_spec = spec.model_copy(update={"phase": "mine"})
            steps.append(
                await model_download_step(
                    spec.model_fileset,
                    profile=profile,
                    async_sdk=async_sdk,
                )
            )
            steps.append(
                await retrieval_step(
                    "retrieval-prepare-mine",
                    RETRIEVAL_MINE_MODULE,
                    mine_spec,
                    profile=profile,
                    async_sdk=async_sdk,
                    gpu=True,
                )
            )
        return PlatformJobSpec(steps=steps)

    def run(self, config: dict, ctx: JobContext, sdk: NeMoPlatform) -> dict:
        step = RetrievalPrepareStepConfig.model_validate(config)
        if step.phase == "mine":
            raise RuntimeError("Mining runs as nmp.automodel.tasks.retrieval_mine, not this module")
        return _run_convert(step.job_config, work_dir(ctx, "stage1_data_prep"), ctx, sdk)


def _materialize_input(ref: str, dest: Path, ctx: JobContext, sdk: NeMoPlatform) -> Path:
    hf_token = hf_token_from_env()
    if Path(ref).is_absolute():
        return materialize_corpus(ref, dest=dest, sdk=sdk, workspace=ctx.workspace, hf_token=hf_token)
    storage_root = (ctx.storage.persistent or ctx.storage.ephemeral).resolve()
    staged = (storage_root / ref).resolve()
    if not staged.is_relative_to(storage_root):
        raise ValueError(f"Staged input path escapes job storage: {ref}")
    if staged.exists():
        return staged
    return materialize_corpus(ref, dest=dest, sdk=sdk, workspace=ctx.workspace, hf_token=hf_token)


def _run_convert(job: RetrievalPrepareJobConfig, output_dir: Path, ctx: JobContext, sdk: NeMoPlatform) -> dict:
    if job.train_input_file:
        train_file = _materialize_input(
            job.train_input_file,
            ctx.storage.ephemeral / "train_input",
            ctx,
            sdk,
        )
        if train_file.is_dir():
            candidate = train_file / "train.json"
            if candidate.exists():
                train_file = candidate
            else:
                matches = list(train_file.rglob("train.json"))
                if not matches:
                    raise FileNotFoundError(f"No train.json under {train_file}")
                train_file = matches[0]
    else:
        assert job.sdg_input is not None
        sdg_root = _materialize_input(
            job.sdg_input,
            ctx.storage.ephemeral / "sdg_input",
            ctx,
            sdk,
        )
        from nemo_data_designer_plugin.retrieval.conversion import execute_conversion

        input_path = _resolve_generation_input(sdg_root, job.generation_file)
        conversion = execute_conversion(
            input_path=input_path,
            output_dir=output_dir,
            corpus_id=job.corpus_id,
            quality_threshold=job.quality_threshold,
            train_ratio=job.train_ratio,
            val_ratio=job.val_ratio,
            seed=job.seed,
            max_pos_docs=job.max_pos_docs,
            use_group_id_in_eval=job.use_group_id_in_eval,
            split_strategy=job.split_strategy,
        )
        if conversion.train_file is None:
            raise RuntimeError("Retrieval SDG conversion did not produce a training file")
        train_file = conversion.train_file

    source_train = Path(train_file)
    train_file = _stage_train_file(source_train, output_dir)
    _assert_nonempty_training_split(train_file)
    if job.enable_mining:
        _stage_corpus_for_mining(source_train, output_dir)

    if not job.enable_mining:
        inline_path = output_dir / "training.jsonl"
        wrapped_to_inline_jsonl(train_file, inline_path, output_dir / "corpus" / "train.parquet")
        move_aux_files_to_additional(output_dir)

    artifacts = ctx.results.save(name="artifacts", local_path=output_dir)
    return {
        "exit_code": 0,
        "workspace": ctx.workspace,
        "train_file": str(train_file),
        "results": {"artifacts": artifacts.model_dump()},
    }


def _assert_nonempty_training_split(train_file: Path) -> None:
    """Fail before mining when conversion parked every query in the test split."""
    try:
        payload = json.loads(train_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read training file {train_file}: {exc}") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(rows, list) and not rows:
        raise RuntimeError(
            "Retrieval conversion produced an empty training split. Tiny corpora can "
            "land entirely in the test split. Generate with more source files "
            "(recommended 50+ documents) or raise train_ratio before enabling mining."
        )


def _stage_train_file(train_file: Path, output_dir: Path) -> Path:
    dest = output_dir / "train.json"
    if train_file.resolve() == dest.resolve():
        return dest
    if train_file.is_file():
        shutil.copy2(train_file, dest)
        return dest
    raise FileNotFoundError(f"Training file is not a file: {train_file}")


def _stage_corpus_for_mining(train_file: Path, output_dir: Path) -> None:
    """Copy sibling ``corpus/`` so the miner can load merlin_metadata.json.

    ``train_input_file`` skips conversion, which normally writes that directory.
    Mining then fails with ``Metadata File for Corpus does not exist``.
    """
    src = train_file.parent / "corpus"
    dest = output_dir / "corpus"
    if not src.is_dir():
        return
    if src.resolve() == dest.resolve():
        return
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _resolve_generation_input(staged: Path, generation_file: str) -> Path:
    """Use a materialized file, or the named file inside a Stage 0 directory."""
    if staged.is_file():
        return staged
    candidate = (staged / generation_file).resolve()
    if not candidate.is_relative_to(staged.resolve()):
        raise ValueError(f"generation_file escapes Stage 0 directory: {generation_file}")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Expected Stage 0 file {generation_file!r} under {staged}. "
        "Live generate writes generation_result.json; skip-SDG dumps set generation_file "
        "or pass sdg_input as fileset#path / hf://org/dataset@rev/file."
    )


if __name__ == "__main__":
    from nemo_data_designer_plugin.jobs.retrieval_bridge import run_job_module

    raise SystemExit(run_job_module(RetrievalPrepareJob, RetrievalPrepareStepConfig))
