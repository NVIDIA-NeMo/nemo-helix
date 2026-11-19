# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF convert job: evaluation trajectories to an SFT dataset.

The input is the evaluation runs' output filesets (Jobs ``job-fileset-<job>``). Each holds one or more
agent-eval result bundles, ``<dir>/trials.jsonl`` beside ``scores.jsonl``, with each trial's ATIF
trajectory located through its ``evidence.descriptors``. The job:

1. lists trials and keeps completed ones that pass ``pass_filter``,
2. holds out whole tasks for validation,
3. converts every kept trajectory to one SFT row per assistant turn (see ``nemo_customizer.atif_sft``),
4. uploads ``train.jsonl``, ``validation.jsonl`` and ``output.json`` under ``nodes/<node>/`` in the
   workflow fileset.

``stub_row`` and ``stub_output`` stay as fixtures for plumbing tests that need a schema-valid
``output.json`` without running a conversion.
"""

from __future__ import annotations

import json
import logging
import posixpath
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.refs import parse_entity_ref
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_customizer.atif_sft import PassFilter, convert_trajectory, held_out_tasks, join_bundle_path, trace_ref
from nemo_customizer.token_budget import ChatTemplateCounter, MaxTokens, TokenCounter, apply_budget

logger = logging.getLogger(__name__)

SOURCE = "customization"

#: Trial ids listed in ``output.json`` for chains the length limit dropped. The node outputs live on the
#: workflow entity, so the list is a sample; `skipped` has the full counts.
MAX_REPORTED_TRIALS = 50

#: Files of a model checkpoint that load its tokenizer and chat template. Weights are never read.
TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "generation_config.json",
    }
)


class AtifConvertSpec(BaseModel):
    """What the workflow controller hands the convert job."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    fileset: str
    node_key: str
    trajectory_filesets: list[str] = Field(
        default_factory=list,
        description="Evaluation output filesets, as `workspace/name` (or `name` in the workflow workspace).",
    )
    pass_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Score filter, see `nemo_customizer.atif_sft.PassFilter`. Empty keeps every completed trial.",
    )
    validation_fraction: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="Share of tasks held out for validation. Whole tasks, never single trials.",
    )
    student_model: str | None = Field(
        default=None,
        description=(
            "Student model entity, `workspace/name` (or `name` in the job workspace). Its tokenizer and chat "
            "template measure row length. Required when `max_tokens` is enabled."
        ),
    )
    max_tokens: MaxTokens | None = Field(
        default=None,
        description="Length limit and what to do with rows over it. Unset leaves length to the trainer.",
    )

    @model_validator(mode="after")
    def _limit_needs_a_student(self) -> AtifConvertSpec:
        if self.max_tokens is not None and self.max_tokens.enabled and not self.student_model:
            raise ValueError("max_tokens needs student_model: the student's tokenizer measures the rows")
        return self

    @property
    def budget(self) -> MaxTokens | None:
        """The limit when it is set and enabled."""
        return self.max_tokens if self.max_tokens is not None and self.max_tokens.enabled else None


def fileset_prefix(node_key: str) -> str:
    """Fileset directory for one node. Brackets in matrix keys are not path-safe."""
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in node_key).strip("_")
    if not safe:
        raise ValueError(f"node key {node_key!r} has no path-safe characters")
    return f"nodes/{safe}"


def stub_row() -> dict[str, Any]:
    """One ``SFTChatMessage`` conversation: system, user, thinking, tool call, tool result, reply."""
    return {
        "messages": [
            {"role": "system", "content": "You are a stub teacher. This row is plumbing, not a real trajectory."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "thinking": "Add the two numbers."},
            {
                "role": "assistant",
                "content": "I will calculate that.",
                "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expression": "2+2"}}}],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def stub_output(
    *, workspace: str, fileset: str, node_key: str, trajectory_filesets: list[str] | None = None
) -> dict[str, Any]:
    """The ``output.json`` the workflow controller reads after this job completes."""
    prefix = fileset_prefix(node_key)
    sources = list(trajectory_filesets or [])
    return {
        "fileset": f"{workspace}/{fileset}",
        "dataset": f"{workspace}/{fileset}#{prefix}/",
        "trajectory_filesets": sources,
        "held_out": [],
        "runs": len(sources),
        "examples": 2,
    }


def _jsonl(row: dict[str, Any]) -> bytes:
    return (json.dumps(row) + "\n").encode("utf-8")


class FilesetSource(Protocol):
    """Read access to evaluation output filesets."""

    def list_paths(self, ref: str) -> list[str]: ...

    def read(self, ref: str, path: str) -> bytes: ...


class FilesClientSource:
    """`FilesetSource` over the Files API. A ref is `workspace/name`, or `name` in the default workspace."""

    def __init__(self, files: FilesClient, default_workspace: str) -> None:
        self._files = files
        self._default_workspace = default_workspace

    def _split(self, ref: str) -> tuple[str, str]:
        workspace, _, name = ref.rpartition("/")
        return workspace or self._default_workspace, name

    def list_paths(self, ref: str) -> list[str]:
        workspace, name = self._split(ref)
        return [item.path for item in self._files.list_files(workspace=workspace, name=name).data().data]

    def read(self, ref: str, path: str) -> bytes:
        workspace, name = self._split(ref)
        return self._files.download_file(workspace=workspace, name=name, path=path).read()


@dataclass(frozen=True)
class _Candidate:
    ref: str
    trial_id: str
    task_id: str
    trace_path: str


@dataclass
class ConversionResult:
    train_path: Path
    validation_path: Path
    output: dict[str, Any]


def _rows(blob: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]


def _candidates(
    source: FilesetSource, filesets: list[str], flt: PassFilter, skipped: Counter[str]
) -> tuple[list[_Candidate], int]:
    """Trials worth converting: completed, passing the filter, with an ATIF trace. Also the trials seen."""
    found: dict[tuple[str, str], _Candidate] = {}
    seen: set[tuple[str, str]] = set()
    for ref in filesets:
        paths = source.list_paths(ref)
        present = set(paths)
        bundles = [p for p in sorted(paths) if posixpath.basename(p) == "trials.jsonl"]
        if not bundles:
            raise RuntimeError(f"{ref} holds no agent-eval results (no trials.jsonl)")
        for trials_path in bundles:
            bundle = posixpath.dirname(trials_path)
            scores_path = posixpath.join(bundle, "scores.jsonl")
            scores: dict[str, list[dict[str, Any]]] = {}
            if scores_path in present:
                for score in _rows(source.read(ref, scores_path)):
                    scores.setdefault(str(score.get("trial_id")), []).append(score)
            for trial in _rows(source.read(ref, trials_path)):
                key = (ref, str(trial.get("id")))
                if key in seen:  # a retried job attempt repeats trial ids; the first bundle wins
                    continue
                seen.add(key)
                if trial.get("status") != "completed":
                    skipped["trial_not_completed"] += 1
                elif not flt.passes(scores.get(str(trial.get("id")), [])):
                    skipped["filtered_by_score"] += 1
                elif (reference := trace_ref(trial)) is None:
                    skipped["no_atif_trace"] += 1
                else:
                    found[key] = _Candidate(
                        ref=ref,
                        trial_id=str(trial.get("id")),
                        task_id=str(trial.get("task_id")),
                        trace_path=join_bundle_path(bundle, reference),
                    )
    return [found[key] for key in sorted(found)], len(seen)


def convert_filesets(
    source: FilesetSource, spec: AtifConvertSpec, out_dir: Path, counter: TokenCounter | None = None
) -> ConversionResult:
    """Convert the spec's evaluation filesets into ``train.jsonl`` and ``validation.jsonl`` in ``out_dir``.

    Rows are streamed to disk as they are produced: one row per assistant turn repeats its prefix, so
    the dataset is much larger than the trajectories it comes from. ``counter`` measures rows for the
    spec's ``max_tokens`` limit and is required when that limit is enabled.
    """
    flt = PassFilter.model_validate(spec.pass_filter)
    skipped: Counter[str] = Counter()
    candidates, trials_seen = _candidates(source, spec.trajectory_filesets, flt, skipped)
    held_out = held_out_tasks({c.task_id for c in candidates}, spec.validation_fraction)

    train_path = out_dir / "train.jsonl"
    validation_path = out_dir / "validation.jsonl"
    train_rows = validation_rows = runs = truncated_rows = 0
    dropped_trials: list[str] = []
    budget = spec.budget
    limit: tuple[MaxTokens, TokenCounter] | None = None
    if budget is not None:
        if counter is None:
            raise ValueError("a max_tokens limit needs a token counter")
        limit = (budget, counter)
    with train_path.open("wb") as train, validation_path.open("wb") as validation:
        for candidate in candidates:
            try:
                trajectory = json.loads(source.read(candidate.ref, candidate.trace_path))
            except NotFoundError:
                skipped["trace_missing"] += 1
                continue
            except ValueError:  # JSONDecodeError and UnicodeDecodeError
                skipped["trace_invalid"] += 1
                continue
            if not isinstance(trajectory, dict):
                skipped["trace_invalid"] += 1
                continue
            converted = convert_trajectory(trajectory)
            skipped.update(converted.skipped)
            chains = converted.chains
            if limit is not None:
                outcome = apply_budget(chains, *limit)
                chains = outcome.chains
                skipped["chain_over_max_tokens"] += outcome.dropped_chains - outcome.render_errors
                skipped["row_over_max_tokens"] += outcome.dropped_rows
                skipped["token_count_failed"] += outcome.render_errors
                truncated_rows += outcome.truncated_rows
                if outcome.dropped_chains and len(dropped_trials) < MAX_REPORTED_TRIALS:
                    dropped_trials.append(candidate.trial_id)
            rows = [row for chain in chains for row in chain]
            if not rows:
                continue
            runs += 1
            into_validation = candidate.task_id in held_out
            for row in rows:
                (validation if into_validation else train).write(_jsonl(row))
            if into_validation:
                validation_rows += len(rows)
            else:
                train_rows += len(rows)

    if train_rows == 0:
        raise RuntimeError(
            f"no training rows from {trials_seen} trials in {spec.trajectory_filesets}; skipped: {dict(skipped)}"
        )
    validation_from_train = validation_rows == 0
    if validation_from_train:
        # Too few tasks to hold one out: validate on the training rows rather than ship an empty file.
        validation_path.write_bytes(train_path.read_bytes())
        validation_rows = train_rows
        held_out = set()
    prefix = fileset_prefix(spec.node_key)
    output = {
        "fileset": f"{spec.workspace}/{spec.fileset}",
        "dataset": f"{spec.workspace}/{spec.fileset}#{prefix}/",
        "trajectory_filesets": list(spec.trajectory_filesets),
        "held_out": sorted(held_out),
        "runs": runs,
        "examples": train_rows,
        "validation_examples": validation_rows,
        "validation_from_train": validation_from_train,
        "trials_seen": trials_seen,
        "max_tokens": {"behavior": budget.behavior, "tokens": budget.tokens} if budget is not None else None,
        "truncated_rows": truncated_rows,
        "dropped_trials": dropped_trials,
        "skipped": {reason: count for reason, count in sorted(skipped.items()) if count},
    }
    return ConversionResult(train_path=train_path, validation_path=validation_path, output=output)


def _chunks(path: Path, size: int = 1 << 20) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(size):
            yield chunk


def load_student_counter(sdk: NemoClient, student_model: str, workspace: str, directory: Path) -> TokenCounter:
    """A token counter for the student, from the tokenizer files in its model entity's weights fileset.

    Only the tokenizer and chat-template files are downloaded, so this needs no GPU and no weights.
    """
    ref = parse_entity_ref(student_model, workspace)
    model = ModelsClient.from_client(sdk).get_model(name=ref.name, workspace=ref.workspace).data()
    if not model.fileset:
        raise RuntimeError(f"student model {ref.workspace}/{ref.name} has no weights fileset to read a tokenizer from")
    try:
        weights = parse_entity_ref(model.fileset, ref.workspace)
    except ValueError as exc:
        raise RuntimeError(
            f"student model {ref.workspace}/{ref.name} has an unsupported fileset {model.fileset!r}"
        ) from exc
    files = FilesClient.from_client(sdk)
    names = [item.path for item in files.list_files(workspace=weights.workspace, name=weights.name).data().data]
    wanted = [
        name
        for name in names
        if "/" not in name
        and (
            name in TOKENIZER_FILES
            or name.endswith((".jinja", ".tiktoken"))
            or (model.trust_remote_code and name.endswith(".py"))
        )
    ]
    if "tokenizer_config.json" not in wanted:
        raise RuntimeError(f"fileset {weights.workspace}/{weights.name} has no tokenizer_config.json at its root")
    for name in wanted:
        content = files.download_file(workspace=weights.workspace, name=weights.name, path=name).read()
        (directory / name).write_bytes(content)
    return ChatTemplateCounter(_load_tokenizer(directory, trust_remote_code=bool(model.trust_remote_code)))


def _load_tokenizer(directory: Path, *, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer  # heavy; only the length limit needs it

    return AutoTokenizer.from_pretrained(str(directory), trust_remote_code=trust_remote_code)


def _tasks_image() -> tuple[str, list[str]]:
    from nhx.customization_common.service.images import (
        CUSTOMIZER_PYTHON_ENTRYPOINT,
        get_customizer_tasks_image,
    )

    return get_customizer_tasks_image(), CUSTOMIZER_PYTHON_ENTRYPOINT


class AtifConvertJob(NemoJob):
    """Convert evaluation trajectories into the workflow fileset's SFT dataset."""

    name: ClassVar[str] = "atif-convert"
    description: ClassVar[str] = "Convert evaluation ATIF trajectories into an SFT training fileset."
    container: ClassVar[str] = "nhx-customizer-tasks"
    spec_schema: ClassVar[type[BaseModel] | None] = AtifConvertSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HelixJobSpec:
        del workspace, entity_client, job_name, async_sdk, options
        image, entrypoint = _tasks_image()
        return HelixJobSpec(
            steps=[
                HelixJobStepSpec(
                    name="convert",
                    executor=CPUExecutionProvider(
                        provider="cpu",
                        profile=profile or "default",
                        container=ContainerSpec(
                            image=image,
                            entrypoint=[*entrypoint, "-m"],
                            command=["nemo_customizer.tasks.atif_convert"],
                        ),
                    ),
                    config=spec.model_dump(mode="json"),
                    environment=[
                        HelixJobEnvironmentVariable(
                            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
                            value=DEFAULT_JOB_STORAGE_PATH,
                        )
                    ],
                )
            ]
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient | None = None) -> dict[str, Any]:
        del ctx
        spec = AtifConvertSpec.model_validate(config)
        if not spec.trajectory_filesets:
            raise RuntimeError("atif-convert needs the evaluation output filesets")
        if sdk is None:
            raise RuntimeError("atif-convert requires a platform client to upload the workflow fileset")
        prefix = fileset_prefix(spec.node_key)
        files = FilesClient.from_client(sdk)
        with tempfile.TemporaryDirectory(prefix="atif-convert-") as scratch:
            counter = None
            if spec.budget is not None and spec.student_model:
                # Before any trajectory is read, so a missing tokenizer fails the job in seconds.
                tokenizer_dir = Path(scratch) / "tokenizer"
                tokenizer_dir.mkdir()
                counter = load_student_counter(sdk, spec.student_model, spec.workspace, tokenizer_dir)
            result = convert_filesets(FilesClientSource(files, spec.workspace), spec, Path(scratch), counter)
            for name, local in (("train.jsonl", result.train_path), ("validation.jsonl", result.validation_path)):
                files.upload_file(
                    workspace=spec.workspace, name=spec.fileset, path=f"{prefix}/{name}", content=_chunks(local)
                )
        # Last, because the workflow controller treats output.json as proof the dataset is complete.
        files.upload_file(
            workspace=spec.workspace,
            name=spec.fileset,
            path=f"{prefix}/output.json",
            content=json.dumps(result.output).encode("utf-8"),
        )
        logger.info(
            "atif-convert: %d train and %d validation rows from %d runs (skipped %s)",
            result.output["examples"],
            result.output["validation_examples"],
            result.output["runs"],
            result.output["skipped"],
        )
        return {"status": "completed", "dataset": result.output["dataset"]}
