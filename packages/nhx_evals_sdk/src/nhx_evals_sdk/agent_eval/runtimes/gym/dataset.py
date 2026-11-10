# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym rows to Evaluator tasks, and back to a dataset Gym will collect against.

Task identity is the content hash of a row (or of its key, with ``id_field``), and attribution is an index this module *assigns*
rather than infers — see :func:`materialize_dataset`. Both halves live together because they are
two directions of one translation: what breaks one silently breaks the other.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nhx_evals_sdk.agent_eval.runtimes.gym.records import (
    NG_TASK_INDEX,
    RUNTIME_KEYS,
    iter_jsonl_objects,
)
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.metrics.protocol import Metric

logger = logging.getLogger(__name__)


def _canonical_row_hash(row: Mapping[str, Any]) -> str:
    """Stable ``sha256`` of a Gym dataset row, excluding runtime-injected fields.

    A given row keeps its id across dataset revisions/reorderings (so Intake
    rollups stay consistent), and a changed row becomes a new task. No
    dataset-revision component — identity is the row content alone.
    """
    payload = {key: value for key, value in row.items() if key not in RUNTIME_KEYS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


#: Hash-mode task ids are ``gym-`` plus this many hex digits of the row hash (236 bits): the longest
#: id that is still a valid platform entity name (a lowercase letter first, at most 63 characters), so
#: the id a row gets locally is the name it is stored under and the test case id Intake records.
_HASH_ID_HEX_DIGITS = 59


def _hash_task_id(row: Mapping[str, Any]) -> str:
    """A row's content-derived task id: ``gym-`` and a prefix of :func:`_canonical_row_hash`."""
    return f"gym-{_canonical_row_hash(row)[:_HASH_ID_HEX_DIGITS]}"


#: Longest platform entity name, which a keyed task id fills: ``<prefix>-`` plus as many hex digits of
#: the key's hash as fit.
_MAX_TASK_ID_LENGTH = 63
#: A key prefix: a lowercase letter, then up to 7 lowercase letters or digits. No dashes, so the id can
#: never contain ``--`` or end in ``-``, both of which the platform's entity names refuse.
ID_PREFIX_PATTERN = r"^[a-z][a-z0-9]{0,7}$"


def check_id_prefix(id_prefix: str) -> None:
    """Raise ``ValueError`` unless ``id_prefix`` is a valid keyed-id prefix (:data:`ID_PREFIX_PATTERN`)."""
    if not re.fullmatch(ID_PREFIX_PATTERN, id_prefix):
        raise ValueError(
            f"id_prefix {id_prefix!r} must be 1-8 characters: a lowercase letter, then lowercase letters or digits"
        )


def _keyed_task_id(value: Any, id_prefix: str) -> str:
    """A keyed row's task id: ``<id_prefix>-`` and the hash of the key's JSON encoding, filling 63 characters.

    Only the key is hashed, so editing any other field of the row keeps its id. The key is hashed in
    its JSON form, so ``1`` and ``"1"`` are different keys.
    """
    digest = hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()
    return f"{id_prefix}-{digest[: _MAX_TASK_ID_LENGTH - len(id_prefix) - 1]}"


def _content_text(content: Any) -> str:
    """Flatten a message ``content`` (a string, or a list of ``{type,text}`` parts) to text.

    Anything else contributes nothing. A missing ``content`` is unremarkable, but a *populated* value
    in some other shape means we're silently dropping prompt text, so warn — :func:`_render_instruction`
    only fails loudly when the whole row renders empty, which a single skipped message wouldn't trigger.
    """
    if isinstance(content, str):
        return content
    # `str` is itself a Sequence, so this branch only ever sees non-str sequences (handled above).
    if isinstance(content, Sequence):
        parts = [part["text"] for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
        return "\n".join(parts)
    if content is not None:
        logger.warning(
            "Ignoring message content of unsupported type %s (expected a string or a list of {type,text} parts); "
            "any prompt text it carries will be missing from the task instruction.",
            type(content).__name__,
        )
    return ""


def _render_instruction(responses_create_params: Mapping[str, Any]) -> str:
    """Derive a task instruction from a row's ``responses_create_params``, or ``""`` when it carries none.

    Uses ``instructions`` (system prompt, when present) + ``input`` (a plain string
    or an OpenAI message list).

    An empty result is *not* an error. A whole class of Gym environments ships
    ``"responses_create_params": {"input": []}`` and keeps the task elsewhere — ``gdpval``
    in a top-level ``prompt``, ``legal_agent_bench`` in an ``instance_id`` its own agent
    resolves, ``aviary``/``toolsandbox`` in nothing but a ``task_idx``. Their prompt is
    materialized by Gym's data-preparation step or by the environment's agent, neither of
    which this runner drives (``--no-serve --input`` hands Gym a ready-to-run dataset by
    design). Nothing here can reconstruct it, and no single row field generalizes across
    them, so the honest answer is no instruction rather than a guess. See
    :func:`discover_gym_tasks` for what that means downstream.
    """
    parts: list[str] = []
    instructions = responses_create_params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        parts.append(instructions.strip())
    raw_input = responses_create_params.get("input")
    if isinstance(raw_input, str):
        parts.append(raw_input)
    elif isinstance(raw_input, Sequence):
        for item in raw_input:
            if isinstance(item, Mapping):
                text = _content_text(item.get("content"))
                if text:
                    parts.append(text)
    return "\n\n".join(part for part in parts if part).strip()


def _default_gym_metric() -> Metric:
    """The default reward metric, imported lazily (see ``metrics.runner_rewards``)."""
    from nhx_evals_sdk.metrics.runner_rewards import GymRewardMetric

    return GymRewardMetric()


def discover_gym_tasks(
    dataset: str | Path | Iterable[bytes],
    *,
    metrics: Sequence[Any] | None = None,
    name: str | None = None,
    max_rows: int | None = None,
    id_field: str | None = None,
    id_prefix: str | None = None,
) -> list[AgentEvalTask]:
    """Build one :class:`AgentEvalTask` per distinct row in a Gym dataset (jsonl).

    ``dataset`` is a path, or an iterable of byte lines (``io.BytesIO``) that ``name`` labels in
    errors and logs. ``max_rows``, when set, refuses a dataset with more non-blank rows, stopping
    at the first row past the limit.

    **Task ids.** A task's id is ``gym-`` and 59 hex digits of its row's content hash, a valid
    platform entity name, so the same row has the same id locally, inline and stored. An edited row
    is therefore a new task.

    With ``id_field`` and ``id_prefix`` (always together), a task's id instead comes from the row's
    value of ``id_field``: ``<id_prefix>-`` and the hash of that value, 63 characters in all. Only
    the key is hashed, so an edited row keeps its id. Every row must carry a non-empty string or
    number in ``id_field``, and two rows with the same key are an error naming both lines.
    ``id_prefix`` keeps two files' keys apart in one workspace: see :data:`ID_PREFIX_PATTERN`.

    Every task's ``intent`` is the fixed label ``"Gym row"``: it never names the file, so the same
    row yields the same task content from any file. Each task's ``instruction`` is rendered
    from ``responses_create_params.input`` *when the row carries one*; the raw params
    are stashed under ``inputs['gym_row']`` for provenance; and it is scored by a
    :class:`GymRewardMetric`. A path dataset is stamped on
    ``metadata['gym_dataset_path']``, and every *other* row key (the verifier's
    ground-truth fields, ``agent_ref``, and so on) on ``metadata['gym_row_extras']``.
    Together with ``inputs['gym_row']`` those reconstruct the complete source row,
    which :class:`GymAgentTaskRunner` re-materializes into the dataset it hands to Gym.

    **One distinct row is one task.** Duplicate rows collapse (identity is row content,
    so they are by definition the same task) and are reported as a warning: repeated
    attempts are a run-level concern — ``GymRuntimeConfig.num_repeats`` — not something
    a dataset expresses by repeating a row, so duplicates almost always mean bad data.

    **A row need not carry a prompt.** A sweep of the 106 built-in environments shipping example
    data found 5 — ``aviary``, ``gdpval``, ``legal_agent_bench``, ``scicode`` and ``toolsandbox``
    — whose every row ships ``responses_create_params.input == []``, letting Gym's data-prep step
    or the environment's own agent materialize the prompt instead (see
    :func:`_render_instruction`). Such a row yields a task with **no** ``inputs['instruction']``
    key rather than a fabricated one, and is counted in a summary log line. This costs the
    Gym path nothing: the runner never reads ``instruction`` — it re-materializes the source
    row from ``inputs['gym_row']`` + ``metadata['gym_row_extras']`` and hands *that* to Gym,
    and ``intent`` is a dataset label either way. It stays absent (not ``""``) so that
    :meth:`AgentEvalTask.agent_prompt` still fails loudly for the runners that *do* need a
    prompt — an instruction-less task must not reach an agent as an empty one.

    ``responses_create_params`` itself remains required: Gym indexes into it unconditionally
    (``rollout_collection._preprocess_rows_from_config``), so a row without it is rejected here
    rather than crashing Gym mid-collection.
    """
    if (id_field is None) != (id_prefix is None):
        raise ValueError("id_field and id_prefix go together: set both, or neither")
    if id_prefix is not None:
        check_id_prefix(id_prefix)
    path = Path(dataset) if isinstance(dataset, (str, Path)) else None
    label = name or (str(path) if path is not None else "<stream>")
    tasks: list[AgentEvalTask] = []
    seen: set[str] = set()
    key_lines: dict[str, int] = {}
    rows = duplicates = promptless = 0
    for line_no, row in iter_jsonl_objects(path if path is not None else dataset, name=label):
        rows += 1
        if max_rows is not None and rows > max_rows:
            raise ValueError(f"{label}:{line_no}: more than the {max_rows}-row limit")
        params = row.get("responses_create_params")
        if not isinstance(params, Mapping):
            raise ValueError(
                f"{label}:{line_no}: row has no 'responses_create_params' mapping; Gym requires that key "
                "on every dataset row"
            )
        if id_field is not None and id_prefix is not None:
            key = row.get(id_field)
            if isinstance(key, bool) or not isinstance(key, (str, int, float)) or key == "":
                raise ValueError(
                    f"{label}:{line_no}: row has no usable {id_field!r} value (need a non-empty string or number, "
                    f"got {key!r})"
                )
            task_id = _keyed_task_id(key, id_prefix)
            if task_id in key_lines:
                raise ValueError(
                    f"{label}:{line_no}: {id_field} {key!r} repeats line {key_lines[task_id]}; each row needs its own key"
                )
            key_lines[task_id] = line_no
        else:
            task_id = _hash_task_id(row)
        if task_id in seen:
            duplicates += 1
            continue
        seen.add(task_id)
        instruction = _render_instruction(params)
        if not instruction:
            promptless += 1
        tasks.append(
            AgentEvalTask(
                id=task_id,
                # Deliberately not the file name: the same row must be the same task content whichever
                # file it came from, so stored tasks are shared across tasksets and runs compare.
                intent="Gym row",
                inputs={
                    # Absent, not empty, when the row carries no prompt — see the docstring.
                    **({"instruction": instruction} if instruction else {}),
                    "gym_row": params,
                },
                metrics=list(metrics) if metrics is not None else [_default_gym_metric()],
                metadata={
                    **({"gym_dataset_path": str(path)} if path is not None else {}),
                    # Everything except responses_create_params, which already lives in inputs['gym_row'].
                    "gym_row_extras": {
                        key: value
                        for key, value in row.items()
                        if key != "responses_create_params" and key not in RUNTIME_KEYS
                    },
                },
            )
        )
    if duplicates:
        logger.warning(
            "Collapsed %d duplicate row(s) in %s into %d distinct task(s): task identity is row content, so "
            "repeating a row cannot request repeated attempts — set GymRuntimeConfig.num_repeats for that. "
            "Duplicate rows usually indicate a data problem.",
            duplicates,
            label,
            len(tasks),
        )
    if promptless:
        logger.info(
            "%d of %d task(s) in %s carry no prompt in responses_create_params and were discovered without an "
            "inputs['instruction']. Expected for environments whose prompt is built by Gym's data-prep step or by "
            "their own agent (e.g. aviary, gdpval, legal_agent_bench, scicode, toolsandbox); the Gym runner does "
            "not read the instruction. A runner that hands the task straight to an agent will reject these tasks.",
            promptless,
            len(tasks),
            label,
        )
    if not tasks:
        raise ValueError(f"no rows found in Gym dataset {label}")
    return tasks


def source_datasets(tasks: Sequence[AgentEvalTask]) -> str:
    """Human-readable summary of the dataset(s) the tasks were discovered from, for a log line.

    Purely provenance. Gym reads the dataset :func:`materialize_dataset` writes from the tasks
    themselves, so nothing here gates the run: tasks drawn from two datasets are legitimate (the
    materialized file is their union), and a task with no stamped path runs fine. Never raises —
    a missing or inconsistent label is not a reason to fail an otherwise valid evaluation.
    """
    stamped = {
        task.metadata["gym_dataset_path"]
        for task in tasks
        if isinstance(task.metadata.get("gym_dataset_path"), str) and task.metadata["gym_dataset_path"]
    }
    return ", ".join(sorted(stamped)) if stamped else "<tasks with no stamped dataset path>"


def gym_task_row(
    *, task_id: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate and return the source parameters and extras required to execute a Gym task."""
    extras = metadata.get("gym_row_extras")
    params = inputs.get("gym_row")
    if not isinstance(extras, Mapping) or not isinstance(params, Mapping):
        raise ValueError(
            f"task {task_id!r} is missing inputs['gym_row'] and/or metadata['gym_row_extras']; build tasks "
            "with discover_gym_tasks so the Gym dataset can be re-materialized for the run"
        )
    return params, extras


def validate_gym_task_row(*, task_id: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Validate a task's Gym row without returning its materialization inputs.

    Args:
        task_id: Task identifier used in validation errors.
        inputs: Task inputs containing the source Gym row.
        metadata: Task metadata containing the source row's extra fields.

    Returns:
        None when the row and extras are mappings.

    Raises:
        ValueError: The row or extras are missing or are not mappings.
    """
    gym_task_row(task_id=task_id, inputs=inputs, metadata=metadata)


def materialize_dataset(tasks: Sequence[AgentEvalTask], dest: Path) -> dict[int, str]:
    """Write the normalized dataset Gym will read, and return its ``_ng_task_index`` → task-id map.

    Algorithm:
        - Reassemble each source row from task inputs and metadata in task order.
        - Remove source ``_ng_*`` fields and stamp the authoritative task index.
        - Write one JSON object per task and return the index-to-task-ID join map.

    The explicit index makes rollout matching order-independent and confines Gym to the requested
    tasks. Gym assigns the separate rollout index for repeated attempts.
    """
    index_to_task_id: dict[int, str] = {}
    seen_task_ids: set[str] = set()
    lines: list[str] = []
    for index, task in enumerate(tasks):
        # The source row is split across inputs and metadata so the run bundle doesn't persist
        # responses_create_params twice; reassemble it here.
        params, extras = gym_task_row(task_id=task.id, inputs=task.inputs, metadata=task.metadata)
        if task.id in seen_task_ids:
            raise ValueError(
                f"task {task.id!r} was supplied more than once; one distinct row is one task, and repeated "
                "attempts come from GymRuntimeConfig.num_repeats"
            )
        seen_task_ids.add(task.id)
        payload = {key: value for key, value in extras.items() if key not in RUNTIME_KEYS}
        payload["responses_create_params"] = params
        payload[NG_TASK_INDEX] = index
        lines.append(json.dumps(payload, default=str))
        index_to_task_id[index] = task.id
    if not index_to_task_id:
        raise ValueError("GymAgentTaskRunner was given no tasks to run")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_to_task_id
