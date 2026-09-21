# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect Gym rollouts from a sandboxed Gym host instead of a local ``gym`` CLI.

:class:`~nemo_evaluator_sdk.agent_eval.runtimes.gym.runtime.GymAgentTaskRunner` runs Gym as a
subprocess tree in this process's own environment: it needs the ``gym`` executable on PATH and a
Gym checkout as its working directory, and whatever the environment's ``resources_server`` code
does, it does with this process's credentials. That is fine for a vetted environment on trusted
hardware and unacceptable once the environment arrives as a user-supplied FileSet, which is the
case the sandboxed-GRPO RFC exists for (§2.3).

This runner keeps everything about the evaluation the same and changes only where the rollouts are
produced. The dataset is materialized identically, the rollout records are read back through the
same parser, and the same coverage check runs -- so a run here and a run there are comparable, and
a scoring bug cannot hide in one path but not the other. What differs is one step: instead of
driving ``gym env start`` and ``gym eval run``, it POSTs the rows as ``examples`` to a sandboxed
host's ``/rollouts/run`` and writes the returned records where the parser expects them.

Attribution survives that hop because the host copies each example's ``_ng_task_index`` onto its
result. Nothing here joins by position.

Gym's model-call captures make the same hop: the host enables capture, reads each rollout's file,
and returns it on the record. This runner writes them back out in the layout the CLI runner's parser
expects, so per-call timing reaches a trace either way.

The host itself is provisioned by ``nemo-sandboxed-gym``: start a session, take its rollout URL and
token off the descriptor, and hand them to this runner.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import DEFAULT_REWARD_KEY
from nemo_evaluator_sdk.agent_eval.runtimes.gym.dataset import materialize_dataset, source_datasets
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nemo_evaluator_sdk.agent_eval.runtimes.gym.results import (
    aggregate_scores_from_gym,
    capture_filename,
    ensure_fresh_output,
    read_run_aggregations,
    require_full_coverage,
    trials_from_rollouts,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, RunnerInfo
from nemo_evaluator_sdk.values.results import AggregateScore
from pydantic import BaseModel, ConfigDict, Field
from sandboxed_gym.host.models import MIN_PROXY_CUTOFF_S

logger = logging.getLogger(__name__)

#: Auth header the orchestrator proxy expects. Mirrors ``sandboxed_gym.wire.PROXY_AUTH_HEADER``,
#: duplicated rather than imported so this module carries no dependency on that package: a caller
#: who has a session descriptor already has the token, and the header name is part of the contract.
PROXY_AUTH_HEADER = "X-Sandboxed-Gym-Token"


#: Key the sandboxed host attaches a rollout's captured model calls under. Mirrors
#: ``sandboxed_gym.runtime.gym_host_runtime.MODEL_CALLS_RESULT_KEY``, duplicated rather than
#: imported so this runner carries no dependency on that package -- a caller with a session
#: descriptor never needs it installed.
MODEL_CALLS_RESULT_KEY = "_nmp_model_calls"
#: Where captures are written locally, matching the CLI runtime's ``model_call_capture_dir``.
_CAPTURE_SUBDIR = "model_calls"


def _stamp_rollout_indices(examples: list[dict[str, Any]]) -> None:
    """Number each example within its task, as Gym's CLI preprocessing does.

    ``run_examples`` does not assign ``_ng_rollout_index`` -- only ``gym eval run`` does, while
    preprocessing the dataset. Without it Gym's ``maybe_rollout_id_from_run_body`` returns None, so
    it writes **no model-call capture at all**, and the trial falls back to a synthesized id.
    Confirmed by running a real sandboxed rollout: the capture directory was created and stayed
    empty until the index was supplied.

    Only where the row is silent, so a caller that numbers its own repeats keeps its numbering.
    """
    seen: dict[Any, int] = {}
    for example in examples:
        if example.get(NG_ROLLOUT_INDEX) is not None:
            continue
        task = example.get(NG_TASK_INDEX)
        example[NG_ROLLOUT_INDEX] = seen[task] = seen.get(task, -1) + 1


def _unpack_model_call_captures(records: list[dict[str, Any]], work_dir: Path) -> Path | None:
    """Write each record's captured model calls to disk, and return the directory holding them.

    The captures cross the wire attached to their rollout, but the parser reads them from a
    directory, so they are unpacked here. Deliberately on-disk rather than passed in memory: one
    code path reads a capture whichever runner produced it, so a parsing bug cannot hide in the
    sandboxed runner and not the CLI one.

    Mutates each record to drop the transport key, so the ``rollouts.jsonl`` this runner writes is
    the shape Gym itself would have written.

    Returns None when no record carried a capture -- an older host, or a run where none was
    written. Passing a directory that will never contain anything would make the parser warn once
    per trial about a capture nobody asked for.
    """
    capture_dir = work_dir / _CAPTURE_SUBDIR
    written = 0
    for record in records:
        calls = record.pop(MODEL_CALLS_RESULT_KEY, None)
        name = capture_filename(record) if isinstance(calls, list) and calls else None
        if name is None:
            continue
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / name).write_text(
            "".join(f"{json.dumps(call)}\n" for call in calls),
            encoding="utf-8",
        )
        written += 1
    if not written:
        logger.info("Sandboxed Gym host returned no model-call captures; traces will carry no per-call timing.")
        return None
    return capture_dir


class SandboxedGymRuntimeConfig(BaseModel):
    """Where to reach a running sandboxed Gym host, and how to read its rollouts."""

    model_config = ConfigDict(extra="forbid")

    rollout_url: str = Field(description="The session's `/rollouts/run` URL, from its descriptor.")
    auth_token: str | None = Field(
        default=None,
        description="Bearer token for the orchestrator proxy (`rollout_auth_token` on the descriptor).",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers the host requires, e.g. the OpenSandbox proxy headers on its endpoint.",
    )
    timeout_s: float = Field(
        default=3600.0,
        gt=0,
        description="Ceiling on one rollout collection. Covers every example in the run, not each one.",
    )
    agent_ref_name: str | None = Field(
        default=None,
        description="Gym agent instance each example is routed to, stamped as `agent_ref` on rows that do "
        "not already carry one. Required by the host path: `RolloutCollectionHelper.run_examples` reads "
        "`row['agent_ref']['name']` with no fallback, while the `gym eval run` CLI resolves the agent from "
        "config instead -- so a dataset that runs under the CLI can arrive here unroutable. This is the "
        "instance an environment's config defines (`mcqa_simple_agent`), not the agent component "
        "(`simple_agent`).",
    )
    reward_key: str = Field(default=DEFAULT_REWARD_KEY, description="Key read from each rollout record.")


class SandboxedGymAgentTaskRunner:
    """An ``AgentTaskRunner`` that collects rollouts from a sandboxed Gym host over HTTP."""

    def __init__(self, *, config: SandboxedGymRuntimeConfig) -> None:
        self._config = config
        self._run_aggregations: dict[str, Any] | None = None

    def run_aggregate_scores(self) -> Sequence[AggregateScore]:
        """Gym's ``agent_metrics`` as typed aggregate scores, namespaced ``runner.gym.<metric>``.

        Empty in practice today: those numbers come from a sidecar the CLI writes, and a host that
        returns rollout records writes no such file. Implemented anyway so this runner satisfies the
        same protocol as the CLI one and starts reporting if the host grows the sidecar.
        """
        return aggregate_scores_from_gym(self._run_aggregations)

    def runner_info(self) -> RunnerInfo:
        """Identify the runner and the host it collected from.

        The token is omitted rather than redacted: unlike Gym's free-form ``hydra_params``, there is
        exactly one credential here and nothing about it is worth recording.
        """
        cfg = self._config
        return RunnerInfo(
            name="gym",
            kind="runner",
            config={
                "mode": "sandboxed",
                "rollout_url": cfg.rollout_url,
                "agent_ref_name": cfg.agent_ref_name,
                "reward_key": cfg.reward_key,
                "timeout_s": cfg.timeout_s,
            },
        )

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self._config.headers)
        if self._config.auth_token is not None:
            headers[PROXY_AUTH_HEADER] = self._config.auth_token
        return headers

    def _decode_body(self, response: httpx.Response, elapsed: float | None = None) -> Any:
        """Decode a 2xx rollout body, naming the host when there is nothing decodable in it.

        The host commits its 200 before the batch finishes and pads the open connection with
        whitespace heartbeats until it has an envelope to write. A host that dies mid-batch --
        OOMKilled, evicted -- therefore leaves a well-formed 200 whose body is heartbeats and
        nothing else. Left to ``response.json()`` that surfaces as a bare ``JSONDecodeError``,
        which is a ``ValueError`` and so escapes callers guarding this runner for ``RuntimeError``:
        a dead sandbox reads as a bug in the evaluator. ``RolloutOrchestrator._decode_results``
        classifies the same three bodies for the other client of this endpoint.
        """
        try:
            # Strict, and caught rather than avoided: errors="replace" would let a body with one
            # corrupt byte still parse, handing the caller U+FFFD where the host wrote data.
            text = response.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"sandboxed Gym host returned a body that is not UTF-8 from {self._config.rollout_url}: {exc}"
            ) from exc
        if not text:
            if elapsed is not None and elapsed < MIN_PROXY_CUTOFF_S:
                raise RuntimeError(
                    f"sandboxed Gym host at {self._config.rollout_url} answered and then sent "
                    f"nothing in {elapsed:.1f}s -- too fast to have been cut in transit, so it "
                    f"died before it could write; check whether the sandbox was OOMKilled or "
                    f"evicted"
                )
            raise RuntimeError(
                f"sandboxed Gym host at {self._config.rollout_url} sent no body at all"
                + (f" in {elapsed:.1f}s" if elapsed is not None else "")
                + " -- nothing was ever written. Either it died before writing anything (check the "
                "sandbox for an OOMKill or an eviction), or its image predates the rollout "
                "heartbeat and the proxy cut the silent request at its own cap. Without a "
                "Content-Length the body ends at the close, so this client cannot tell them apart"
            )
        if not text.strip():
            raise RuntimeError(
                f"sandboxed Gym host at {self._config.rollout_url} answered and then stopped "
                f"without sending a `results` envelope ({len(response.content)} byte(s) of "
                f"heartbeat, then silence); it most likely died mid-batch -- check whether the "
                f"sandbox was OOMKilled or evicted"
            )
        try:
            # The host heartbeats leading whitespace while a batch runs; json tolerates it.
            return json.loads(text)
        except (ValueError, RecursionError) as exc:
            # Wider than JSONDecodeError because json also refuses input outright -- the
            # integer-digit limit, deep nesting -- and narrowing this back lets those escape as
            # something other than a named host failure.
            raise RuntimeError(
                f"sandboxed Gym host returned a body that is not JSON from "
                f"{self._config.rollout_url} ({exc}); first 200 bytes: {text[:200]!r}"
            ) from exc

    async def _collect(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """POST the examples and return the host's rollout records."""
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self._config.timeout_s) as client:
            response = await client.post(
                self._config.rollout_url,
                json={"examples": examples},
                headers=self._request_headers(),
            )
        elapsed = time.monotonic() - started
        if response.status_code >= 400:
            # The body is the host's own error envelope; it names which example or server failed,
            # which the status code alone does not.
            raise RuntimeError(
                f"sandboxed Gym host returned {response.status_code} from {self._config.rollout_url}: "
                f"{response.text[:2000]}"
            )
        body = self._decode_body(response, elapsed)
        error = body.get("error") if isinstance(body, Mapping) else None
        if error is not None:
            # The host commits its 200 before the batch finishes, so that it can hold the
            # connection open past the sandbox proxy's first-byte cap. A failure after that point
            # has only the body left to travel in, and carries the code and traceback that say
            # which of Gym's layers raised.
            raise RuntimeError(
                f"sandboxed Gym host reported an error from {self._config.rollout_url}: "
                f"{error if isinstance(error, str) else json.dumps(error)[:2000]}"
            )
        results = body.get("results") if isinstance(body, Mapping) else None
        if not isinstance(results, list):
            raise RuntimeError(
                f"sandboxed Gym host returned no `results` list from {self._config.rollout_url}; "
                f"got {type(results).__name__}"
            )
        return [record for record in results if isinstance(record, dict)]

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        self._run_aggregations = None  # reset per run so a reused runner never leaks prior numbers
        cfg = self._config

        if config is not None and config.work_dir is not None:
            work_dir = Path(config.work_dir) / "gym_run"
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="gym_sandboxed_run_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        rollouts_path = work_dir / "rollouts.jsonl"
        ensure_fresh_output(rollouts_path)

        # Same materialization the CLI runner uses, so the rows the host sees are the rows Gym
        # would have read, and `_ng_task_index` is stamped by the same code.
        input_path = work_dir / "gym_input.jsonl"
        index_to_task_id = materialize_dataset(tasks, input_path)
        examples = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if cfg.agent_ref_name:
            # Only where the row is silent: a dataset that names its own agent per row keeps doing so,
            # which is how multi-agent Gym datasets are meant to work.
            for example in examples:
                example.setdefault("agent_ref", {"name": cfg.agent_ref_name})
        _stamp_rollout_indices(examples)
        logger.info(
            "Collecting %d example(s) from %s via sandboxed Gym host %s.",
            len(examples),
            source_datasets(tasks),
            cfg.rollout_url,
        )

        records = await self._collect(examples)

        # Before the records are written: unpacking strips the transport key, and `rollouts.jsonl`
        # has to be the shape Gym itself would have written for the shared parser to read it.
        capture_dir = _unpack_model_call_captures(records, work_dir)

        # Written where the parser expects it, so the records are read by exactly the code that
        # reads the CLI runner's output -- including its handling of records with no usable index.
        rollouts_path.write_text(
            "".join(f"{json.dumps(record)}\n" for record in records),
            encoding="utf-8",
        )

        self._run_aggregations = read_run_aggregations(rollouts_path)
        trials = trials_from_rollouts(
            rollouts_path, tasks, index_to_task_id, reward_key=cfg.reward_key, capture_dir=capture_dir
        )
        require_full_coverage(tasks, covered_task_ids={trial.task_id for trial in trials}, rollouts_path=rollouts_path)
        return trials
