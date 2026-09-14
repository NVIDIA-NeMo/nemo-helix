# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runner itself: the two-step Gym invocation, start to scored trials.

Orchestration only. The stages it drives live in sibling modules — :mod:`config` for what Gym is
told, :mod:`dataset` for what it is given, :mod:`process` for how it is run, and :mod:`results`
for what comes back.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import (
    HYDRA_SUBDIR,
    GymRuntimeConfig,
    hydra_scalar,
    model_call_capture_dir,
    selection_args,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.dataset import materialize_dataset, source_datasets
from nemo_evaluator_sdk.agent_eval.runtimes.gym.process import (
    LOG_TAIL_LINES,
    VALIDATE_TIMEOUT_S,
    drain_pumps,
    gym_executable,
    gym_invocation_env,
    pending_servers,
    pump_stream,
    terminate,
)
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import ENV_LOG_NAME
from nemo_evaluator_sdk.agent_eval.runtimes.gym.results import (
    aggregate_scores_from_gym,
    ensure_fresh_output,
    read_run_aggregations,
    require_full_coverage,
    trials_from_rollouts,
)
from nemo_evaluator_sdk.agent_eval.runtimes.provenance import redact_credentials
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, RunnerInfo
from nemo_evaluator_sdk.values.results import AggregateScore

logger = logging.getLogger(__name__)


class GymAgentTaskRunner:
    """An :class:`AgentTaskRunner` that runs an existing Gym env, then adapts its rollouts.

    :meth:`run_tasks` materializes the requested tasks into a normalized dataset with
    ``_ng_task_index`` stamped per row, runs the two-step Gym flow (``env start`` +
    ``eval run --no-serve --input``) against it via subprocess, and fans the rollout
    records out into one trial per ``_ng_rollout_index``. Because we assign the index,
    attribution is a dict lookup rather than a positional guess, and Gym only rolls out
    the tasks we asked for. The durable task **identity** stays our content hash.
    """

    def __init__(self, *, config: GymRuntimeConfig) -> None:
        self._config = config
        self._run_aggregations: dict[str, Any] | None = None

    @property
    def config(self) -> GymRuntimeConfig:
        """The settings this runner was constructed with.

        Read-only, and the whole config rather than a property per field: unlike the Fabric
        runtimes, everything shaping a Gym run already lives in one validated object.

        Exposed so a live runner can be described as the job-spec target that reproduces it, without
        reaching into a private attribute from another package. ``runner_info()`` cannot serve that
        purpose — it redacts credential-shaped values, so what it returns is provenance to read, not
        configuration to rebuild from.
        """
        return self._config

    def run_aggregate_scores(self) -> Sequence[AggregateScore]:
        """Gym's ``agent_metrics`` mapped onto typed aggregate scores, namespaced ``runner.gym.<metric>``.

        Satisfies :class:`RunAggregationsProvider`. ``reward`` is skipped: the SDK already scores it
        natively as ``gym_reward.reward``, and two differently-derived numbers under one name invites
        exactly the confusion the namespace is there to prevent.
        """
        return aggregate_scores_from_gym(self._run_aggregations)

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the Gym settings that shape its results.

        Credentials normally live in the Gym checkout's gitignored ``env.yaml`` and never reach this
        object — but ``hydra_params`` and ``env_vars`` are free-form escape hatches, so their values
        are redacted by key (see :func:`redact_credentials`) rather than trusted. ``env_vars``
        needs it at least as much: environment variables are the conventional way to pass an API
        key, so a caller doing the obvious thing would otherwise write one into the run bundle.
        """
        cfg = self._config
        return RunnerInfo(
            name="gym",
            kind="runner",
            config={
                "resources_server": cfg.resources_server,
                "agent": cfg.agent,
                "agent_config": cfg.agent_config,
                "model_type": cfg.model_type,
                "num_repeats": cfg.num_repeats,
                "concurrency": cfg.concurrency,
                "bind_resources_server": cfg.bind_resources_server,
                "hydra_params": redact_credentials(cfg.hydra_params),
                "env_vars": redact_credentials(cfg.env_vars),
                "reward_key": cfg.reward_key,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        cfg = self._config
        self._run_aggregations = None  # reset per run so a reused runner never leaks a prior run's numbers
        # Provenance for the log line only — the file Gym actually reads is the normalized one we
        # materialize below from the tasks themselves.
        source_dataset = source_datasets(tasks)

        # config.parallelism still governs how the evaluator *scores* the trials we return (its scoring
        # semaphore). It is deliberately not mapped onto Gym's rollout `--concurrency`: those are different
        # phases — parallelism bounds concurrent scoring (SDK-side, cheap), while Gym's `--concurrency`
        # bounds concurrent rollouts against the model endpoint during collection (tuned to that endpoint's
        # limits via GymRuntimeConfig.concurrency).
        if config is not None and config.work_dir is not None:
            work_dir = Path(config.work_dir) / "gym_run"
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="gym_run_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        rollouts_path = work_dir / "rollouts.jsonl"
        ensure_fresh_output(rollouts_path)

        # Hand Gym a dataset we control: one row per requested task, each carrying the _ng_task_index
        # we assign. Gym honors a pre-stamped index, so the rollout->task join becomes a total map
        # instead of a positional guess about Gym's raw-line dedup (see the module docstring).
        input_path = work_dir / "gym_input.jsonl"
        index_to_task_id = materialize_dataset(tasks, input_path)
        logger.info(
            "Materialized %d task(s) from %s into %s for Gym collection.",
            len(index_to_task_id),
            source_dataset,
            input_path,
        )

        await self._run_two_step(input_path, rollouts_path, work_dir)
        self._run_aggregations = read_run_aggregations(rollouts_path)
        trials = trials_from_rollouts(
            rollouts_path,
            tasks,
            index_to_task_id,
            reward_key=cfg.reward_key,
            capture_dir=model_call_capture_dir(work_dir),
        )
        require_full_coverage(tasks, covered_task_ids={trial.task_id for trial in trials}, rollouts_path=rollouts_path)
        return trials

    async def _validate_config(
        self, gym: str, selection: Sequence[str], subprocess_env: Mapping[str, str], work_dir: Path
    ) -> None:
        """Pre-flight the composed Gym config with ``gym env validate`` before starting anything.

        Gym does not publish what configuration an environment requires — the typed
        ``*ResourcesServerConfig`` classes cover behavioural knobs, while the model wiring lives in
        each environment's YAML under names that vary per environment. ``gym env validate`` is the
        only way to find out short of running: it merges configs, flags, and overrides, then reports
        unresolved ``???`` values, bad paths, and cross-references to servers that are not defined —
        without Ray and without starting a server.

        Running it every time is worth the second it costs. A config mistake otherwise surfaces after
        ``gym env start`` has brought up a Ray cluster and several uvicorn servers, as a readiness
        timeout up to ``startup_timeout_s`` (240s by default) whose message says nothing about the
        actual problem.
        """
        proc = await asyncio.create_subprocess_exec(
            gym,
            "env",
            "validate",
            *selection,
            env=dict(subprocess_env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Its own process group, like the other two Gym invocations. `terminate` signals the
            # group via `killpg`, so without this a validate timeout would signal *our* group — the
            # SDK process included.
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=VALIDATE_TIMEOUT_S)
        except TimeoutError:
            await terminate(proc, grace_s=self._config.shutdown_grace_s)
            raise RuntimeError(
                f"`gym env validate` did not finish within {VALIDATE_TIMEOUT_S}s for resources-server "
                f"{self._config.resources_server!r}"
            ) from None

        report = stdout.decode("utf-8", errors="replace").strip()
        (work_dir / "gym_validate.log").write_text(report, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Gym rejected the composed config for resources-server {self._config.resources_server!r} "
                f"before any server started. `gym env validate` said:\n\n{report}"
            )
        logger.debug("gym env validate: %s", report)

    async def _run_two_step(self, input_path: Path, output_path: Path, work_dir: Path) -> None:
        """Start the Gym servers, collect against them with ``--no-serve``, then tear them down."""
        cfg = self._config
        gym = gym_executable()
        env_log = work_dir / ENV_LOG_NAME

        # Gym launches each server from its own subdir with its own .venv. Ray (>=2.56) otherwise
        # detects a `uv run` ancestor and tries to replicate that uv project onto its workers,
        # asserting the project pyproject.toml lives in the driver's cwd — which aborts startup. That
        # hook is wrong for Gym (servers manage their own deps), so disable it for the subprocesses.
        subprocess_env = gym_invocation_env(cfg)

        selection = selection_args(cfg, work_dir)

        await self._validate_config(gym, selection, subprocess_env, work_dir)

        env_cmd = [gym, "env", "start", *selection]
        # start_new_session=True puts `gym env start` in its own process group so teardown can signal
        # the *whole* Ray-cluster + uvicorn tree, not just the direct child (else they orphan).
        # stderr is merged into stdout here (unlike `eval run`, which splits them) because readiness
        # detection scans a single chronological transcript and Gym's readiness line is not guaranteed
        # to land on a particular stream.
        env_proc = await asyncio.create_subprocess_exec(
            *env_cmd,
            env=subprocess_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        env_pump = asyncio.create_task(pump_stream(env_proc.stdout, env_log, label="gym env start"))
        try:
            await self._wait_for_servers(env_log, env_proc)
            await self._collect_rollouts(gym, input_path, output_path, work_dir, subprocess_env)
        finally:
            await terminate(env_proc, grace_s=cfg.shutdown_grace_s)
            # Bounded: a leaked grandchild holding the inherited pipe must not wedge teardown.
            await drain_pumps([env_pump], grace_s=cfg.shutdown_grace_s, what="gym env start")

    async def _collect_rollouts(
        self, gym: str, input_path: Path, output_path: Path, work_dir: Path, subprocess_env: dict[str, str]
    ) -> None:
        """Run ``gym eval run --no-serve`` against the live servers, bounded and self-cleaning.

        Spawned in its own process group and always torn down (even on timeout/cancellation), so a hung
        collection can't wedge the run or leak Ray workers.

        Output is *streamed* to ``gym_eval.stdout.log`` / ``gym_eval.stderr.log`` in ``work_dir`` rather
        than buffered in memory (a long collection's transcript can be large), and mirrored to this
        module's logger at ``DEBUG`` so callers can surface it in their own terminal through ordinary
        ``logging`` configuration. Failures point at both files and inline the last few lines.
        """
        cfg = self._config
        eval_cmd = [
            gym,
            "eval",
            "run",
            "--no-serve",
            "--agent",
            cfg.agent,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--num-repeats",
            str(cfg.num_repeats),
            "--concurrency",
            str(cfg.concurrency),
            # `gym eval run` is a Hydra app too, and unlike validate/start it does not go through
            # selection_args — so without this it writes `outputs/<date>/<time>/` into the caller's
            # cwd on every single collection. Quoted for the same reason as there: a work dir
            # containing `,` or `[` is otherwise misread as sweep syntax.
            f"hydra.run.dir={hydra_scalar(str(work_dir / HYDRA_SUBDIR))}",
        ]
        stdout_log = work_dir / "gym_eval.stdout.log"
        stderr_log = work_dir / "gym_eval.stderr.log"
        eval_proc = await asyncio.create_subprocess_exec(
            *eval_cmd,
            env=subprocess_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        tails: dict[str, deque[str]] = {}
        pumps = [
            asyncio.create_task(pump_stream(eval_proc.stdout, stdout_log, label="gym eval", tails=tails, key="stdout")),
            asyncio.create_task(pump_stream(eval_proc.stderr, stderr_log, label="gym eval", tails=tails, key="stderr")),
        ]
        try:
            await asyncio.wait_for(eval_proc.wait(), timeout=cfg.collection_timeout_s)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"`gym eval run` exceeded collection_timeout_s={cfg.collection_timeout_s}s; collection aborted. "
                f"See {stdout_log} and {stderr_log}."
            ) from exc
        finally:
            # Terminate first so the pipes close, then drain (bounded) to flush whatever was buffered.
            await terminate(eval_proc, grace_s=cfg.shutdown_grace_s)
            await drain_pumps(pumps, grace_s=cfg.shutdown_grace_s, what="gym eval run")
        if eval_proc.returncode != 0:
            tail = "\n".join(tails.get("stderr") or tails.get("stdout") or [])
            # A collection failure is frequently *server*-side: the resources-server raises, the
            # agent surfaces it as a bare HTTP 500, and the traceback that explains it is only in
            # gym_env.log. Naming the two eval logs alone sends the reader to the one place the
            # cause is not. (Observed on wmt_translation: a 500 here, `PermissionError: /opt/Gym`
            # there.)
            env_log = work_dir / ENV_LOG_NAME
            raise RuntimeError(
                f"`gym eval run` failed (rc={eval_proc.returncode}). Collection output: {stdout_log} "
                f"and {stderr_log}\n"
                f"If the tail below is an HTTP error from a Gym server, the cause is server-side: "
                f"see {env_log}, whose lines are prefixed with the server that emitted them.\n"
                f"--- last {LOG_TAIL_LINES} line(s) ---\n{tail}"
            )

    async def _wait_for_servers(self, env_log: Path, env_proc: asyncio.subprocess.Process) -> None:
        """Wait until ``gym env start`` reports every composed server ready.

        Keys off Gym's own readiness line (``All N / N servers ready!``), which it prints only after
        HTTP-health-checking each server it composed. So the caller never has to know or guess the
        server count — Gym derives it from the config and health-checks for us.
        """
        cfg = self._config
        loop = asyncio.get_running_loop()
        ready = re.compile(r"All \d+ / \d+ servers ready")
        deadline = loop.time() + cfg.startup_timeout_s
        text = ""
        while loop.time() < deadline:
            if env_proc.returncode is not None:
                raise RuntimeError(f"`gym env start` exited early (rc={env_proc.returncode}); see {env_log}")
            text = env_log.read_text(encoding="utf-8", errors="replace") if env_log.exists() else ""
            if "required uv version" in text.lower():
                raise RuntimeError(f"`gym env start` hit the uv required-version gate; see {env_log}")
            if ready.search(text):
                return
            await asyncio.sleep(2)
        raise TimeoutError(self._startup_timeout_message(text, env_log))

    def _startup_timeout_message(self, env_log_text: str, env_log: Path) -> str:
        """Explain a startup timeout in terms of *which* server did not come up.

        The bare "servers not ready" message sends you to a log that is mostly noise from the servers
        that started fine. Gym polls and prints the outstanding set every time, so naming it costs
        nothing and is usually the whole diagnosis: one server out of four means that environment's
        own startup, not the platform.

        The timeout is also a plausible *cause* rather than a symptom — a Docker-backed environment
        builds its image on first run, which routinely exceeds the 240s default — so the message says
        how to extend it instead of leaving the reader to find the knob.
        """
        cfg = self._config
        pending = pending_servers(env_log_text)
        if pending is None:
            detail = (
                "Gym never reported server readiness, so it likely failed before starting them "
                "(config composition or dependency installation)."
            )
        elif not pending[2]:
            # Gym printed a readiness line but named no outstanding server. Saying "waiting on:
            # unknown" would contradict the counts in the same sentence, so report only what is known.
            ready_count, total, _ = pending
            detail = (
                f"{ready_count} of {total} server(s) started; Gym did not name which remained "
                f"outstanding. The per-server output in {env_log.name} is prefixed with the server "
                "that emitted each line."
            )
        else:
            ready_count, total, names = pending
            detail = (
                f"{ready_count} of {total} server(s) started; still waiting on: {', '.join(names)}. "
                f"Search {env_log.name} for '({names[0]})' — each line is prefixed with the server "
                "that emitted it."
            )
        return (
            f"Gym servers for resources-server {cfg.resources_server!r} were not ready within "
            f"startup_timeout_s={cfg.startup_timeout_s}s. {detail}\n"
            f"Full startup output: {env_log}\n"
            "If the environment installs heavy dependencies or builds a container image on first run, "
            "this timeout is expected on a cold cache — raise GymRuntimeConfig.startup_timeout_s and "
            "re-run; the second run reuses what the first installed."
        )


#: Gym's per-poll readiness line, e.g.
#: ``3 / 4 servers ready. Waiting for servers to spin up: ['legal_agent_bench']``.
#: Gym knows exactly which servers are outstanding; without this we discard that and report only
#: that *something* timed out.
