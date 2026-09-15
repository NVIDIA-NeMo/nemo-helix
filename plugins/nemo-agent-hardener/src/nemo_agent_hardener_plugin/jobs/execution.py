# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the agent-hardener war-game by subprocess.

Two invocation paths against agent-hardener's CLI (own venv): the one-shot ``agent-hardener run`` and the
Studio service-driven flow (sandbox up -> benign-suite synth HITL over ``agent-hardener serve`` -> reuse
run). Both return a :class:`RunOutcome`; the job (:mod:`~nemo_agent_hardener_plugin.jobs.run`) orchestrates.

Every primary subprocess runs through :func:`_run_agent_hardener`, which points agent-hardener at a structured
``run-error.json`` and classifies a non-zero exit into a :class:`RunFailure`. The service path guards all
work after the run record is created so a mid-run failure returns a *failed* ``RunOutcome`` carrying that
record's name — the job finalizes it rather than leaving it orphaned as ``running``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_agent_hardener_plugin.cli.client import base_url
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import _common, benign_suite
from nemo_agent_hardener_plugin.jobs.errors import (
    AGENT_HARDENER_ERROR_FILE_ENVVAR,
    CATEGORY_SYNTH_SERVICE,
    AgentHardenerRunError,
    RunFailure,
    classify_exception,
    classify_subprocess,
    read_run_error,
)
from nemo_agent_hardener_plugin.jobs.hitl import StatusDetailsChannel, drive_synth_hitl
from nemo_agent_hardener_plugin.jobs.records import _create_run, _run_data, read_and_persist_suite
from nemo_agent_hardener_plugin.jobs.synth_client import launch_synth_service
from nemo_platform_plugin.job_context import JobContext

logger = logging.getLogger(__name__)

_LOG_TAIL = 4000


@dataclass
class RunOutcome:
    """Result of a war-game path.

    ``record_name`` is set only when the path created the run record up front (service-driven), so
    ``run()`` finalizes rather than creates it. ``failure`` classifies a failed run so ``run()`` records
    the cause on the run entity.
    """

    status: str
    returncode: int
    log_text: str = ""
    log_ref: Any = None
    record_name: str | None = None
    failure: RunFailure | None = None


def _event_sink_url(workspace: str, run_name: str) -> str:
    """Where agent-hardener's EventBus POSTs live events for this run (relayed to Studio over SSE)."""
    return f"{base_url()}/apis/agent-hardener/v2/workspaces/{workspace}/runs/{run_name}/events"


def _run_command(
    bin_path: Any,
    manifest: str,
    *,
    benign_suite: str | None = None,
    env_file: str | None = None,
    rounds: int = 1,
    replay_args: list[str] | None = None,
    reuse: bool = False,
) -> list[str]:
    """Build an ``agent-hardener run`` command line (the single source of truth for its flags).

    ``agent-hardener run`` has no ``--yes``; it auto-detects interactivity from stdin's tty. ``--rounds`` is
    omitted for the default single round (agent-hardener's own default), so multi-round hardening only appears
    when asked for.
    """
    cmd = [str(bin_path), "run", "--config", manifest]
    if reuse:
        cmd.append("--reuse")
    if benign_suite:
        cmd += ["--benign-suite", benign_suite]
    if env_file:
        cmd += ["--env-file", env_file]
    if rounds > 1:
        cmd += ["--rounds", str(rounds)]
    cmd += replay_args or []
    return cmd


def _run_agent_hardener(
    cmd: list[str], env: dict[str, str], log_path: Path, ctx: JobContext, *, artifact_name: str
) -> tuple[Any, str, Any, RunFailure | None]:
    """Run a primary ``agent-hardener`` subprocess, classifying a non-zero exit into a :class:`RunFailure`.

    Points agent-hardener at a fresh ``run-error.json`` (its CLI boundary writes a structured cause there) and,
    on a non-zero exit, prefers that file over a log-tail heuristic. Returns ``(completed, log_text, log_ref,
    failure)`` where ``failure`` is ``None`` on success. Teardown/best-effort commands use
    :func:`~nemo_agent_hardener_plugin.jobs._common.execute` directly instead, so they never write the error file.
    """
    err_path = ctx.storage.persistent / "run-error.json"
    if err_path.exists():
        err_path.unlink()  # a stale file from an earlier command in this run would misattribute the cause
    cmd_env = {**env, AGENT_HARDENER_ERROR_FILE_ENVVAR: str(err_path)}
    completed, log_text, log_ref = _common.execute(cmd, cmd_env, log_path, ctx, artifact_name=artifact_name)
    failure: RunFailure | None = None
    if completed.returncode != 0:
        classified = classify_subprocess(completed.returncode, log_text[-_LOG_TAIL:], read_run_error(err_path))
        failure = classified.as_failure()
    return completed, log_text, log_ref, failure


def _outcome(
    completed: Any, log_text: str, log_ref: Any, record_name: str | None, failure: RunFailure | None
) -> RunOutcome:
    """Map a finished primary subprocess to a :class:`RunOutcome` (status derived from the exit code)."""
    status = "completed" if completed.returncode == 0 else "failed"
    return RunOutcome(status, completed.returncode, log_text, log_ref, record_name, failure)


def _prepare_invocation(
    manifest: str,
    env_file: str | None,
    plugin_config: AgentHardenerConfig,
    replay_args: list[str] | None = None,
    benign_suite: str | None = None,
    model_env: dict[str, str] | None = None,
    rounds: int = 1,
) -> tuple[list[str], dict[str, str]]:
    """Build the `agent-hardener run` command + subprocess env, failing fast on missing victim secrets."""
    cmd = _run_command(
        plugin_config.agent_hardener_bin,
        manifest,
        benign_suite=benign_suite,
        env_file=env_file,
        rounds=rounds,
        replay_args=replay_args,
    )
    env = _common.build_subprocess_env(plugin_config, model_env)
    _common.check_victim_secrets(manifest, env, env_file)
    return cmd, env


def _run_one_shot(
    manifest: str,
    env_file: str | None,
    plugin_config: AgentHardenerConfig,
    ctx: JobContext,
    replay_args: list[str] | None = None,
    benign_suite: str | None = None,
    model_env: dict[str, str] | None = None,
    rounds: int = 1,
) -> RunOutcome:
    """The default path: one `agent-hardener run`, which consumes a benign suite but never generates one.

    ``agent-hardener run`` is a pure consumer (its ``_load_benign_suite`` raises ``BenignSuiteError``
    before infrastructure startup when no suite is supplied, and never prompts). So a ``benign_suite``
    CSV must be passed as ``--benign-suite`` — from an upload, or from the manifest's cached suite
    written out by the caller. Generating one is ``synth-benign``'s job, not this path's.

    ``rounds`` is honoured here too: iterative hardening is a property of the war-game, not of the
    driver, so the native (CLI) path is not limited to a single round.
    """
    cmd, env = _prepare_invocation(manifest, env_file, plugin_config, replay_args, benign_suite, model_env, rounds)
    log_path = ctx.storage.persistent / "agent-hardener.log"
    completed, log_text, log_ref, failure = _run_agent_hardener(
        cmd, env, log_path, ctx, artifact_name="agent-hardener-log"
    )
    return _outcome(completed, log_text, log_ref, None, failure)


def _run_service_driven(
    manifest: str,
    env_file: str | None,
    plugin_config: AgentHardenerConfig,
    ctx: JobContext,
    sdk: Any,
    agent: str,
    port: int,
    *,
    manifest_id: str | None = None,
    cached_suite: list[dict[str, str]] | None = None,
    stop_after_synth: bool = False,
    prepared_run_name: str | None = None,
    rounds: int = 1,
    replay_args: list[str] | None = None,
    benign_suite_override: str | None = None,
    source_run: str = "",
    model_env: dict[str, str] | None = None,
) -> RunOutcome:
    """Studio-driven path: build sandbox -> (reuse or generate the benign suite) -> replay it in the attack.

    The run record is created up front (status ``running``) so its name can address the live event stream.
    From that point on, any failure is classified and returned as a *failed* ``RunOutcome`` carrying the
    record name, so ``run()`` finalizes the record instead of leaving it orphaned as ``running``.
    """
    if not ctx.job_id:
        raise RuntimeError(
            "service-driven mode needs a submitted platform job (Studio drives the HITL via status_details)."
        )
    env = _common.build_subprocess_env(plugin_config, model_env)
    _common.check_victim_secrets(manifest, env, env_file)

    # Record the run up front so its name addresses the SSE event stream; point agent-hardener's sink at it.
    # `compile` usually pre-creates it at submit (so Studio opens the live view instantly) — reuse that;
    # otherwise create it now.
    record_name = prepared_run_name or _create_run(
        sdk,
        workspace=ctx.workspace,
        data=_run_data(
            agent,
            port,
            manifest,
            "running",
            -1,
            job_id=ctx.job_id,
            manifest_id=manifest_id or "",
            source_run=source_run,
        ),
    )
    if record_name:
        env["AGENT_HARDENER_EVENT_SINK_URL"] = _event_sink_url(ctx.workspace, record_name)

    try:
        return _drive_service_run(
            manifest,
            env_file,
            env,
            plugin_config.agent_hardener_bin,
            ctx,
            sdk,
            manifest_id=manifest_id,
            cached_suite=cached_suite,
            stop_after_synth=stop_after_synth,
            rounds=rounds,
            replay_args=replay_args,
            benign_suite_override=benign_suite_override,
            record_name=record_name,
        )
    except Exception as exc:  # classify + finalize the record rather than orphaning it as `running`
        failure = classify_exception(exc)
        logger.exception("service-driven war-game failed [%s]: %s", failure.category, failure.message)
        return RunOutcome("failed", 1, record_name=record_name, failure=failure)


def _drive_service_run(
    manifest: str,
    env_file: str | None,
    env: dict[str, str],
    bin_path: Any,
    ctx: JobContext,
    sdk: Any,
    *,
    manifest_id: str | None,
    cached_suite: list[dict[str, str]] | None,
    stop_after_synth: bool,
    rounds: int,
    replay_args: list[str] | None,
    benign_suite_override: str | None,
    record_name: str | None,
) -> RunOutcome:
    """Execute the chosen service strategy against a warm/cold sandbox (assumes the record already exists)."""
    # Explicit-suite path: use an uploaded suite override, else the manifest's cached suite. Hand the CSV to
    # agent-hardener as a file (`--benign-suite`); it seeds the file into the target's own requests.csv, so the
    # plugin doesn't mirror agent-hardener's storage layout and no synthesis/interview is needed. A single
    # self-contained war-game (`run` builds its own sandbox + forward) — no separate `up`, whose forward
    # would collide with the attack's.
    suite_path = benign_suite_override
    if suite_path is None and cached_suite:
        suite_csv = ctx.storage.persistent / "benign-suite.csv"
        benign_suite.write_suite(suite_csv, cached_suite)
        suite_path = str(suite_csv)
    if suite_path and not stop_after_synth:
        cmd = _run_command(
            bin_path, manifest, benign_suite=suite_path, env_file=env_file, rounds=rounds, replay_args=replay_args
        )
        completed, log_text, log_ref, failure = _run_agent_hardener(
            cmd, env, ctx.storage.persistent / "agent-hardener.log", ctx, artifact_name="agent-hardener-log"
        )
        return _outcome(completed, log_text, log_ref, record_name, failure)

    # No cached suite (or regenerating): bring the sandbox up so synth can probe the live victim, run the
    # interview/review HITL, and cache the reviewed suite back on the manifest.
    up_cmd = [str(bin_path), "up", "--config", manifest, *(["--env-file", env_file] if env_file else [])]
    up_done, _t, _r, up_failure = _run_agent_hardener(
        up_cmd, env, ctx.storage.persistent / "up.log", ctx, artifact_name="up-log"
    )
    if up_failure is not None:
        return RunOutcome("failed", up_done.returncode, record_name=record_name, failure=up_failure)

    # The sandbox is up. Guarantee teardown on every exit path — normal return, exception, or a SIGTERM
    # during the (minutes-long) HITL wait — so a cancelled or crashed run never orphans the victim
    # container. `agent-hardener run` self-cleans via its own teardown, so the `down` below is a redundant
    # no-op on the happy path but the safety net whenever `run` is never reached.
    try:
        # ctx.job_id is guaranteed set by _run_service_driven's guard before we get here.
        channel = StatusDetailsChannel(sdk, name=ctx.job_id or "", workspace=ctx.workspace)
        with launch_synth_service(bin_path, env, log_path=ctx.storage.persistent / "serve.log") as client:
            csv_path = drive_synth_hitl(client, manifest, channel.publish, channel.await_response)
        reviewed = (
            read_and_persist_suite(sdk, ctx, manifest_id, csv_path, interview=channel.interview) if csv_path else []
        )
        if stop_after_synth:
            # Generate/refresh only. The reviewed suite is cached on the manifest; the later attack is a
            # separate job that builds its own sandbox, so the finally below frees this one's port forward.
            return RunOutcome("completed", 0, record_name=record_name)

        # War-game against the warm sandbox, validating the just-reviewed suite. `run` is a pure consumer
        # now, so hand it the suite as a file — written to a distinct path so `run --benign-suite` doesn't
        # copy the serve artifact onto itself. An empty suite (synth found nothing) is simply omitted.
        suite_path = None
        if reviewed:
            suite_csv = ctx.storage.persistent / "benign-suite.csv"
            benign_suite.write_suite(suite_csv, reviewed)
            suite_path = str(suite_csv)
        cmd = _run_command(
            bin_path,
            manifest,
            benign_suite=suite_path,
            env_file=env_file,
            rounds=rounds,
            replay_args=replay_args,
            reuse=True,
        )
        completed, log_text, log_ref, failure = _run_agent_hardener(
            cmd, env, ctx.storage.persistent / "agent-hardener.log", ctx, artifact_name="agent-hardener-log"
        )
        return _outcome(completed, log_text, log_ref, record_name, failure)
    finally:
        _teardown_sandbox(bin_path, manifest, env, ctx)


def run_synth_benign(
    bin_path: Any,
    manifest: str,
    env_file: str | None,
    env: dict[str, str],
    ctx: JobContext,
    *,
    interview: str = "interactive",
) -> Path:
    """Run native ``agent-hardener synth-benign`` against *manifest* and return the produced ``requests.csv``.

    ``synth-benign`` is self-contained (builds the victim sandbox, probes it, tears down). The interview
    mode maps to agent-hardener's own flags: ``interactive`` (default TTY interview, inherited by
    :func:`_common.execute`), ``auto`` (``--yes`` — accept recommended defaults), ``skip``
    (``--no-interactive`` — rules-only, no prompts). Storage is pinned so the output CSV is at a known path.
    """
    root = _pin_synth_storage(manifest, ctx)
    cmd = [str(bin_path), "synth-benign", "--config", manifest]
    if env_file:
        cmd += ["--env-file", env_file]
    if interview == "auto":
        cmd.append("--yes")
    elif interview == "skip":
        cmd.append("--no-interactive")
    _completed, _log, _ref, failure = _run_agent_hardener(
        cmd, env, ctx.storage.persistent / "synth-benign.log", ctx, artifact_name="synth-benign-log"
    )
    if failure is not None:
        raise AgentHardenerRunError(failure.category, failure.message, remediation=failure.remediation)
    return _benign_requests_csv(root)


def _pin_synth_storage(manifest: str, ctx: JobContext) -> Path:
    """Point the manifest's agent-hardener ``storage.root_dir`` at a fresh dir so the output CSV is findable.

    ``synth-benign`` writes ``<storage.root_dir>/benign_profiles/<target>/requests.csv``; pinning an
    absolute, empty root lets us locate that one file without deriving agent-hardener's ``<target>`` slug.
    """
    root = (ctx.storage.persistent / "synth-storage").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = Path(manifest)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # agent-hardener's AgentManifest only permits agent|backends|garak|overrides; `storage` lives under
    # `overrides` and is deep-merged into the expanded config (mirrors jobs/manifest.py's victim_policy_path).
    data.setdefault("overrides", {}).setdefault("storage", {})["root_dir"] = str(root)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _benign_requests_csv(root: Path) -> Path:
    """The ``requests.csv`` synth-benign wrote under the pinned storage root (newest if several targets)."""
    matches = sorted(root.glob("benign_profiles/*/requests.csv"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise AgentHardenerRunError(
            CATEGORY_SYNTH_SERVICE, "synth-benign produced no requests.csv (see synth-benign.log on the host)."
        )
    return matches[-1]


def _teardown_sandbox(bin_path: Any, manifest: str, env: dict[str, str], ctx: JobContext) -> None:
    """Best-effort ``agent-hardener down`` — never masks the run outcome, and never writes the run-error file.

    (`down` takes only ``--config``; ``--env-file`` is an `up`/`run` option.)
    """
    try:
        down_cmd = [str(bin_path), "down", "--config", manifest]
        _common.execute(down_cmd, env, ctx.storage.persistent / "down.log", ctx, artifact_name="down-log")
    except Exception:  # teardown is a best-effort safety net; never mask the original outcome
        logger.warning("failed to tear down victim sandbox after service-driven run", exc_info=True)
