# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit job — runs garak against a target using inline config + target.

``nemo auditor audit run --spec-file spec.yaml`` shells out to a pre-installed
garak interpreter (default ``/app/.garak_venv/bin/python``, overridable via
``NEMO_AUDITOR_GARAK_PYTHON``).

The probe spec is expanded into individual per-probe YAML configs tracked
through ``todo/``, ``running/``, ``complete/``, and ``failed/`` directories
under persistent storage. SIGTERM is handled by saving partial results so a
resumed invocation picks up from the last completed probe. Completed per-probe
reports are aggregated via ``garak.analyze.aggregate_reports`` and registered
as job results via :meth:`~nemo_platform_plugin.job_results.JobResults.save`.
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated, ClassVar, TypeVar, cast
from uuid import uuid4

import garakapi
import yaml
from nemo_auditor.entities import AuditConfig, AuditTarget
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import AsyncPlatformClient, client_from_platform
from nemo_platform_plugin.client.response import NemoResponse
from nemo_platform_plugin.entities import parse_qualified_name
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.job_results import JobResults
from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient
from nemo_platform_plugin.models.types import ModelProvider
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

logger = logging.getLogger(__name__)

DEFAULT_GARAK_PYTHON = "/app/.garak_venv/bin/python"
GARAK_PYTHON_ENVVAR = "NEMO_AUDITOR_GARAK_PYTHON"

# garak writes reports to <XDG_DATA_HOME>/garak/<reporting.report_dir>/
# with filenames driven by reporting.report_prefix. Same layout for both
# per-probe runs and the aggregated output.
_GARAK_OUTPUT_TYPES = (
    ("report-jsonl", ".report.jsonl"),
    ("report-html", ".report.html"),
    ("report-hitlog-jsonl", ".hitlog.jsonl"),
)

# garak refuses to start unless these are set even when unused (e.g. when
# the actual creds come through IGW). services/auditor sets the same four.
_REQUIRED_API_KEY_VARS = (
    "NIM_API_KEY",
    "OPENAI_API_KEY",
    "REST_API_KEY",
    "OPENAICOMPATIBLE_API_KEY",
)

# Stdout/stderr can be MB-sized after a long garak run; keep just a tail in
# the result envelope so it stays useful for diagnostics without blowing up
# the JSON payload.
_LOG_TAIL_BYTES = 4000


class GarakFailure(Exception):
    """Raised when a garak invocation fails unrecoverably."""


# Workspace-qualified-or-bare name reference, e.g. "my-cfg" or "prod/my-cfg".
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class AuditInputSpec(BaseModel):
    """User-facing spec — each field accepts an inline entity payload OR a
    workspace-qualified name string referencing one in the entity store.

    Resolved by :meth:`AuditJob.to_spec` into a canonical :class:`AuditSpec`
    before :meth:`AuditJob.run` is invoked.
    """

    model_config = ConfigDict(extra="forbid")

    config: AuditConfig | NonEmptyStr
    target: AuditTarget | NonEmptyStr
    max_probe_retries: int = Field(default=0, ge=0)
    fail_job_on_retries_exhausted: bool = True


class AuditSpec(BaseModel):
    """Canonical, fully-resolved spec passed to :meth:`AuditJob.run`."""

    model_config = ConfigDict(extra="forbid")

    config: AuditConfig
    target: AuditTarget
    max_probe_retries: int = Field(default=0, ge=0)
    fail_job_on_retries_exhausted: bool = True


def _garak_config_dict(config: AuditConfig) -> dict:
    """Project ``AuditConfig`` to the dict shape garak's ``--config`` expects.

    Drops the ``NemoEntity`` base fields (``name``/``workspace``/etc.) and the
    plugin-specific ``description`` — garak's config schema only knows about
    ``system``/``run``/``plugins``/``reporting``.
    """
    return config.model_dump(mode="json", include={"system", "run", "plugins", "reporting"})


def _collect_report_artifacts(
    report_dir: Path,
    report_prefix: str,
    results: JobResults,
) -> dict[str, dict]:
    """Register whichever of the three garak reports exist as job results."""
    artifacts: dict[str, dict] = {}
    for name, suffix in _GARAK_OUTPUT_TYPES:
        path = report_dir / f"{report_prefix}{suffix}"
        if path.exists():
            ref = results.save(name, path)
            artifacts[name] = ref.model_dump()
        else:
            logger.debug("Garak report %s not found at %s", name, path)
    return artifacts


def _resolve_garak_python() -> str:
    return os.environ.get(GARAK_PYTHON_ENVVAR) or os.path.expanduser(DEFAULT_GARAK_PYTHON)


_EntityT = TypeVar("_EntityT", AuditConfig, AuditTarget)


async def _resolve_ref(
    value: _EntityT | str,
    entity_class: type[_EntityT],
    *,
    default_workspace: str,
    entity_client: NemoEntitiesClient | None,
    kind: str,
) -> _EntityT:
    """Return ``value`` if it's already an entity, otherwise look it up by name.

    ``entity_client`` may be ``None`` only when ``value`` is already an entity
    (the inline case). The str path always requires a client; the caller is
    expected to have surfaced a clear error before reaching here if no client
    was available.
    """
    if isinstance(value, entity_class):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected {kind} entity or name reference, got {type(value).__name__}")
    assert entity_client is not None, f"entity_client is required to resolve {kind} name ref {value!r}"
    # value is NonEmptyStr; parse "[ws/]name" via the platform helper.
    ws, name = parse_qualified_name(value, default_workspace=default_workspace)
    try:
        return await entity_client.get(entity_class, name=name, workspace=ws)
    except NemoEntityNotFoundError as exc:
        raise RuntimeError(f"{kind} '{ws}/{name}' not found in entity store") from exc


def _rewrite_options_uris(
    options: dict,
    sdk: NeMoPlatform | None,
    async_sdk: AsyncPlatformClient | None = None,
) -> None:
    """Replace ``nmp_uri_spec`` sentinels in ``options`` with concrete ``uri`` values.

    Walks the options tree (BFS over dict values, non-dicts are skipped) and,
    for every dict containing an ``nmp_uri_spec`` key, resolves
    ``nmp_uri_spec.inference_gateway`` (which must contain ``workspace`` and
    ``provider``) through the platform SDK, sets the dict's ``uri`` to the
    resolved URL, and removes the sentinel.

    Mutates ``options`` in place. Mirrors
    ``services/auditor/src/nmp/auditor/tasks/audit/main.py:rewrite_target``.

    Raises:
        ValueError: malformed sentinel, or ``uri``/``nmp_uri_spec`` conflict
            in the same dict.
        RuntimeError: sentinel present but no SDK was injected, or the SDK
            lookup itself failed.
    """
    import asyncio

    queue: list = list(options.values())
    while queue:
        node = queue.pop()
        if not isinstance(node, dict):
            continue
        queue.extend(node.values())
        spec = node.get("nmp_uri_spec")
        if not spec:
            continue
        igw_ref = spec.get("inference_gateway") if isinstance(spec, dict) else None
        if not isinstance(igw_ref, dict) or "workspace" not in igw_ref or "provider" not in igw_ref:
            raise ValueError(
                f"Invalid nmp_uri_spec: {spec!r} (expected inference_gateway with both 'workspace' and 'provider')."
            )
        if "uri" in node:
            raise ValueError("Cannot specify both 'uri' and 'nmp_uri_spec' in the same options block.")
        if sdk is None and async_sdk is None:
            raise RuntimeError(
                "nmp_uri_spec resolution requires a connected platform SDK; AuditJob.run was invoked without one."
            )
        try:
            if sdk is not None:
                models = client_from_platform(sdk, ModelsClient)
                provider = models.get_provider(workspace=igw_ref["workspace"], name=igw_ref["provider"]).data()
                uri = models.get_provider_route_openai_url(provider)
            else:
                assert async_sdk is not None
                # async_sdk path: AuditJob.run() executes inside asyncio.to_thread(), so
                # this worker thread has no running event loop — asyncio.run() is safe.
                async_models = client_from_platform(async_sdk, AsyncModelsClient)

                async def _fetch_provider() -> NemoResponse[ModelProvider]:
                    return await async_models.get_provider(workspace=igw_ref["workspace"], name=igw_ref["provider"])

                provider = asyncio.run(_fetch_provider()).data()
                uri = async_models.get_provider_route_openai_url(provider)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve inference gateway provider '{igw_ref['workspace']}/{igw_ref['provider']}': {exc}"
            ) from exc
        node["uri"] = uri
        del node["nmp_uri_spec"]


def _build_env(persistent_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in _REQUIRED_API_KEY_VARS:
        env.setdefault(key, "NOT_SET")
    env["XDG_DATA_HOME"] = str(persistent_dir)
    env["GARAK_LOG_FILE"] = str(persistent_dir / "garak.log")
    return env


def _divide_and_write_confs(config_dict: dict, todo_dir: Path) -> None:
    """Expand probe_spec into individual per-probe YAML configs in todo_dir."""
    probe_spec_str = config_dict.get("plugins", {}).get("probe_spec", "")
    probe_tags_str = config_dict.get("run", {}).get("probe_tags") or ""

    activated, unknown = garakapi.parse_plugin_spec(probe_spec_str, "probes", probe_tags_str)
    if unknown:
        raise GarakFailure(f"Invalid probe(s): '{', '.join(unknown)}'")
    if not activated:
        probe_tags_err = f" and probe tags: {probe_tags_str}" if probe_tags_str else ""
        raise GarakFailure(f"No probes found for probe spec: {probe_spec_str}{probe_tags_err}")

    for plugin in activated:
        probe = plugin.removeprefix("probes.")
        per_probe = {**config_dict, "plugins": {**config_dict.get("plugins", {}), "probe_spec": probe}}
        (todo_dir / f"{probe}.yaml").write_text(yaml.safe_dump(per_probe))


def _aggregate_reports(
    persistent: Path,
    report_dir_name: str,
    report_prefix: str,
    garak_python: str,
) -> bool:
    """Aggregate per-probe reports into a single combined report.

    Returns True if at least one completed probe had a report to aggregate,
    False if no per-probe JSONL files were found.
    """
    jsonl_pattern = str(persistent / "complete" / "*" / "garak" / report_dir_name / f"{report_prefix}.report.jsonl")
    jsonls = glob.glob(jsonl_pattern)
    if not jsonls:
        return False

    agg_dir = persistent / "garak" / report_dir_name
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_jsonl = agg_dir / f"{report_prefix}.report.jsonl"

    result = subprocess.run(
        [garak_python, "-m", "garak.analyze.aggregate_reports", "-o", str(agg_jsonl)] + jsonls,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GarakFailure(
            f"garak aggregate_reports failed (rc={result.returncode}): {result.stderr[-_LOG_TAIL_BYTES:]}"
        )

    agg_html = agg_dir / f"{report_prefix}.report.html"
    with agg_html.open("w") as html_fd:
        result = subprocess.run(
            [garak_python, "-m", "garak.analyze.report_digest", "-r", str(agg_jsonl)],
            stdout=html_fd,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise GarakFailure(f"garak report_digest failed (rc={result.returncode}): {result.stderr[-_LOG_TAIL_BYTES:]}")

    hitlog_pattern = str(persistent / "complete" / "*" / "garak" / report_dir_name / f"{report_prefix}.hitlog.jsonl")
    agg_hitlog = agg_dir / f"{report_prefix}.hitlog.jsonl"
    agg_hitlog_tmp = agg_dir / f"{report_prefix}.hitlog.jsonl.tmp"
    with agg_hitlog_tmp.open("wb") as out_fd:
        for hitlog_path in glob.glob(hitlog_pattern):
            with open(hitlog_path, "rb") as in_fd:
                shutil.copyfileobj(in_fd, out_fd)
    if agg_hitlog_tmp.stat().st_size > 0:
        shutil.move(str(agg_hitlog_tmp), str(agg_hitlog))
    else:
        agg_hitlog_tmp.unlink()

    return True


class AuditJob(NemoJob):
    """Run an audit (per-probe garak invocations) against a configured target."""

    name: ClassVar[str] = "audit"
    description: ClassVar[str] = "Run an auditor scan against a configured target."
    container: ClassVar[str] = "auditor-tasks"
    input_spec_schema: ClassVar[type[BaseModel] | None] = AuditInputSpec
    spec_schema: ClassVar[type[BaseModel] | None] = AuditSpec

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncPlatformClient | None,
        is_local: bool,
    ) -> BaseModel:
        """Resolve any name-string refs on ``input_spec`` into inline entities.

        Signature matches :meth:`NemoJob.to_spec` exactly so the override is
        Liskov-clean; we narrow types internally.

        Local-run mode: the platform scheduler passes ``entity_client=None``,
        so we build one on demand from ``async_sdk``. API-mode submissions go
        through the same path with whatever client the platform already constructed.

        ``workspace`` is used as the fallback for unqualified name strings;
        a string like ``"prod/my-cfg"`` overrides it via ``parse_qualified_name``.
        """
        assert isinstance(input_spec, AuditInputSpec), (
            f"AuditJob.to_spec received unexpected input type: {type(input_spec).__name__}"
        )
        # Only need a client if at least one field is a name reference.
        needs_lookup = isinstance(input_spec.config, str) or isinstance(input_spec.target, str)
        client = cls._resolve_entity_client(entity_client, async_sdk) if needs_lookup else None
        config = await _resolve_ref(
            input_spec.config,
            AuditConfig,
            default_workspace=workspace,
            entity_client=client,
            kind="audit config",
        )
        target = await _resolve_ref(
            input_spec.target,
            AuditTarget,
            default_workspace=workspace,
            entity_client=client,
            kind="audit target",
        )
        return AuditSpec(
            config=config,
            target=target,
            max_probe_retries=input_spec.max_probe_retries,
            fail_job_on_retries_exhausted=input_spec.fail_job_on_retries_exhausted,
        )

    @staticmethod
    def _resolve_entity_client(
        entity_client: object,
        async_sdk: AsyncPlatformClient | None,
    ) -> NemoEntitiesClient:
        """Return a ``NemoEntitiesClient`` from whatever the scheduler handed us.

        Order of preference: existing client → adapt ``async_sdk``.
        Raises ``RuntimeError`` if neither is available, which is the case
        when ``run`` is invoked locally with no SDK and the input spec
        contains a name reference (no way to resolve it).
        """
        if entity_client is not None:
            return cast(NemoEntitiesClient, entity_client)
        if async_sdk is not None:
            typed_client = client_from_platform(async_sdk, AsyncEntitiesClient)
            return NemoEntitiesClient(typed_client)
        raise RuntimeError(
            "AuditInputSpec contained a name reference but no platform "
            "client was injected. Either inline the config/target payloads, "
            "or run with a connected platform SDK."
        )

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncPlatformClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> object:
        from nemo_auditor.config import get_config
        from nemo_platform_plugin.jobs.api_factory import (
            ContainerSpec,
            CPUExecutionProviderSpec,
            EnvironmentVariable,
            PlatformJobSpec,
            PlatformJobStep,
        )
        from nemo_platform_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
        from nemo_platform_plugin.jobs.image import get_qualified_image

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="audit-job",
                    executor=CPUExecutionProviderSpec(
                        profile=profile or get_config().job_executor_profile,
                        provider="cpu",
                        container=ContainerSpec(
                            image=get_qualified_image("auditor-tasks"),
                            entrypoint=["python", "-m"],
                            command=["nemo_auditor.tasks.audit"],
                        ),
                    ),
                    config=spec.model_dump(mode="json"),
                    environment=[
                        EnvironmentVariable(
                            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
                            value=DEFAULT_JOB_STORAGE_PATH,
                        ),
                    ],
                )
            ],
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NeMoPlatform | None = None,
        async_sdk: AsyncPlatformClient | None = None,
    ) -> dict:
        spec = AuditSpec.model_validate(config)

        persistent = ctx.storage.persistent
        persistent.mkdir(parents=True, exist_ok=True)
        ctx.storage.ephemeral.mkdir(parents=True, exist_ok=True)

        todo_dir = persistent / "todo"
        running_dir = persistent / "running"
        complete_dir = persistent / "complete"
        failed_dir = persistent / "failed"
        failed_logs_dir = persistent / "failed_probe_logs"
        run_log_path = persistent / "run.log"
        target_opts_path = persistent / "target_options.json"

        garak_python = _resolve_garak_python()
        if not Path(garak_python).exists():
            raise FileNotFoundError(
                f"garak interpreter not found at {garak_python}. "
                f"Install garak in a venv there, or set ${GARAK_PYTHON_ENVVAR} "
                "to point at an existing one."
            )

        try:
            # Register SIGTERM handler before the probe loop so partial results
            # are saved if the job is paused by the scheduler.
            def _on_sigterm(signum, frame):
                logger.warning("SIGTERM received — saving partial results and exiting.")
                try:
                    _aggregate_reports(
                        persistent,
                        spec.config.reporting.report_dir,
                        spec.config.reporting.report_prefix,
                        garak_python,
                    )
                    agg_dir = persistent / "garak" / spec.config.reporting.report_dir
                    _collect_report_artifacts(agg_dir, spec.config.reporting.report_prefix, ctx.results)
                except Exception as exc:
                    logger.error("Partial aggregation failed during SIGTERM: %s", exc)
                sys.exit(0)

            try:
                signal.signal(signal.SIGTERM, _on_sigterm)
            except ValueError:
                # Only supported by jobs scheduler.
                pass

            if not running_dir.exists():
                # First run: write per-probe configs and resolve target options.
                #
                # Everything in this if-block must be idempotent until the final
                # mkdir of running_dir because it will be re-run if pause happens
                # in the middle of initialization.
                if spec.target.options:
                    rewritten_options = copy.deepcopy(spec.target.options)
                    _rewrite_options_uris(rewritten_options, sdk, async_sdk)
                    target_opts_path.write_text(json.dumps(rewritten_options))

                for d in (todo_dir, complete_dir, failed_dir, failed_logs_dir):
                    d.mkdir(parents=True, exist_ok=True)

                _divide_and_write_confs(_garak_config_dict(spec.config), todo_dir)

                running_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Resume: re-queue any probes interrupted mid-flight.
                for probe_dir in list(running_dir.iterdir()):
                    if probe_dir.is_dir():
                        probe_yaml = todo_dir / f"{probe_dir.name}.yaml"
                        if not probe_yaml.exists():
                            src = probe_dir / "config.yaml"
                            if src.exists():
                                shutil.copy(src, probe_yaml)
                        shutil.rmtree(probe_dir)

            env = _build_env(persistent)
            base_cmd = [
                garak_python,
                "-m",
                "garak",
                "--target_type",
                spec.target.type,
                "--target_name",
                spec.target.model,
            ]
            if target_opts_path.exists():
                base_cmd += ["--generator_option_file", str(target_opts_path)]

            n_total = (
                sum(1 for _ in todo_dir.glob("*.yaml"))
                + len(list(complete_dir.iterdir()))
                + len(list(failed_dir.iterdir()))
            )
            n_done = len(list(complete_dir.iterdir())) + len(list(failed_dir.iterdir()))
            garak_log = persistent / "garak.log"

            for probe_yaml in sorted(todo_dir.glob("*.yaml")):
                probe_name = probe_yaml.stem
                probe_dir = running_dir / probe_name
                report_marker = (
                    probe_dir
                    / "garak"
                    / spec.config.reporting.report_dir
                    / f"{spec.config.reporting.report_prefix}.report.html"
                )

                shutil.rmtree(probe_dir, ignore_errors=True)
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_config = probe_dir / "config.yaml"
                shutil.copy(probe_yaml, probe_config)
                cmd = base_cmd + ["--config", str(probe_config)]
                probe_env = {**env, "XDG_DATA_HOME": str(probe_dir)}

                for retry_n in range(spec.max_probe_retries + 1):
                    self.report_progress(
                        ctx,
                        work_done=n_done,
                        work_total=n_total,
                        status="running",
                        details={"probe": probe_name, "retry": str(retry_n)},
                    )
                    with run_log_path.open("a") as run_log_fd:
                        completed = subprocess.run(
                            cmd,
                            env=probe_env,
                            stdout=run_log_fd,
                            stderr=run_log_fd,
                            check=False,
                        )

                    if completed.returncode == 0 and report_marker.exists():
                        shutil.move(str(probe_dir), str(complete_dir / probe_name))
                        logger.info("Probe %s completed (retry %d).", probe_name, retry_n)
                        break

                    logger.error(
                        "Probe %s retry %d/%d failed (rc=%d).",
                        probe_name,
                        retry_n,
                        spec.max_probe_retries,
                        completed.returncode,
                    )
                    attempt_log_dir = failed_logs_dir / probe_name
                    attempt_log_dir.mkdir(parents=True, exist_ok=True)
                    if garak_log.exists():
                        shutil.copy(garak_log, attempt_log_dir / f"{uuid4()}.log")
                        garak_log.write_bytes(b"")
                else:
                    shutil.move(str(probe_dir), str(failed_dir / probe_name))
                    if garak_log.exists():
                        garak_log.write_bytes(b"")
                    if spec.fail_job_on_retries_exhausted:
                        raise GarakFailure(f"Retries exhausted for probe {probe_name!r}")
                    logger.error("Retries exhausted for %s — continuing.", probe_name)

                probe_yaml.unlink()
                n_done += 1
                self.report_progress(ctx, work_done=n_done, work_total=n_total, status="running")

            n_complete = len(list(complete_dir.iterdir()))
            if n_complete == 0:
                raise GarakFailure("All probes failed.")

            has_reports = _aggregate_reports(
                persistent,
                spec.config.reporting.report_dir,
                spec.config.reporting.report_prefix,
                garak_python,
            )
            agg_report_dir = persistent / "garak" / spec.config.reporting.report_dir
            artifacts = (
                _collect_report_artifacts(agg_report_dir, spec.config.reporting.report_prefix, ctx.results)
                if has_reports
                else {}
            )
            n_failed = len(list(failed_dir.iterdir()))
            status = "completed" if n_failed == 0 else "partial"
            self.report_progress(
                ctx,
                work_done=n_done,
                work_total=n_total,
                status=status,
                details={"probes_complete": str(n_complete), "probes_failed": str(n_failed)},
            )
            return {
                "status": status,
                "probes_total": n_total,
                "probes_complete": n_complete,
                "probes_failed": n_failed,
                "results": artifacts,
            }

        except GarakFailure as exc:
            logger.error("Audit job failed: %s", exc)
            self.report_progress(ctx, work_done=0, work_total=0, status="failed", details={"error": str(exc)})
            return {"status": "failed", "error": str(exc), "results": {}}
