# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvaluateAgentJob — evaluate a NAT agent workflow against a dataset.

Registered under the ``nemo.jobs`` entry-point group as ``agents.evaluate``.

Evaluation is provided by the ``nvidia-nat-eval`` package alongside the
``nvidia-nat-core`` runtime.  This job delegates to the ``nat eval`` CLI
subprocess so it remains decoupled from any specific eval Python API, which
lives in ``nvidia-nat-eval``.

The job is dispatched by the platform's host subprocess executor — there is
no dedicated container image.  ``nat eval`` is resolved from the workspace
venv where ``nvidia-nat-eval`` is installed alongside ``nemo-agents-plugin``.

Before invoking ``nat eval``, the eval config's judge LLMs are injected with
the platform Inference Gateway URL (same ``setdefault`` semantics as agent
deployments), so eval configs can omit ``base_url`` and route through the IGW
automatically.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Iterator

from filesets import FilesetFileSystem
from nemo_agents_plugin.refs import AgentRef, AgentTarget, classify_agent_target
from nemo_agents_plugin.utils import (
    get_base_url,
    output_dir_override,
    preflight_validate_llm_models,
    temp_injected_config,
)
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.jobs.file_manager import FilesetFileManager
from nemo_platform_plugin.refs import (
    EndpointURL,
    FilesetRef,
    LocalDir,
    OutputTarget,
    classify_output_target,
)
from nemo_platform_plugin.run_dependencies import LocalRunError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvaluateAgentSpec(BaseModel):
    """Spec for an agent evaluation job.

    Field declaration order also drives the auto-generated CLI flag
    order — keep the most-frequently-set knobs first.

    Attributes:
        agent: The agent to evaluate against.  Accepts either a
            platform-managed agent reference (``"name"`` or
            ``"workspace/name"``) or a literal HTTP(S) endpoint URL.  The
            shape is auto-detected at run time — values containing
            ``"://"`` are treated as URLs (and forwarded to
            ``nat eval --endpoint`` verbatim); anything else is resolved
            to the platform gateway URL
            ``{base_url}/apis/agents/v2/workspaces/{workspace}/agents/{name}/-``.
            When ``None`` the eval config is expected to include an
            inline agent workflow.
        eval_config: Path to the NAT evaluation YAML config file.  When
            ``eval_config_fileset`` is set, this is interpreted relative
            to the downloaded fileset's contents.
        eval_config_fileset: Optional fileset reference (``name`` or
            ``workspace/name``) that pre-stages the eval YAML and any
            sibling files (e.g. dataset).  Used by platform-managed
            submissions where the ``agents.evaluate-agent`` function
            uploads everything before submitting; local CLI runs leave
            this ``None`` and let ``eval_config`` be a real local path.
        output: Where to put the eval outputs.  Accepts either a local
            directory path (``./out``, ``/abs/out``, ``~/out``) or an
            NeMo Platform fileset reference (``"name"`` or ``"workspace/name"``).
            Path-shaped values write directly to disk; bare names upload
            results to the named fileset, creating it on demand.  When
            ``None`` the job writes to ``ctx.storage.persistent /
            "results"`` — the platform-injected persistent volume in
            container runs, a tempdir under ``$TMPDIR`` for local CLI
            runs.
        workspace: NeMo Platform workspace used to scope gateway URL injection,
            ``--agent`` resolution, and ``--output`` fileset creation
            when those values are given as bare names.
    """

    # Field order is intentional — it's also the order the auto-generated
    # CLI surfaces flags in `--help`.  The most-specified-first ordering
    # (agent → eval-config → output → workspace) puts the "what / how /
    # where" knobs ahead of the rarely-touched scoping flag.
    agent: AgentTarget | None = Field(
        default=None,
        description="Agent to evaluate against — either a platform-managed agent "
        "reference (e.g. 'calculator', 'workspace/calculator') or an HTTP(S) "
        "endpoint URL (e.g. 'http://localhost:8080').  Bare names resolve to the "
        "platform gateway URL "
        "'{base_url}/apis/agents/v2/workspaces/{workspace}/agents/{name}/-'; URLs "
        "are passed through to 'nat eval --endpoint' verbatim.  When omitted, the "
        "eval config must include an inline agent workflow.",
    )
    eval_config: str = Field(description="Path to the NAT evaluation YAML config file.")
    eval_config_fileset: FilesetRef | None = Field(
        default=None,
        description=(
            "Optional fileset reference (``name`` or ``workspace/name``).  When set, "
            "the runner downloads the fileset's contents into a tempdir and resolves "
            "``eval_config`` relative to that dir.  Local CLI runs leave this ``None``."
        ),
    )
    output: OutputTarget | None = Field(
        default=None,
        description="Where to write eval outputs — either a local directory "
        "(path-shaped: starts with '/', './', '../', '~/') or a NeMo Platform fileset "
        "reference ('name' or 'workspace/name').  Filesets are created on "
        "demand if missing.  Defaults to <ctx.storage.persistent>/results "
        "(the platform-injected persistent volume) when not provided.",
    )
    workspace: str = Field(
        default="default",
        description="Workspace name used to construct the Inference Gateway URL when "
        "injecting base_url into judge LLMs that have none set, and to resolve "
        "--agent / --output to gateway endpoints / fileset names when given a bare name.",
    )


# Defensive ceiling on `nat eval` runtime — the platform scheduler also
# enforces a job-level timeout, but pinning the worker on a hung subprocess is
# bad UX even if the scheduler eventually kills the container.
EVAL_TIMEOUT_SECONDS = 60 * 60


class EvaluateAgentJob(NemoJob):
    """Evaluate a NAT agent workflow against a dataset.

    Entry point: ``agents.evaluate = nemo_agents_plugin.jobs.evaluate_agent:EvaluateAgentJob``
    """

    name: ClassVar[str] = "evaluate"
    description: ClassVar[str] = "Evaluate an agent workflow against a dataset as a scheduled platform job."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = EvaluateAgentSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: EvaluateAgentSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Single-step PlatformJobSpec running ``nemo_agents_plugin.tasks.evaluate`` in ``nmp-agents-tasks``."""
        from nemo_platform_plugin.jobs.api_factory import (
            EnvironmentVariable,
            PlatformJobStep,
            SubprocessExecutionProviderSpec,
        )
        from nemo_platform_plugin.jobs.constants import (
            DEFAULT_JOB_STORAGE_PATH,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
        )

        spec_dict = spec.model_dump(mode="json")
        # URL workspace is the auth boundary; overwrite the spec field
        # (which has its own default and would never be missing).
        spec_dict["workspace"] = workspace

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="evaluate-agent",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agents_plugin.tasks.evaluate"],
                    ),
                    config=spec_dict,
                    # Only PERSISTENT_JOB_STORAGE_PATH_ENVVAR is declared here —
                    # the jobs backend uses the step's value to provision the
                    # job-scoped storage volume.  The ephemeral path is already
                    # set by the backend, so we don't duplicate it.
                    environment=[
                        EnvironmentVariable(
                            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
                            value=DEFAULT_JOB_STORAGE_PATH,
                        ),
                    ],
                ),
            ],
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NeMoPlatform | None = None,
    ) -> dict:
        """Run the evaluation by delegating to the ``nat eval`` CLI.

        Same entry point on both local CLI and the platform container; the
        difference is whether ``cfg.eval_config_fileset`` is set.  When it is,
        :meth:`_resolve_eval_config` downloads the fileset's contents into a
        tempdir before running ``nat eval``; when it isn't, ``cfg.eval_config``
        is used as a real local path.

        The resolved eval config is preprocessed to inject the Inference
        Gateway URL into any judge LLMs that do not already have ``base_url``
        set.  A temporary copy is written to the same directory as the
        original so that relative paths (e.g. dataset ``file_path``) continue
        to resolve.

        ``nat eval`` failures skip the fileset upload — the
        :class:`subprocess.CalledProcessError` propagates out through
        ``_resolve_output``'s except-clause (which clears ``should_upload``)
        before being translated into a ``"failed"`` status return.

        Args:
            config: Dict matching :class:`EvaluateAgentSpec`.
            sdk: Platform SDK handle, injected by the
                :class:`~nemo_platform_plugin.scheduler.NemoJobScheduler` (locally) or
                :func:`~nemo_platform_plugin.tasks.dispatcher.run_task`
                (in-container) from the ambient SDK handle.  Required when
                ``cfg.eval_config_fileset`` or a fileset-shaped ``cfg.output``
                is set (download / upload respectively); a local-directory
                output runs without it, so the parameter is declared optional
                and validated at the point of use.
            ctx: Runtime context bound by signature DI.  Both
                :class:`~nemo_platform_plugin.scheduler.NemoJobScheduler.run_local`
                and :func:`~nemo_platform_plugin.tasks.dispatcher.run_task`
                always supply one; the no-output fallback writes to
                ``ctx.storage.persistent / "results"`` and tempdirs land
                under ``ctx.storage.ephemeral`` so they sit on the
                platform-injected scratch volume.

        Returns:
            Dict with ``status`` and ``returncode`` keys.
        """
        cfg = EvaluateAgentSpec.model_validate(config)

        # Catch CalledProcessError outside the `with` so _resolve_output's
        # except-clause fires and skips the upload on failed runs.
        try:
            with (
                self._resolve_eval_config(cfg, sdk=sdk, ctx=ctx) as eval_config_path,
                self._resolve_output(cfg.output, workspace=cfg.workspace, sdk=sdk, ctx=ctx) as output_base,
            ):
                # Pre-flight: surface a missing-VirtualModel error before the
                # ``nat eval`` subprocess starts, instead of letting users see
                # an opaque subprocess failure when (e.g.) a hardcoded judge
                # model isn't provisioned.  No-op when ``sdk`` is None
                # (local-only paths that have nothing to look up against).
                preflight_validate_llm_models(eval_config_path, workspace=cfg.workspace, sdk=sdk)

                with temp_injected_config(eval_config_path, cfg.workspace) as injected_path:
                    # Run nat eval with cwd=config file's directory so relative
                    # paths in the eval YAML (e.g. dataset file_path) resolve
                    # correctly.
                    cwd = injected_path.parent
                    cmd = ["nat", "eval", "--config_file", injected_path.name]
                    # Read output_dir from the injected YAML so any ``$VAR``
                    # references inside it are honoured by ``temp_injected_config``'s
                    # ``expand_env_vars`` pass before we hand the value to
                    # ``nat eval``.
                    cmd.extend(output_dir_override(injected_path, output_base))

                    # Pass the agent endpoint via ``nat eval --endpoint`` so NAT
                    # routes evaluation requests to the running agent's /generate
                    # endpoint.  The ``agent`` field accepts both platform-managed
                    # names and literal URLs — :func:`classify_agent_target` picks
                    # one based on the presence of ``"://"`` so the user only ever
                    # sets one flag.
                    endpoint = self._resolve_endpoint(cfg.agent, workspace=cfg.workspace)
                    if endpoint:
                        cmd.extend(["--endpoint", endpoint])
                        logger.info("Evaluating against agent at %s", endpoint)

                    logger.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
                    result = subprocess.run(cmd, check=True, cwd=cwd, timeout=EVAL_TIMEOUT_SECONDS)
                    logger.info("EvaluateAgentJob completed (returncode=%d).", result.returncode)
                    return {"status": "completed", "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out after %ds; fileset upload was skipped.", EVAL_TIMEOUT_SECONDS)
            return {"status": "failed", "returncode": 124}
        except subprocess.CalledProcessError as exc:
            logger.error("Evaluation failed (returncode=%d); fileset upload was skipped.", exc.returncode)
            return {"status": "failed", "returncode": exc.returncode}

    @staticmethod
    def _resolve_endpoint(agent: AgentTarget | None, *, workspace: str) -> str | None:
        """Project the union-typed ``agent`` field down to a single URL.

        ``None`` means "no override" — the eval config carries an inline
        agent workflow.  An :class:`EndpointURL`-shaped value is returned
        verbatim.  An :class:`AgentRef`-shaped value (a bare name, with
        or without a ``workspace/`` prefix) is resolved against the
        platform gateway.
        """
        if agent is None:
            return None
        cls = classify_agent_target(agent)
        if cls is EndpointURL:
            return str(agent)
        # AgentRef: split off an explicit ``workspace/name`` prefix so
        # the spec's ``--workspace`` flag can stay implicit ("default")
        # for the common single-workspace case.
        ref = AgentRef(agent)
        if "/" in ref:
            ws, name = ref.split("/", 1)
        else:
            ws, name = workspace, ref
        base_url = get_base_url()
        endpoint = f"{base_url}/apis/agents/v2/workspaces/{ws}/agents/{name}/-"
        logger.info("Resolved --agent %s to %s", ref, endpoint)
        return endpoint

    @contextlib.contextmanager
    def _resolve_eval_config(
        self,
        cfg: EvaluateAgentSpec,
        *,
        ctx: JobContext,
        sdk: NeMoPlatform | None,
    ) -> Iterator[Path]:
        """Yield a local path to the eval YAML.

        When ``cfg.eval_config_fileset`` is set, download the fileset into a
        tempdir under ``ctx.storage.ephemeral`` and resolve
        ``cfg.eval_config`` relative to it.  Otherwise pass through verbatim.
        ``sdk`` is required on the fileset branch — when the scheduler can't
        supply one and the spec asks for a fileset, raise
        :class:`LocalRunError` early so the caller sees an actionable error
        instead of failing later inside the subprocess.
        """
        if not cfg.eval_config_fileset:
            yield Path(cfg.eval_config)
            return

        if sdk is None:
            raise LocalRunError(
                "EvaluateAgentJob.run requires a 'sdk: NeMoPlatform' to download "
                "eval_config_fileset contents, but no platform SDK was available. "
                "Set NMP_BASE_URL or pass sdk via NemoJobScheduler.run_local(sdk=...)."
            )

        ref = FilesetRef(cfg.eval_config_fileset)
        if "/" in ref:
            ws, name = ref.split("/", 1)
        else:
            ws, name = cfg.workspace, ref

        with tempfile.TemporaryDirectory(
            prefix=f".eval-config-{name}-",
            dir=str(ctx.storage.ephemeral),
        ) as tmp:
            tmp_path = Path(tmp)
            logger.info("Downloading fileset %s/%s into %s for eval config.", ws, name, tmp_path)
            files_client = client_from_platform(sdk, FilesClient)
            manager = FilesetFileManager(
                workspace=ws,
                fileset_name=name,
                filesystem=FilesetFileSystem(client=files_client),
                ensure_fileset_exists=False,
            )
            manager.download_from_url(f"{ws}/{name}", local_dir=tmp_path)
            # ``cfg.eval_config`` is caller-controlled — resolve and confirm
            # it stays inside the downloaded fileset before yielding it, so
            # an absolute path or ``..`` segment can't make ``nat eval`` read
            # arbitrary files from the task container.
            root = tmp_path.resolve()
            local_eval_config = (tmp_path / cfg.eval_config).resolve()
            if not local_eval_config.is_relative_to(root):
                raise ValueError(f"eval_config {cfg.eval_config!r} resolves outside the downloaded fileset")
            if not local_eval_config.is_file():
                raise FileNotFoundError(
                    f"eval_config '{cfg.eval_config}' was not found in fileset '{ws}/{name}' "
                    f"after download.  Available files: {sorted(p.name for p in tmp_path.iterdir())}"
                )
            yield local_eval_config

    @contextlib.contextmanager
    def _resolve_output(
        self,
        output: OutputTarget | None,
        *,
        workspace: str,
        ctx: JobContext,
        sdk: NeMoPlatform | None,
    ) -> Iterator[Path]:
        """Yield a local base directory for ``nat eval`` outputs.

        Branches on the union shape:

        - ``None`` → ``ctx.storage.persistent / "results"``.
        - :class:`LocalDir` → that directory, ``mkdir -p``-ed.
        - :class:`FilesetRef` → a fresh tempdir under
          ``ctx.storage.ephemeral``; on successful exit the tempdir is
          uploaded to the named fileset (auto-created if missing) via
          the typed Files client before being cleaned up.

        The tempdir is removed regardless of whether the upload
        succeeds; failures during upload propagate so the caller sees
        them.  ``nat eval`` failures (non-zero exit) skip the upload —
        we don't pollute the fileset with broken / partial runs.

        *sdk* is required only on the fileset branch.  When the
        scheduler can't supply one (no SDK handle in scope) and the
        output points at a fileset, we raise :class:`LocalRunError`
        early — before the subprocess runs — so the user gets an
        actionable message instead of losing the eval artifacts.
        """
        if output is None:
            local = ctx.storage.persistent / "results"
            local.mkdir(parents=True, exist_ok=True)
            logger.info("Writing eval outputs to platform-persistent dir %s", local)
            yield local
            return

        cls = classify_output_target(output)
        if cls is LocalDir:
            # ``nat eval`` runs with ``cwd=injected_path.parent``, so a
            # relative ``--output ./eval-out`` would otherwise land inside
            # the eval YAML's directory rather than the caller's CWD.
            # Resolve here so the subprocess writes to the directory we
            # advertise in the log line below.
            local = Path(str(output)).expanduser().resolve()
            local.mkdir(parents=True, exist_ok=True)
            logger.info("Writing eval outputs to local dir %s", local)
            yield local
            return

        if sdk is None:
            raise LocalRunError(
                "EvaluateAgentJob.run requires a 'sdk: NeMoPlatform' to upload "
                "results to a fileset, but no platform SDK was available. "
                "Set NMP_BASE_URL (so the local CLI can build a default SDK), "
                "pass an explicit sdk via NemoJobScheduler.run_local(sdk=...), "
                "or use --output <path> to write results to a local directory instead."
            )

        ref = FilesetRef(output)
        if "/" in ref:
            ws, name = ref.split("/", 1)
        else:
            ws, name = workspace, ref

        with tempfile.TemporaryDirectory(
            prefix=f".eval-output-{name}-",
            dir=str(ctx.storage.ephemeral),
        ) as tmp:
            tmp_path = Path(tmp)
            logger.info(
                "Staging eval outputs in %s; will upload to fileset %s/%s on success.",
                tmp_path,
                ws,
                name,
            )
            should_upload = True
            try:
                yield tmp_path
            except BaseException:
                # Don't upload partial / broken outputs from a crashed run.
                should_upload = False
                raise
            finally:
                if should_upload:
                    self._upload_to_fileset(tmp_path, fileset=name, workspace=ws, sdk=sdk)

    @staticmethod
    def _upload_to_fileset(
        local_dir: Path,
        *,
        fileset: str,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> None:
        """Upload *local_dir* recursively to the named fileset.

        The fileset is auto-created (idempotent) if it doesn't already
        exist — same semantics as ``nemo files upload <dir> <fileset>``.

        *sdk* is the platform SDK handle injected into :meth:`run` by
        the :class:`~nemo_platform_plugin.scheduler.NemoJobScheduler` (signature-based
        DI). The upload goes through the typed Files client that shares the
        SDK's transport.
        """
        files_client = client_from_platform(sdk, FilesClient)
        manager = FilesetFileManager(
            workspace=workspace,
            fileset_name=fileset,
            filesystem=FilesetFileSystem(client=files_client),
            ensure_fileset_exists=True,
        )
        manager.validate_storage()
        manager.upload(local_path=local_dir, remote_path="")
        logger.info(
            "Uploaded eval outputs from %s to fileset %s/%s.",
            local_dir,
            workspace,
            fileset,
        )
