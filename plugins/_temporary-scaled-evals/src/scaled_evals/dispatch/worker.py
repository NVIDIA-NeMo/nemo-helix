# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dispatch worker: queued evaluation row -> launched run.

The durable dispatch queue is the ``evaluations`` table itself: ``POST
/evaluations`` inserts a row at ``status='queued'`` and returns. A separate
``scaled-evals-dispatch-worker`` process repeatedly claims one active row with
``SELECT … FOR UPDATE SKIP LOCKED`` and calls :meth:`Dispatcher.run`.

Recovery is table-driven. Worker startup (and every polling pass) considers
``queued``, ``provisioning``, and ``running`` rows. ``queued`` rows are claimed
and launched; ``provisioning`` / ``running`` rows are treated as restart
recovery and resumed via the backend status reader instead of launching a new
run. If a worker process dies, no in-memory queue state is lost; the next
worker pass will see the still-active row.

:class:`Dispatcher` is the unit of work. ``claim_next`` selects durable work from
Postgres; ``run`` performs one evaluation end-to-end: load the evaluation, build a
:class:`~scaled_evals.dispatch.runtime_backend.LaunchSpec`, hand it to the
backend, advance ``evaluations.status`` to ``running``, poll the backend to a
terminal state, sync per-run artifacts to the object store, write the result
envelope back to Postgres (``result`` JSONB plus derived summary columns),
optionally upload post-run ATIF trajectories to NMP Intake when
``intake_profile_id`` is set, and set a terminal ``status`` of ``succeeded`` or
``failed``.

The synchronous poll-to-terminal loop blocks the worker slot for the whole run.
Run multiple worker containers/processes for a small worker pool; row claiming
uses ``SKIP LOCKED`` so only one worker owns a row at a time.

Both the backend resolver and the DB connection factory are injected so the
whole path runs in unit tests with a fake backend and a fake connection — no
cluster, no live Postgres. The background task opens its *own* connection
(the request-scoped one from ``get_conn`` is gone by the time it runs).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import yaml
from psycopg.rows import dict_row

from scaled_evals.api import s3
from scaled_evals.api.build.task_image_identity import verify_stored_task_image
from scaled_evals.api.failure_diagnostics import failure_category_for_code, is_retryable_failure
from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.api.repositories.benchmark_run_repository import BenchmarkRunRepository
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.repositories.execution_cleanup_repository import (
    ExecutionCleanupRepository,
)
from scaled_evals.api.repositories.execution_telemetry_repository import (
    ExecutionTelemetryRepository,
)
from scaled_evals.api.repositories.resource_usage_repository import ResourceUsageRepository
from scaled_evals.api.repositories.runtime_resource_repository import (
    RuntimeResourceRepository,
    switchyard_lease_from_row,
)
from scaled_evals.api.repositories.switchyard_campaign_repository import (
    SwitchyardCampaignRepository,
)
from scaled_evals.api.settings import settings
from scaled_evals.benchmark_archive import build_benchmark_archive
from scaled_evals.benchmark_archive_cleanup import cleanup_benchmark_archives as cleanup_archive_objects
from scaled_evals.dispatch.credentials import materialize_credential_envs
from scaled_evals.dispatch.harbor_dataset_images import (
    dataset_configs,
    effective_image_mode,
    prepare_dataset_images,
)
from scaled_evals.dispatch.registry import get_backend, get_backend_capabilities
from scaled_evals.dispatch.runtime_backend import (
    LaunchHandle,
    LaunchSpec,
    ResultSummary,
    RuntimeBackend,
    RuntimeStatus,
)
from scaled_evals.dispatch.switchyard import (
    SwitchyardProvisioner,
    SwitchyardProvisionError,
    SwitchyardReadinessError,
    SwitchyardRender,
    build_switchyard_provisioner,
    switchyard_routing_runner_env,
    switchyard_runner_env,
)
from scaled_evals.dispatch.switchyard_archive import check_campaign_evidence
from scaled_evals.dispatch.switchyard_run_manifest import write_switchyard_run_manifest
from scaled_evals.harbor_runners import resolve_harbor_runner
from scaled_evals.harbor_viewer import (
    publish_harbor_job_archive,
    result_with_harbor_viewer_publication,
)
from scaled_evals.intake.config import resolve_intake_target, resolve_routing_task
from scaled_evals.intake.experiments import ExperimentRequest, build_experiment_name, str_metadata
from scaled_evals.intake.upload import upload_job_atif_warn
from scaled_evals.models.evaluations import EvaluationResultWrite
from scaled_evals.models.execution_snapshot import (
    snapshot_credential_expectations,
    snapshot_profile_config,
    validate_execution_snapshot,
)
from scaled_evals.models.gym_identity import snapshot_evaluation
from scaled_evals.models.provenance import write_run_provenance_manifest
from scaled_evals.models.runtime import SwitchyardLease
from scaled_evals.telemetry import summarize_job_telemetry

LOG = logging.getLogger(__name__)

# Backend-reported phases that end the poll loop; everything else means the run
# is still in flight. Mapped onto evaluation_status by run() below.
_TERMINAL_PHASES = frozenset({"succeeded", "failed"})
_DB_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
# Rows the worker may pick up (including resume after a restart).
_RESUMABLE_STATUSES = frozenset({"queued", "provisioning", "running"})
_MODEL_ROUTE_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NGC_INFERENCE_API_KEY",
        "POLICY_API_KEY",
        "POLICY_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "SWITCHYARD_API_KEY",
    }
)

ConnFactory = Callable[[], AbstractContextManager[psycopg.Connection]]
BackendResolver = Callable[[str], RuntimeBackend]
_DATABASE_CONNECT_RETRY_SECONDS = 300.0
_DATABASE_CONNECT_RETRY_INTERVAL = 1.0
_LIFECYCLE_TIMEOUT_GRACE_SECONDS = 300.0
_HARBOR_PROFILE_TEMPLATE_KEYS = ("config", "harbor_config", "template", "harbor_template")
# sandbox_k8s stops a sandbox after this long when the profile omits lifecycle_timeout.
_SANDBOX_LIFECYCLE_DEFAULT_SECONDS = 3600.0
_SANDBOX_K8S_RUNTIME = "sandbox_k8s"


def _profile_lifecycle_timeout_seconds(row: Mapping[str, Any]) -> float | None:
    """Read the sandbox hard-stop budget from the immutable framework profile."""
    raw_config = row.get("framework_config") or row.get("harbor_config")
    if not isinstance(raw_config, Mapping):
        snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
        if snapshot is None:
            return None
        raw_config = snapshot_profile_config(snapshot, "framework")
        if not raw_config:
            return None

    config: Mapping[str, Any] = raw_config
    for key in _HARBOR_PROFILE_TEMPLATE_KEYS:
        template = raw_config.get(key)
        if template is None:
            continue
        if not isinstance(template, str):
            return None
        try:
            loaded = yaml.safe_load(template)
        except yaml.YAMLError:
            return None
        if not isinstance(loaded, Mapping):
            return None
        config = loaded

    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        return None
    kwargs = environment.get("kwargs")
    if not isinstance(kwargs, Mapping):
        return None
    value = kwargs.get("lifecycle_timeout")
    if isinstance(value, bool):
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    return timeout


def snapshot_agent_timeout_floor(row: Mapping[str, Any]) -> int | None:
    """Read the benchmark variant agent timeout floor frozen into the submission."""
    variant = snapshot_evaluation(row).get("benchmark_variant")
    if not isinstance(variant, Mapping):
        return None
    policy = variant.get("operational_policy")
    if not isinstance(policy, Mapping):
        return None
    floor = policy.get("agent_timeout_floor_sec")
    return None if floor is None else int(floor)


def assert_lifecycle_covers_agent_floor(row: Mapping[str, Any], floor_sec: int) -> None:
    """Fail the launch when the sandbox hard stop would truncate the agent floor.

    A variant raises the agent budget in ``task.toml``; the sandbox hard stop
    lives in the framework profile, which a variant must not rewrite. Only
    sandbox_k8s imposes a hard stop when the profile is silent, so other
    runtimes are checked against a declared ``lifecycle_timeout`` or not at all.
    """
    lifecycle = _profile_lifecycle_timeout_seconds(row)
    if lifecycle is None:
        if str(row.get("runtime") or "") != _SANDBOX_K8S_RUNTIME:
            return
        lifecycle = _SANDBOX_LIFECYCLE_DEFAULT_SECONDS
    needed = float(floor_sec) + _LIFECYCLE_TIMEOUT_GRACE_SECONDS
    if lifecycle < needed:
        raise RuntimeError(
            f"benchmark variant requires agent_timeout_floor_sec={floor_sec}, but the framework "
            f"profile stops the sandbox after {lifecycle:g}s; raise "
            f"environment.kwargs.lifecycle_timeout to at least {needed:g}"
        )


def _connect_database() -> psycopg.Connection[Any]:
    deadline = time.monotonic() + _DATABASE_CONNECT_RETRY_SECONDS
    attempt = 0
    while True:
        try:
            return psycopg.Connection[dict[str, Any]].connect(settings.resolved_database_url(), row_factory=dict_row)
        except psycopg.OperationalError:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if attempt == 1 or attempt % 10 == 0:
                LOG.warning(
                    "database connection unavailable; retrying for up to %.0f more seconds",
                    remaining,
                )
            time.sleep(min(_DATABASE_CONNECT_RETRY_INTERVAL, remaining))


@contextmanager
def _default_connect() -> Iterator[psycopg.Connection[Any]]:
    # Standalone connection for the background task. Mirrors api.db.get_conn
    # rather than importing it — get_conn is a request-scoped generator
    # dependency, not a context manager. Autocommit so status writes from the
    # long poll loop are visible without holding one transaction open for the
    # whole run.
    conn = _connect_database()
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def _resolve_backend(runtime: str) -> RuntimeBackend:
    """Resolve a runtime backend through the registry."""
    return get_backend(runtime)


def _artifact_root(evaluation_id: str, runtime: str) -> Path:
    return get_backend_capabilities(runtime).artifact_root(evaluation_id)


def _switchyard_capture_session_ids(
    artifact_root: Path,
    evaluation_id: str,
) -> tuple[str, ...]:
    """Return native Harbor trial session candidates before Switchyard teardown."""
    candidates: list[str] = []
    if artifact_root.is_dir():
        for trial_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
            trajectory_path = trial_dir / "agent" / "trajectory.json"
            result_path = trial_dir / "result.json"
            if not trajectory_path.is_file() and not result_path.is_file():
                continue

            if trajectory_path.is_file():
                try:
                    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    LOG.warning("failed to read Switchyard session from %s: %s", trajectory_path, exc)
                else:
                    if isinstance(trajectory, dict):
                        session_id = trajectory.get("session_id")
                        if isinstance(session_id, str) and session_id.strip():
                            candidates.append(session_id.strip())

            if result_path.is_file():
                try:
                    trial_result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    LOG.warning("failed to read Switchyard session from %s: %s", result_path, exc)
                else:
                    if isinstance(trial_result, dict):
                        for key in ("trial_name", "id"):
                            value = trial_result.get(key)
                            if isinstance(value, str) and value.strip():
                                candidates.append(value.strip())

            candidates.append(trial_dir.name)

    candidates.append(evaluation_id)
    return tuple(dict.fromkeys(candidates))


def _execution_id(evaluation_id: str, execution_number: int) -> str:
    return f"{evaluation_id}-x{execution_number}"


def _switchyard_topology(
    row: Mapping[str, Any],
    execution_number: int | None = None,
) -> str | None:
    if not row.get("switchyard_profile_id"):
        return None
    number = execution_number or int(row.get("execution_number") or row.get("current_execution") or 1)
    shared_benchmark = bool(row.get("benchmark_run_id") and row.get("max_concurrent_members"))
    if shared_benchmark and number == 1:
        return "shared_campaign"
    if shared_benchmark:
        return "dedicated_retry"
    return "dedicated"


def _retry_delay_seconds(evaluation_id: str, execution_number: int) -> float:
    base = min(30, 5 * (2 ** (execution_number - 1)))
    jitter = int(hashlib.sha256(evaluation_id.encode()).hexdigest()[:2], 16) % 5
    return float(base + jitter)


def _copy_runtime_logs_to_artifact_root(evaluation_id: str, runtime: str) -> None:
    capabilities = get_backend_capabilities(runtime)
    artifact_root = capabilities.artifact_root(evaluation_id)
    try:
        artifact_root_resolved = artifact_root.resolve()
    except OSError:
        artifact_root_resolved = artifact_root
    for source in capabilities.log_file_candidates(evaluation_id):
        if not source.is_file():
            continue
        try:
            if source.resolve().is_relative_to(artifact_root_resolved):
                continue
        except OSError:
            continue
        destination_name = (
            capabilities.dispatch_log_name
            if source == capabilities.dispatch_log_path(evaluation_id) and capabilities.dispatch_log_name
            else source.name
        )
        destination = artifact_root / destination_name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as exc:
            LOG.warning(
                "failed to copy runtime log %s for %s into artifacts: %s",
                source,
                evaluation_id,
                exc,
            )


def _sync_live_log_warn(
    evaluation_id: str,
    runtime: str,
    execution_id: str,
    execution_number: int,
    previous_signature: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Publish a changed runner log snapshot for APIs on another filesystem."""
    path = get_backend_capabilities(runtime).dispatch_log_path(execution_id)
    if path is None:
        return previous_signature
    try:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == previous_signature:
            return previous_signature
        s3.put_text_object(
            s3.evaluation_live_log_key(evaluation_id, execution_number),
            redact_secret_text(path.read_text(errors="replace")),
        )
        return signature
    except FileNotFoundError:
        return previous_signature
    except Exception as exc:  # noqa: BLE001 — live logs must never fail a run
        LOG.warning("failed to publish live runner log for %s: %s", evaluation_id, exc)
        return previous_signature


def _external_handle(value: Any) -> str | None:
    if not value:
        return None
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return None
    if isinstance(data, Mapping) and data.get("external_id"):
        return str(data["external_id"])
    return None


def _task_pack_missing_detail(object_key: str) -> str:
    return f"task_object_missing: task pack object is missing: {object_key}"


class DispatchClaimLost(RuntimeError):
    """The inline worker no longer owns the evaluation dispatch lease."""


@dataclass
class Dispatcher:
    """Launches queued evaluations via a pluggable backend, then writes results.

    ``resolve`` maps an evaluation's ``runtime`` to a backend; ``connect``
    yields a fresh DB connection. ``sleep`` is the inter-poll delay function.
    All default to production wiring and are overridden in tests (a fake
    backend that returns a terminal status immediately never sleeps).

    ``poll_interval`` / ``max_polls`` provide the baseline wait for a run to
    finish (default ~1h at 10s). A finite sandbox lifecycle timeout in the
    immutable framework profile extends that baseline through its hard stop plus
    a short finalization grace. Hitting the effective limit records the run as
    failed rather than leaving it stuck at ``running``.
    """

    resolve: BackendResolver = _resolve_backend
    connect: ConnFactory = _default_connect
    sleep: Callable[[float], None] = time.sleep
    poll_interval: float = field(default_factory=lambda: settings.dispatch_run_poll_interval_seconds)
    max_polls: int = field(default_factory=lambda: settings.dispatch_run_max_polls)
    claim_timeout: float = 30.0
    switchyard: SwitchyardProvisioner = field(default_factory=build_switchyard_provisioner)
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}")
    job_launcher: Callable[[str], None] | None = None
    job_reconciler: Callable[[], bool] | None = None
    _expected_dispatch_owner: str | None = field(default=None, init=False, repr=False)
    _claim_lost: threading.Event | None = field(default=None, init=False, repr=False)

    def claim_next(self) -> str | None:
        """Claim one active evaluation row for this worker process.

        The row lock is held only for this short claim transaction. Long-running
        backend work happens in :meth:`run`; status transitions on the row are
        the durable recovery marker if the process exits mid-run.
        """
        with self.connect() as conn:
            row = EvaluationRepository(conn).claim_next(
                claim_timeout=self.claim_timeout,
                worker_id=self.worker_id,
                cluster_slot_limit=settings.control_plane_cluster_run_limit,
                per_user_slot_limit=settings.control_plane_per_user_run_limit,
            )
            return None if row is None else row["id"]

    def work_once(self) -> bool:
        """Claim and process one unit of worker-owned work.

        Evaluation dispatch has priority; archive builds run only when no active
        evaluation row was claimable. Both paths use Postgres leases and
        ``SKIP LOCKED`` so multiple workers can scale out safely.
        """
        did_work = False
        if settings.dispatch_kubernetes_jobs_enabled:
            reconciler = self.job_reconciler
            if reconciler is None:
                from scaled_evals.dispatch.kubernetes_job import (
                    KubernetesEvaluationJobLauncher,
                )

                def reconcile_job() -> bool:
                    return KubernetesEvaluationJobLauncher().reconcile_one(worker_id=self.worker_id)

                reconciler = reconcile_job
            try:
                did_work = reconciler() or did_work
            except Exception:  # noqa: BLE001 - reconciliation must not stop queue dispatch
                LOG.exception("Kubernetes evaluation Job reconciliation failed")
            execution_cleanup = self.claim_next_execution_cleanup()
            if execution_cleanup is not None:
                self.cleanup_failed_execution(execution_cleanup)
                did_work = True

        # Lifecycle work must run before admitting more evaluations. Otherwise a
        # continuously non-empty evaluation queue can retain credentialed
        # Switchyard gateways indefinitely and exhaust cluster capacity.
        switchyard_resource = self.claim_next_switchyard_teardown()
        if switchyard_resource is not None:
            self.teardown_switchyard_resource(switchyard_resource)
            return True
        cleanup = self.claim_next_switchyard_campaign_cleanup()
        if cleanup is not None:
            self.cleanup_switchyard_campaign_member(cleanup)
            return True
        campaign = self.claim_next_switchyard_campaign_finalization()
        if campaign is not None:
            self.finalize_switchyard_campaign(campaign)
            return True
        campaign = self.claim_next_switchyard_campaign_deletion()
        if campaign is not None:
            self.delete_switchyard_campaign(campaign)
            return True

        evaluation_id = self.claim_next()
        if evaluation_id is not None:
            if settings.dispatch_kubernetes_jobs_enabled:
                launcher = self.job_launcher
                if launcher is None:
                    from scaled_evals.dispatch.kubernetes_job import (
                        KubernetesEvaluationJobLauncher,
                    )

                    launcher = KubernetesEvaluationJobLauncher().launch
                launcher(evaluation_id)
            else:
                self.run(evaluation_id, maintain_claim=True)
            return True
        if did_work:
            return True
        evidence_evaluation_id = self.claim_next_evidence()
        if evidence_evaluation_id is not None:
            self.build_evidence(evidence_evaluation_id)
            return True
        archive_evaluation_id = self.claim_next_archive()
        if archive_evaluation_id is not None:
            self.build_archive(archive_evaluation_id)
            return True
        with self.connect() as conn:
            benchmark_archive = BenchmarkArchiveRepository(conn).claim(claim_timeout=self.claim_timeout)
        if benchmark_archive is not None:
            if benchmark_archive["status"] == "building":
                self.build_benchmark_archive(benchmark_archive)
            return True
        with self.connect() as conn:
            cleanup_run_id = BenchmarkArchiveRepository(conn).claim_cleanup(
                interval_seconds=settings.benchmark_archive_cleanup_interval_seconds,
            )
        if cleanup_run_id is not None:
            self.cleanup_benchmark_archives(cleanup_run_id)
            return True
        return did_work

    def cleanup_benchmark_archives(self, run_id: str, *, object_keys: list[str] | None = None) -> None:
        """Best-effort immediate cleanup; durable periodic sweeps retry failures."""
        try:
            cleanup_archive_objects(self.connect, run_id, object_keys=object_keys)
        except Exception as exc:  # noqa: BLE001 - retain objects when database/store state is uncertain
            LOG.warning("benchmark archive cleanup deferred for %s: %s", run_id, redact_secret_text(str(exc)))

    def build_benchmark_archive(self, job: dict) -> None:
        stop = threading.Event()
        lost = threading.Event()

        def keep_claim() -> None:
            while not stop.wait(max(0.1, min(30.0, self.claim_timeout / 3))):
                try:
                    with self.connect() as conn:
                        owned = BenchmarkArchiveRepository(conn).heartbeat(job["benchmark_run_id"], job["claim_token"])
                    if not owned:
                        lost.set()
                        return
                except Exception:  # noqa: BLE001 - a transient DB failure is not proof of lease loss
                    LOG.exception("benchmark archive heartbeat failed")

        def check_claim() -> None:
            if lost.is_set():
                raise RuntimeError("benchmark archive worker lease lost")

        keeper = threading.Thread(target=keep_claim, daemon=True)
        keeper.start()
        archive: dict | None = None
        published = False
        try:
            with self.connect() as conn:
                BenchmarkArchiveRepository(conn).validate_members(job["benchmark_run_id"], job["members"])
            archive = build_benchmark_archive(job, check_claim=check_claim, evidence_checks=(check_campaign_evidence,))
            check_claim()
            with self.connect() as conn:
                published = BenchmarkArchiveRepository(conn).finish(job, **archive)
            if not published:
                raise RuntimeError("benchmark archive claim lost before publication")
        except Exception as exc:  # noqa: BLE001 — record a retryable export failure
            # Object-store errors may contain URLs; expose only a redacted message.
            detail = redact_secret_text(str(exc))[:2000]
            LOG.warning("benchmark archive failed for %s: %s", job["benchmark_run_id"], detail)
            with self.connect() as conn:
                BenchmarkArchiveRepository(conn).fail(job, detail)
        finally:
            stop.set()
            keeper.join(timeout=5)
            if archive is not None and not published:
                # A commit acknowledgement can fail after the row became ready.
                # Recheck authoritative references under lock before deleting.
                self.cleanup_benchmark_archives(job["benchmark_run_id"], object_keys=[archive["object_key"]])

    def claim_next_execution_cleanup(self) -> dict | None:
        with self.connect() as conn:
            return ExecutionCleanupRepository(conn).claim_one(
                worker_id=self.worker_id,
                claim_timeout=self.claim_timeout,
            )

    def cleanup_failed_execution(self, cleanup: Mapping[str, Any]) -> None:
        """Teardown one orphaned runtime before its logical evaluation retries."""
        try:
            backend = self.resolve(str(cleanup["runtime"]))
            raw_handle = cleanup.get("backend_handle")
            if isinstance(raw_handle, str):
                raw_handle = json.loads(raw_handle)
            if not isinstance(raw_handle, Mapping):
                raise ValueError("execution cleanup backend handle is not an object")
            handle = LaunchHandle.model_validate(raw_handle)
            backend.teardown(handle)
        except Exception as exc:  # noqa: BLE001 - durable cleanup retries with backoff
            detail = redact_secret_text(str(exc))
            LOG.warning(
                "execution cleanup failed for %s execution %s: %s",
                cleanup["evaluation_id"],
                cleanup["execution_number"],
                detail,
            )
            with self.connect() as conn:
                ExecutionCleanupRepository(conn).mark_failed(
                    int(cleanup["id"]),
                    worker_id=self.worker_id,
                    detail=detail,
                )
            return
        with self.connect() as conn:
            EvaluationRepository(conn).complete_execution_cleanup(
                int(cleanup["id"]),
                worker_id=self.worker_id,
                retry_delay_seconds=_retry_delay_seconds(
                    str(cleanup["evaluation_id"]),
                    int(cleanup["execution_number"]),
                ),
            )

    def claim_next_switchyard_campaign_cleanup(self) -> dict | None:
        with self.connect() as conn:
            return SwitchyardCampaignRepository(conn).claim_cleanup(
                worker_id=self.worker_id,
                claim_seconds=self.claim_timeout,
            )

    def cleanup_switchyard_campaign_member(self, cleanup: Mapping[str, Any]) -> None:
        evaluation_id = str(cleanup["evaluation_id"])
        try:
            with self.connect() as conn:
                row = EvaluationRepository(conn).load_for_dispatch(evaluation_id)
            if row is None:
                raise RuntimeError("campaign member evaluation is missing")
            backend = self.resolve(str(row["runtime"]))
            raw_handle = cleanup.get("backend_handle")
            if isinstance(raw_handle, str):
                raw_handle = json.loads(raw_handle)
            handle = (
                LaunchHandle.model_validate(raw_handle)
                if isinstance(raw_handle, Mapping)
                else LaunchHandle(backend=str(row["runtime"]), external_id=evaluation_id)
            )
            backend.teardown(handle)
        except Exception as exc:  # noqa: BLE001 — durable cleanup retries
            with self.connect() as conn:
                repo = SwitchyardCampaignRepository(conn)
                if int(cleanup.get("cleanup_attempts") or 0) >= 5:
                    repo.abandon_cleanup(evaluation_id, detail=str(exc))
                else:
                    repo.mark_cleanup_failed(evaluation_id, detail=str(exc))
            return
        with self.connect() as conn:
            SwitchyardCampaignRepository(conn).acknowledge_cleanup(evaluation_id)
        # Artifact recovery is best effort and must not keep a successfully
        # removed runtime in cleanup_pending. In particular, a retried
        # evaluation's handle may refer to a later execution number.
        self._sync_artifacts_warn(
            evaluation_id,
            str(row["runtime"]),
            execution_id=handle.external_id,
            replace=False,
        )

    def claim_next_switchyard_campaign_finalization(self) -> dict | None:
        with self.connect() as conn:
            return SwitchyardCampaignRepository(conn).claim_finalizable(
                worker_id=self.worker_id,
                claim_seconds=max(self.claim_timeout, 900.0),
            )

    def finalize_switchyard_campaign(self, campaign: Mapping[str, Any]) -> None:
        benchmark_run_id = str(campaign["benchmark_run_id"])
        lease = _campaign_lease_from_row(campaign)
        if lease is None:
            if campaign.get("resource_name"):
                self._fail_campaign_finalization(
                    benchmark_run_id,
                    "Switchyard campaign lease metadata missing or invalid",
                )
                return
            error = str(campaign.get("evidence_error") or "gateway provisioning failed")
            with self.connect() as conn:
                repo = SwitchyardCampaignRepository(conn)
                members = repo.member_ids(benchmark_run_id)
            reference = {
                "schema_version": "scaled-evals-switchyard-campaign-evidence-v1",
                "benchmark_run_id": benchmark_run_id,
                "status": "unavailable",
                "routing_stats_object_key": None,
                "routing_stats_sha256": None,
                "error": error,
            }
            try:
                for evaluation_id in members:
                    self._write_campaign_member_reference(evaluation_id, reference)
                with self.connect() as conn:
                    repo = SwitchyardCampaignRepository(conn)
                    marked = repo.mark_evidence(
                        benchmark_run_id,
                        status="unavailable",
                        object_key=None,
                        sha256=None,
                        error=error,
                        drain_seconds=0,
                        worker_id=self.worker_id,
                    )
                    if marked:
                        repo.release_member_evidence(benchmark_run_id)
            except Exception as exc:  # noqa: BLE001
                self._retry_or_finish_campaign_unavailable(campaign, str(exc), drain_seconds=0)
            return
        try:
            with self.connect() as conn:
                repo = SwitchyardCampaignRepository(conn)
                member_ids = repo.member_ids(benchmark_run_id)
            with tempfile.TemporaryDirectory(prefix=f"scaled-evals-switchyard-{benchmark_run_id}-") as tmp:
                root = Path(tmp)
                capture_note = self.switchyard.capture(
                    lease,
                    root,
                    final=True,
                    session_ids=member_ids,
                )
                stats_path = root / "switchyard" / "routing_stats_final.json"
                evidence_status = "ready" if stats_path.is_file() else "unavailable"
                evidence_error = (
                    None if stats_path.is_file() else (capture_note or "routing stats artifact was not produced")
                )
                prefix = f"benchmark-runs/{benchmark_run_id}/artifacts/"
                s3.sync_directory_to_prefix(root, prefix)
                object_key = f"{prefix}switchyard/routing_stats_final.json" if stats_path.is_file() else None
                sha256 = _file_hash(stats_path) if stats_path.is_file() else None
                reference = {
                    "schema_version": "scaled-evals-switchyard-campaign-evidence-v1",
                    "benchmark_run_id": benchmark_run_id,
                    "status": evidence_status,
                    "routing_stats_object_key": object_key,
                    "routing_stats_sha256": sha256,
                    "error": evidence_error,
                }
                for evaluation_id in member_ids:
                    self._write_campaign_member_reference(evaluation_id, reference)
                with self.connect() as conn:
                    repo = SwitchyardCampaignRepository(conn)
                    marked = repo.mark_evidence(
                        benchmark_run_id,
                        status=evidence_status,
                        object_key=object_key,
                        sha256=sha256,
                        error=evidence_error,
                        drain_seconds=lease.drain_seconds
                        if lease.drain_seconds is not None
                        else settings.switchyard_drain_seconds,
                        worker_id=self.worker_id,
                    )
                    if marked:
                        repo.release_member_evidence(benchmark_run_id)
        except Exception as exc:  # noqa: BLE001 — durable finalizer retries
            self._retry_or_finish_campaign_unavailable(
                campaign,
                str(exc),
                drain_seconds=(
                    lease.drain_seconds if lease.drain_seconds is not None else settings.switchyard_drain_seconds
                ),
            )

    def _retry_or_finish_campaign_unavailable(
        self,
        campaign: Mapping[str, Any],
        detail: str,
        *,
        drain_seconds: float,
    ) -> None:
        benchmark_run_id = str(campaign["benchmark_run_id"])
        if int(campaign.get("claim_attempt") or 0) < 5:
            self._fail_campaign_finalization(benchmark_run_id, detail)
            return
        LOG.error(
            "Switchyard campaign evidence unavailable after bounded retries for %s: %s",
            benchmark_run_id,
            detail,
        )
        with self.connect() as conn:
            repo = SwitchyardCampaignRepository(conn)
            marked = repo.mark_evidence(
                benchmark_run_id,
                status="unavailable",
                object_key=None,
                sha256=None,
                error=f"campaign evidence unavailable after 5 attempts: {detail}",
                drain_seconds=drain_seconds,
                worker_id=self.worker_id,
            )
            if marked:
                repo.release_member_evidence(benchmark_run_id)

    def _fail_campaign_finalization(self, benchmark_run_id: str, detail: str) -> None:
        LOG.warning("Switchyard campaign finalization failed for %s: %s", benchmark_run_id, detail)
        with self.connect() as conn:
            SwitchyardCampaignRepository(conn).mark_finalization_failed(
                benchmark_run_id,
                worker_id=self.worker_id,
                detail=detail,
            )

    @staticmethod
    def _write_campaign_member_reference(
        evaluation_id: str,
        reference: dict[str, Any],
    ) -> None:
        reference_key = s3.evaluation_artifact_key(
            evaluation_id,
            "switchyard/campaign_evidence.json",
        )
        s3.put_json_object(reference_key, reference)
        manifest_key = s3.evaluation_artifact_key(
            evaluation_id,
            "switchyard/run_manifest.json",
        )
        try:
            manifest = s3.read_json_object(manifest_key)
        except Exception:  # noqa: BLE001 — reference remains independently durable
            return
        outcomes = manifest.setdefault("outcomes", {})
        if isinstance(outcomes, dict):
            outcomes["campaign_routing_stats"] = reference
            s3.put_json_object(manifest_key, manifest)

    def claim_next_switchyard_campaign_deletion(self) -> dict | None:
        with self.connect() as conn:
            return SwitchyardCampaignRepository(conn).claim_due_deletion(
                worker_id=self.worker_id,
                claim_seconds=max(self.claim_timeout, 900.0),
            )

    def delete_switchyard_campaign(self, campaign: Mapping[str, Any]) -> None:
        benchmark_run_id = str(campaign["benchmark_run_id"])
        lease = _campaign_lease_from_row(campaign)
        if lease is None:
            detail = "Switchyard campaign cannot be deleted without durable managed-resource lease metadata"
            with self.connect() as conn:
                repo = SwitchyardCampaignRepository(conn)
                if int(campaign.get("claim_attempt") or 0) >= 5:
                    repo.mark_delete_unavailable(
                        benchmark_run_id,
                        worker_id=self.worker_id,
                        detail=f"{detail} after 5 attempts",
                    )
                else:
                    repo.mark_delete_failed(
                        benchmark_run_id,
                        worker_id=self.worker_id,
                        detail=detail,
                    )
            return
        try:
            self.switchyard.delete(lease)
        except Exception as exc:  # noqa: BLE001 — durable deletion retries
            with self.connect() as conn:
                SwitchyardCampaignRepository(conn).mark_delete_failed(
                    benchmark_run_id,
                    worker_id=self.worker_id,
                    detail=str(exc),
                )
            return
        with self.connect() as conn:
            SwitchyardCampaignRepository(conn).mark_deleted(
                benchmark_run_id,
                worker_id=self.worker_id,
            )

    def claim_next_archive(self) -> str | None:
        """Claim one terminal evaluation that requested an archive rebuild."""
        with self.connect() as conn:
            row = EvaluationRepository(conn).claim_next_archive(
                claim_timeout=self.claim_timeout,
                worker_id=self.worker_id,
            )
            return None if row is None else row["id"]

    def claim_next_evidence(self) -> str | None:
        """Claim one terminal evaluation needing provenance/SBOM generation."""
        with self.connect() as conn:
            row = EvaluationRepository(conn).claim_next_evidence(
                claim_timeout=self.claim_timeout,
                worker_id=self.worker_id,
            )
            return None if row is None else row["id"]

    def build_evidence(self, evaluation_id: str) -> None:
        """Generate terminal evidence and upload it before the archive is built."""
        execution_number: int | None = None
        try:
            with self.connect() as conn:
                row: dict[str, Any] | None = EvaluationRepository(conn).load_for_dispatch(evaluation_id)
                if row is None:
                    raise RuntimeError(f"evaluation not found: {evaluation_id}")
                if row.get("status") not in _DB_TERMINAL_STATUSES:
                    raise RuntimeError(f"evaluation is not terminal: {evaluation_id}")
                snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
                framework_config = self._profile_config(conn, row, snapshot=snapshot, role="framework")
                harbor_config = (
                    framework_config
                    if row.get("framework") == "harbor" and framework_config
                    else self._profile_config(conn, row, snapshot=snapshot, role="harbor")
                )
                if row.get("framework") == "harbor" and not framework_config:
                    framework_config = harbor_config
                row = {
                    **row,
                    "framework_config": framework_config,
                    "harbor_config": harbor_config,
                    "switchyard_config": self._profile_config(conn, row, snapshot=snapshot, role="switchyard"),
                    "intake_config": self._profile_config(conn, row, snapshot=snapshot, role="intake"),
                }
                execution_number = int(row.get("current_execution") or 1)
                row["switchyard_topology"] = _switchyard_topology(row, execution_number)
                if row["switchyard_topology"] == "shared_campaign":
                    resource_row = SwitchyardCampaignRepository(conn).get(str(row["benchmark_run_id"]))
                    switchyard_lease = _campaign_lease_from_row(resource_row)
                else:
                    resource_row = RuntimeResourceRepository(conn).get_switchyard(
                        evaluation_id,
                        execution_number,
                    )
                    switchyard_lease = switchyard_lease_from_row(resource_row)
                row = _row_with_switchyard(
                    row,
                    switchyard_lease,
                    resource_row,
                )
                try:
                    skill_materials = s3.read_json_object(
                        s3.evaluation_artifact_key(
                            evaluation_id,
                            "scaled-evals-extra-skill-materials.json",
                        )
                    ).get("materials", [])
                except Exception:  # noqa: BLE001 — most runs have no extra-skill artifact
                    skill_materials = []
                row["extra_skill_materials"] = skill_materials

            prefix = f"scaled-evals-evidence-{evaluation_id}-"
            with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
                evidence_root = Path(tmp)
                write_run_provenance_manifest(
                    evidence_root,
                    row,
                    status=str(row["status"]),
                    artifact_prefix=s3.evaluation_artifact_prefix(evaluation_id),
                    backend=str(row.get("runtime") or "") or None,
                    handle=_external_handle(row.get("backend_handle")),
                )
                s3.sync_evidence_files(
                    evidence_root,
                    s3.evaluation_artifact_prefix(evaluation_id),
                )
            with self.connect() as conn:
                EvaluationRepository(conn).mark_evidence_ready(
                    evaluation_id,
                    expected_execution_number=execution_number,
                )
        except Exception as exc:  # noqa: BLE001 — durable queue retries up to its limit
            LOG.warning("evidence generation failed for %s: %s", evaluation_id, exc)
            with self.connect() as conn:
                EvaluationRepository(conn).mark_evidence_failed(
                    evaluation_id,
                    str(exc),
                    expected_execution_number=execution_number,
                )

    def build_archive(self, evaluation_id: str) -> None:
        """Build ``results.tar.gz`` for an evaluation from synced artifacts."""
        runtime = None
        execution_number = None
        with self.connect() as conn:
            row = EvaluationRepository(conn).load_status_runtime(evaluation_id)
            if row is not None:
                runtime = row.get("runtime")
                execution_number = int(row.get("current_execution") or 1)
        self._build_archive_warn(
            evaluation_id,
            runtime=str(runtime) if runtime else None,
            already_claimed=True,
            execution_number=execution_number,
        )

    def claim_next_switchyard_teardown(self) -> dict | None:
        """Claim one due per-run Switchyard resource after its drain window."""
        with self.connect() as conn:
            return RuntimeResourceRepository(conn).claim_due_switchyard_teardown(
                claim_timeout=self.claim_timeout,
                worker_id=self.worker_id,
            )

    def teardown_switchyard_resource(self, resource_row: Mapping[str, Any]) -> None:
        """Delete a drained Switchyard resource and refresh artifacts/archive."""
        evaluation_id = str(resource_row["evaluation_id"])
        execution_number = int(resource_row.get("execution_number") or 1)
        execution_id = _execution_id(evaluation_id, execution_number)
        resource_id = int(resource_row["id"])
        lease = switchyard_lease_from_row(dict(resource_row))
        if lease is None:
            with self.connect() as conn:
                RuntimeResourceRepository(conn).mark_delete_failed(
                    resource_id,
                    "switchyard lease metadata missing or invalid",
                )
                self._append_switchyard_event(
                    conn,
                    evaluation_id,
                    status=self._event_status_for(conn, evaluation_id),
                    detail="switchyard teardown failed: lease metadata missing or invalid",
                )
            return

        runtime_row = self._load_status_runtime(evaluation_id)
        runtime = None if runtime_row is None else runtime_row.get("runtime")
        status = "succeeded" if runtime_row is None else str(runtime_row.get("status"))
        artifact_root: Path | None = None
        if runtime:
            try:
                artifact_root = _artifact_root(execution_id, str(runtime))
            except Exception:  # noqa: BLE001 — teardown should still delete Switchyard
                LOG.warning(
                    "could not resolve artifact root for %s runtime=%s",
                    execution_id,
                    runtime,
                )
        try:
            # Every managed resource is captured before it is marked draining.
            # Teardown may run in a later worker turn whose local artifact root
            # is empty; recapturing here would overwrite the uploaded live
            # snapshot after rollback or drain.
            self.switchyard.delete(lease)
        except Exception as exc:  # noqa: BLE001 — retryable resource cleanup
            detail = f"switchyard teardown failed: {exc}"
            with self.connect() as conn:
                RuntimeResourceRepository(conn).mark_delete_failed(resource_id, detail)
                self._append_switchyard_event(
                    conn,
                    evaluation_id,
                    status=status,
                    detail=detail,
                )
            return

        with self.connect() as conn:
            RuntimeResourceRepository(conn).mark_deleted(resource_id)
            self._append_switchyard_event(
                conn,
                evaluation_id,
                status=status,
                detail=f"switchyard deleted: {lease.name}",
            )
        is_current_execution = (
            artifact_root is not None
            and runtime_row is not None
            and int(runtime_row.get("current_execution") or 1) == execution_number
        )
        if is_current_execution:
            self._sync_artifacts_warn(
                evaluation_id,
                str(runtime),
                execution_id=execution_id,
                execution_number=execution_number,
                replace=False,
            )
            self._build_archive_warn(
                evaluation_id,
                runtime=str(runtime),
                artifact_root=artifact_root,
                execution_number=execution_number,
            )

    def work_forever(self, *, idle_sleep: float = 2.0) -> None:
        """Run the durable queue worker loop."""
        while True:
            if not self.work_once():
                self.sleep(idle_sleep)

    def run(
        self,
        evaluation_id: str,
        *,
        maintain_claim: bool = False,
        expected_execution_number: int | None = None,
    ) -> None:
        """Run one evaluation, maintaining the lease only for inline dispatch.

        Kubernetes evaluation Jobs are fenced by ``dispatch_job_name`` and call
        this method with the default. Inline Compose workers call it with
        ``maintain_claim=True`` because task staging and backend launch can take
        longer than the queue's recovery timeout.
        """
        if not maintain_claim:
            if expected_execution_number is None:
                self._run_claimed(evaluation_id)
            else:
                self._run_claimed(
                    evaluation_id,
                    expected_execution_number=expected_execution_number,
                )
            return

        if self.claim_timeout <= 0:
            raise ValueError("claim_timeout must be positive")
        stop = threading.Event()
        lost = threading.Event()
        interval = max(0.01, min(self.claim_timeout / 3, 10.0))
        self._expected_dispatch_owner = self.worker_id
        self._claim_lost = lost
        try:
            self._renew_inline_claim(evaluation_id)
        except DispatchClaimLost:
            LOG.warning("dispatch claim lost for %s before inline run started", evaluation_id)
            self._expected_dispatch_owner = None
            self._claim_lost = None
            return
        keeper = threading.Thread(
            target=self._keep_inline_claim_alive,
            args=(evaluation_id, stop, lost, interval),
            name=f"dispatch-lease-{evaluation_id}",
            daemon=True,
        )
        keeper.start()
        try:
            if expected_execution_number is None:
                self._run_claimed(evaluation_id)
            else:
                self._run_claimed(
                    evaluation_id,
                    expected_execution_number=expected_execution_number,
                )
        except DispatchClaimLost:
            LOG.warning("dispatch claim lost for %s; stale worker stopped", evaluation_id)
        finally:
            stop.set()
            keeper.join(timeout=min(interval + 1.0, 5.0))
            if keeper.is_alive():
                LOG.warning("dispatch lease keeper did not stop promptly for %s", evaluation_id)
            self._expected_dispatch_owner = None
            self._claim_lost = None

    def _keep_inline_claim_alive(
        self,
        evaluation_id: str,
        stop: threading.Event,
        lost: threading.Event,
        interval: float,
    ) -> None:
        while not stop.wait(interval):
            try:
                with self.connect() as conn:
                    owned = EvaluationRepository(conn).heartbeat_claim(
                        evaluation_id,
                        worker_id=self.worker_id,
                    )
            except Exception:  # noqa: BLE001 - a transient DB failure is not proof of lease loss
                LOG.exception("dispatch lease heartbeat failed for %s", evaluation_id)
                continue
            if not owned:
                lost.set()
                return

    def _renew_inline_claim(self, evaluation_id: str) -> None:
        if self._expected_dispatch_owner is None:
            return
        if self._claim_lost is not None and self._claim_lost.is_set():
            raise DispatchClaimLost(evaluation_id)
        with self.connect() as conn:
            owned = EvaluationRepository(conn).heartbeat_claim(
                evaluation_id,
                worker_id=self._expected_dispatch_owner,
            )
        if not owned:
            if self._claim_lost is not None:
                self._claim_lost.set()
            raise DispatchClaimLost(evaluation_id)

    @contextmanager
    def _maintain_campaign_provisioning_claim(
        self,
        benchmark_run_id: str,
        *,
        claim_attempt: int,
    ):
        stop = threading.Event()
        lost = threading.Event()
        interval = max(0.01, min(self.claim_timeout / 3, 10.0))

        def keep_alive() -> None:
            while not stop.wait(interval):
                try:
                    with self.connect() as heartbeat_conn:
                        owned = SwitchyardCampaignRepository(heartbeat_conn).renew_provisioning_claim(
                            benchmark_run_id,
                            worker_id=self.worker_id,
                            claim_attempt=claim_attempt,
                            claim_seconds=self.claim_timeout,
                        )
                except Exception:  # noqa: BLE001 - transient DB errors do not lose ownership
                    LOG.exception(
                        "Switchyard campaign provisioning heartbeat failed for %s",
                        benchmark_run_id,
                    )
                    continue
                if not owned:
                    lost.set()
                    return

        keeper = threading.Thread(
            target=keep_alive,
            name=f"switchyard-campaign-lease-{benchmark_run_id}",
            daemon=True,
        )
        keeper.start()
        try:
            yield lost
        finally:
            stop.set()
            keeper.join(timeout=min(interval + 1.0, 5.0))
            if keeper.is_alive():
                LOG.warning(
                    "Switchyard campaign lease keeper did not stop promptly for %s",
                    benchmark_run_id,
                )

    def _run_claimed(
        self,
        evaluation_id: str,
        *,
        expected_execution_number: int | None = None,
    ) -> None:
        """Load, launch, poll to terminal, persist result, optionally upload ATIF.

        Safe to call directly (unit tests, future out-of-process worker). One
        evaluation id in; terminal ``evaluations.status`` out. Never raises for
        launch/poll/backend failures — those become ``status='failed'`` with
        ``status_detail``.

        **Lifecycle** (maps to ``evaluation_status``):

        1. Load row + task revision ``image_ref``; no-op unless
           ``status`` is in ``queued`` / ``provisioning`` / ``running``.
        2. ``provisioning`` — build :class:`~scaled_evals.dispatch.runtime_backend.LaunchSpec`
           (profile ids, credential refs, built sandbox ``image_ref``) and call
           ``backend.launch`` (skipped on resume — reuses ``backend_handle``).
        3. ``running`` — poll ``backend.status`` until terminal or timeout;
           persist ``backend_handle`` from launch.
        4. On backend ``succeeded``: sync the execution's job dir to S3/RustFS,
           optional post-run **ATIF** upload when
           ``intake_profile_id`` is set (:meth:`_maybe_upload_atif`), ask the
           backend to ``summarize`` its result, then :meth:`_persist_result`
           writes the framework-typed ``result`` JSONB plus the derived
           ``reward`` / ``n_trials`` / ``n_errored`` / ``finished_at`` and sets
           ``status='succeeded'``.
        5. If the outer runner disappears without terminal metadata: record a
           retry event, then requeue the same logical evaluation with a fresh
           execution identity, up to ``max_executions``.
        6. On a non-retryable failure or exhausted execution budget: sync the
           terminal execution to S3/RustFS and persist ``status='failed'``. A
           backend failure carrying a trustworthy raw trial envelope is
           summarized and retained for outcome diagnostics; pure
           dispatch/control-plane failures have no ``result`` row.

        The poll loop blocks this worker process for the whole run (~1h baseline
        at 10s intervals, extended by a longer finite profile lifecycle timeout).
        Scale dispatch by running multiple worker processes.

        The worker only runs single task evaluations. A benchmark run is just a
        ``benchmark_runs`` row plus one ordinary member evaluation per task
        (spawned at create time); the worker runs those members like any other
        evaluation and is otherwise unaware of benchmarks. The run's
        status/reward are derived from its members on read (see
        ``benchmark_run_repository.derive_run_view``), so there is no fan-in here.
        """
        with self.connect() as conn:
            row: dict[str, Any] | None = self._load(conn, evaluation_id)
            if row is None:
                return
            if row["status"] == "cancelled" and row.get("cancel_teardown_status") == "pending":
                EvaluationRepository(conn).record_cancel_teardown_succeeded(evaluation_id)
                return
            if row["status"] not in _RESUMABLE_STATUSES:
                return

            execution_number = int(row.get("current_execution") or 1)
            if expected_execution_number is not None and execution_number != expected_execution_number:
                LOG.info(
                    "execution %s for %s is stale; current execution is %s",
                    expected_execution_number,
                    evaluation_id,
                    execution_number,
                )
                return
            resume = row["status"] in {"provisioning", "running"} and row.get("backend_handle")
            execution_id = (
                _external_handle(row.get("backend_handle"))
                if resume
                else _execution_id(evaluation_id, execution_number)
            ) or _execution_id(evaluation_id, execution_number)
            try:
                artifact_root = _artifact_root(execution_id, row["runtime"])
            except Exception as exc:  # noqa: BLE001 — unknown/misregistered runtime
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=str(exc),
                    failure_code=type(exc).__name__,
                )
                return
            row = {
                **row,
                "execution_number": execution_number,
                "execution_id": execution_id,
            }
            if not resume:
                object_key = row.get("tarball_object_key")
                if object_key:
                    try:
                        pack_exists = s3.object_exists(str(object_key))
                    except Exception as exc:  # noqa: BLE001 — dispatch must terminalize cleanly
                        self._set_status(
                            conn,
                            evaluation_id,
                            "failed",
                            execution_number=execution_number,
                            detail=(f"object_store_unavailable: could not read task pack object: {exc}"),
                            failure_code="object_store_unavailable",
                        )
                        return
                    if not pack_exists:
                        detail = _task_pack_missing_detail(str(object_key))
                        self._set_status(
                            conn,
                            evaluation_id,
                            "failed",
                            execution_number=execution_number,
                            detail=detail,
                            failure_code="task_object_missing",
                        )
                        return
            if not resume and row["status"] != "provisioning":
                # --- provisioning: assemble LaunchSpec and hand off to the backend ---
                self._set_status(
                    conn,
                    evaluation_id,
                    "provisioning",
                    execution_number=execution_number,
                )
                row = {**row, "status": "provisioning"}

            try:
                snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
                expected_credentials = {} if snapshot is None else snapshot_credential_expectations(snapshot)
            except Exception as exc:  # noqa: BLE001 — record invalid execution snapshot
                self._write_provenance_warn(row, status="failed", artifact_root=artifact_root)
                self._sync_artifacts_warn(
                    evaluation_id,
                    row["runtime"],
                    execution_id=execution_id,
                    execution_number=execution_number,
                )
                self._build_archive_warn(
                    evaluation_id,
                    runtime=row["runtime"],
                    artifact_root=artifact_root,
                    execution_number=execution_number,
                )
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=str(exc),
                    failure_code=type(exc).__name__,
                )
                return

            try:
                framework_config = self._load_framework_config(conn, row)
                harbor_config = (
                    framework_config
                    if row.get("framework") == "harbor" and framework_config
                    else self._load_harbor_config(conn, row)
                )
                if row.get("framework") == "harbor" and not framework_config:
                    framework_config = harbor_config
                switchyard_config = self._load_switchyard_config(conn, row)
                intake_config = self._load_intake_config(conn, row)
                materialized_credentials = materialize_credential_envs(
                    conn,
                    row["credentials"] or {},
                    switchyard_bindings=switchyard_config.get("credential_bindings") or None,
                    expected=expected_credentials,
                )
                credential_env = materialized_credentials.runner
                switchyard_credential_env = materialized_credentials.switchyard
            except Exception as exc:  # noqa: BLE001 — record profile load/validation failure
                self._write_provenance_warn(row, status="failed", artifact_root=artifact_root)
                self._sync_artifacts_warn(
                    evaluation_id,
                    row["runtime"],
                    execution_id=execution_id,
                    execution_number=execution_number,
                )
                self._build_archive_warn(
                    evaluation_id,
                    runtime=row["runtime"],
                    artifact_root=artifact_root,
                    execution_number=execution_number,
                )
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=str(exc),
                    failure_code=type(exc).__name__,
                )
                return

            row = {
                **row,
                "framework_config": framework_config,
                "harbor_config": harbor_config,
                "switchyard_config": switchyard_config,
                "intake_config": intake_config,
                "switchyard_topology": _switchyard_topology(row, execution_number),
            }

            routing_env = switchyard_routing_runner_env(
                task=resolve_routing_task(
                    intake_config,
                    task_slug=row.get("task_slug"),
                ),
                session_id=evaluation_id,
            )
            shared_campaign_id = (
                str(row["benchmark_run_id"]) if row.get("switchyard_topology") == "shared_campaign" else None
            )
            repair_shared_campaign: Callable[[str], dict[str, Any] | None] | None = None
            switchyard_lease: SwitchyardLease | None = None
            if resume:
                if shared_campaign_id is not None:
                    resource_row = SwitchyardCampaignRepository(conn).get(shared_campaign_id)
                    switchyard_lease = _campaign_lease_from_row(resource_row)
                else:
                    resource_row = RuntimeResourceRepository(conn).get_switchyard(
                        evaluation_id,
                        execution_number,
                    )
                    switchyard_lease = switchyard_lease_from_row(resource_row)
                row = _row_with_switchyard(row, switchyard_lease, resource_row)
                if switchyard_lease is not None:
                    credential_env = _runner_env_with_switchyard(
                        credential_env,
                        {
                            **switchyard_runner_env(switchyard_lease),
                            **routing_env,
                        },
                    )
            elif row.get("switchyard_profile_id"):
                try:
                    effective_switchyard_config = _switchyard_config_for_network_policy(
                        switchyard_config,
                        str(row.get("network_policy") or "unrestricted"),
                    )
                    resource_row = None
                    if shared_campaign_id is not None:
                        if effective_switchyard_config.get("mode", "managed") != "managed":
                            raise ValueError("shared Switchyard campaigns require managed mode")
                        campaign_repo = SwitchyardCampaignRepository(conn)
                        campaign, owns_provisioning = campaign_repo.ensure_and_claim_provisioning(
                            benchmark_run_id=shared_campaign_id,
                            profile_id=str(row["switchyard_profile_id"]),
                            config_hash=_stable_hash(effective_switchyard_config),
                            credential_hash=_stable_hash(
                                {
                                    "credential_ids": row.get("credentials") or {},
                                    "snapshot": expected_credentials,
                                }
                            ),
                            max_concurrent_members=int(row["max_concurrent_members"]),
                            worker_id=self.worker_id,
                            claim_seconds=self.claim_timeout,
                        )

                        def repair_shared(detail: str) -> dict[str, Any] | None:
                            return self._repair_shared_campaign(
                                conn,
                                shared_campaign_id,
                                detail=detail,
                                evaluation_id=evaluation_id,
                                profile_id=str(row["switchyard_profile_id"]),
                                raw_config=effective_switchyard_config,
                                credential_env=switchyard_credential_env,
                                artifact_root=artifact_root,
                            )

                        repair_shared_campaign = repair_shared
                        if owns_provisioning:
                            resource_row, switchyard_render = self._provision_shared_campaign(
                                conn,
                                campaign,
                                evaluation_id=evaluation_id,
                                profile_id=str(row["switchyard_profile_id"]),
                                raw_config=effective_switchyard_config,
                                credential_env=switchyard_credential_env,
                                artifact_root=artifact_root,
                            )
                        else:
                            resource_row = self._wait_for_campaign_ready(
                                conn,
                                shared_campaign_id,
                                evaluation_id=evaluation_id,
                                repair=repair_shared_campaign,
                            )
                        switchyard_lease = _campaign_lease_from_row(resource_row)
                        if switchyard_lease is None:
                            raise RuntimeError("Switchyard campaign lease is missing")
                        switchyard_render = None
                    else:
                        resource_repo = RuntimeResourceRepository(conn)
                        resource_row = (
                            resource_repo.get_switchyard(evaluation_id, execution_number)
                            if execution_number > 1
                            else None
                        )
                        switchyard_lease = switchyard_lease_from_row(resource_row)
                        switchyard_render = None
                        if (
                            switchyard_lease is None
                            or resource_row is None
                            or resource_row.get("status") != "provisioned"
                        ):

                            def persist_lease(lease: SwitchyardLease) -> None:
                                nonlocal resource_row
                                if lease.mode != "managed":
                                    return
                                resource_row = resource_repo.upsert_switchyard_provisioned(
                                    evaluation_id=evaluation_id,
                                    execution_number=execution_number,
                                    lease=lease,
                                )

                            switchyard_render = self.switchyard.provision(
                                evaluation_id=(execution_id if execution_number > 1 else evaluation_id),
                                profile_id=str(row["switchyard_profile_id"]),
                                raw_config=effective_switchyard_config,
                                credential_env=switchyard_credential_env,
                                artifact_root=artifact_root,
                                persist_lease=persist_lease,
                            )
                            switchyard_lease = switchyard_render.lease
                            # Test doubles and older provisioners may not invoke
                            # the pre-apply callback. Preserve the post-success
                            # upsert as a compatibility fallback.
                            if switchyard_lease.mode == "managed" and resource_row is None:
                                resource_row = resource_repo.upsert_switchyard_provisioned(
                                    evaluation_id=evaluation_id,
                                    execution_number=execution_number,
                                    lease=switchyard_lease,
                                )
                    self._append_switchyard_event(
                        conn,
                        evaluation_id,
                        status="provisioning",
                        detail=(
                            f"switchyard {switchyard_lease.mode}: "
                            f"{switchyard_lease.name or switchyard_lease.endpoint} "
                            f"({switchyard_lease.endpoint_identity or switchyard_lease.endpoint})"
                        ),
                    )
                    if switchyard_lease.trust_warning:
                        self._append_switchyard_event(
                            conn,
                            evaluation_id,
                            status="provisioning",
                            detail=f"warning: {switchyard_lease.trust_warning}",
                        )
                    if switchyard_lease.mode == "external" and row.get("network_policy") == "default_deny":
                        self._append_switchyard_event(
                            conn,
                            evaluation_id,
                            status="provisioning",
                            detail=(
                                "warning: network_policy=default_deny blocks the external "
                                "Switchyard endpoint; use scoped_egress with an explicit "
                                "destination grant or an operator-managed cluster egress path"
                            ),
                        )
                    if switchyard_config.get("book_mode") == "closed" and row.get("network_policy") != "default_deny":
                        self._append_switchyard_event(
                            conn,
                            evaluation_id,
                            status="provisioning",
                            detail=(
                                "warning: Switchyard book_mode=closed restricts configured "
                                f"model traffic, but network_policy={row.get('network_policy')} "
                                "may permit direct gateway bypass; use default_deny for "
                                "proxy-only isolation"
                            ),
                        )
                    row = _row_with_switchyard(row, switchyard_lease, resource_row)
                    credential_env = _runner_env_with_switchyard(
                        credential_env,
                        {
                            **(
                                switchyard_render.runner_env
                                if switchyard_render is not None
                                else switchyard_runner_env(switchyard_lease)
                            ),
                            **routing_env,
                        },
                    )
                    self._write_switchyard_run_manifest_warn(
                        row,
                        status="provisioning",
                        artifact_root=artifact_root,
                    )
                except Exception as exc:  # noqa: BLE001 — record switchyard provisioning failure
                    detail = f"switchyard provision failed: {exc}"
                    readiness_failure = isinstance(exc, SwitchyardReadinessError) or (
                        "switchyard readiness failed" in str(exc).lower()
                    )
                    if shared_campaign_id is None:
                        resource_row = RuntimeResourceRepository(conn).get_switchyard(
                            evaluation_id,
                            execution_number,
                        )
                        failed_lease = switchyard_lease_from_row(resource_row)
                        if failed_lease is not None:
                            row = _row_with_switchyard(row, failed_lease, resource_row)
                            cleanup_note = self._capture_and_drain_switchyard_warn(
                                conn,
                                row,
                                artifact_root,
                                status="failed",
                                drain_seconds_override=0,
                                capture=not isinstance(exc, SwitchyardProvisionError),
                            )
                            if cleanup_note:
                                detail = f"{detail}; {cleanup_note}"
                    self._write_provenance_warn(row, status="failed", artifact_root=artifact_root)
                    self._sync_artifacts_warn(
                        evaluation_id,
                        row["runtime"],
                        execution_id=execution_id,
                        execution_number=execution_number,
                    )
                    self._build_archive_warn(
                        evaluation_id,
                        runtime=row["runtime"],
                        artifact_root=artifact_root,
                        execution_number=execution_number,
                    )
                    if readiness_failure:
                        scheduled = EvaluationRepository(conn).schedule_retry(
                            evaluation_id,
                            execution_number=execution_number,
                            failure_code="SwitchyardReadinessError",
                            failure_category="infrastructure",
                            delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
                            expected_dispatch_owner=self._expected_dispatch_owner,
                        )
                        if scheduled is not None:
                            return
                    self._set_status(
                        conn,
                        evaluation_id,
                        "failed",
                        execution_number=execution_number,
                        detail=detail,
                        failure_code=("SwitchyardReadinessError" if readiness_failure else type(exc).__name__),
                    )
                    return

            # validate_execution_snapshot already guarantees an evaluation object.
            snapshot_evaluation: Mapping[str, Any] = snapshot["evaluation"] if snapshot is not None else row
            runner_metadata = snapshot_evaluation.get("runner_metadata") or {}
            runner_artifact: Mapping[str, Any] | None = (
                runner_metadata.get("artifact") if isinstance(runner_metadata, Mapping) else {}
            )
            if not isinstance(runner_artifact, Mapping):
                runner_artifact = {}
            agent_floor = snapshot_agent_timeout_floor(row)
            task_slug = snapshot["task"].get("slug") if snapshot else row.get("task_slug")
            if task_slug is not None and not isinstance(task_slug, str):
                raise ValueError("task slug must be a string")
            spec = LaunchSpec(
                evaluation_id=execution_id,
                benchmark_run_id=row.get("benchmark_run_id"),
                name=row["name"],
                task_slug=task_slug,
                framework=row["framework"],
                framework_version=snapshot_evaluation.get("framework_version"),
                runner_image_ref=snapshot_evaluation.get("runner_image_ref"),
                runner_image_digest=snapshot_evaluation.get("runner_image_digest"),
                runner_source_revision=runner_artifact.get("source_revision"),
                runner_package_version=runner_artifact.get("package_version"),
                allow_live_runner_fallback=snapshot is None,
                framework_adapter_version=snapshot_evaluation.get("framework_adapter_version"),
                sandbox_k8s_version=snapshot_evaluation.get("sandbox_k8s_version"),
                agent_bundle=runner_metadata.get("agent_bundle") if isinstance(runner_metadata, Mapping) else None,
                harbor_dir=(
                    resolve_harbor_runner(row.get("framework_version")).harbor_dir
                    if row["framework"] == "harbor"
                    else None
                ),
                image_ref=row["image_ref"] or "",
                image_digest=row.get("image_digest"),
                n_attempts=row.get("n_attempts") or 1,
                parallelism=row["parallelism"],
                network_policy=str(row.get("network_policy") or "unrestricted"),
                network_policy_config=row.get("network_policy_config") or {},
                tarball_object_key=row.get("tarball_object_key"),
                extra_skill_object_keys=row.get("extra_skill_object_keys") or [],
                instruction_prefix=row.get("instruction_prefix"),
                instruction_postfix=row.get("instruction_postfix"),
                agent_timeout_floor_sec=agent_floor,
                initial_user_turns=row.get("initial_user_turns") or [],
                harbor_profile_id=row["harbor_profile_id"],
                framework_config=framework_config,
                harbor_config=harbor_config,
                switchyard_profile_id=row["switchyard_profile_id"],
                switchyard_config=switchyard_config,
                switchyard=switchyard_lease,
                intake_profile_id=row["intake_profile_id"],
                credentials=row["credentials"] or {},
                credential_env=credential_env,
            )
            try:
                backend = self.resolve(row["runtime"])
            except Exception as exc:  # noqa: BLE001 — record any launch failure
                self._capture_and_drain_switchyard_warn(conn, row, artifact_root, status="failed")
                self._write_switchyard_run_manifest_warn(
                    row,
                    status="failed",
                    artifact_root=artifact_root,
                    harbor_rc=1,
                )
                self._write_provenance_warn(row, status="failed", artifact_root=artifact_root)
                self._sync_artifacts_warn(
                    evaluation_id,
                    row["runtime"],
                    execution_id=execution_id,
                    execution_number=execution_number,
                )
                self._build_archive_warn(
                    evaluation_id,
                    runtime=row["runtime"],
                    artifact_root=artifact_root,
                    execution_number=execution_number,
                )
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=str(exc),
                    failure_code=type(exc).__name__,
                )
                return

            if resume:
                handle = self._resume_handle(row)
            else:
                try:
                    self._renew_inline_claim(evaluation_id)
                    if agent_floor is not None:
                        assert_lifecycle_covers_agent_floor(row, agent_floor)
                    if (
                        spec.framework == "harbor"
                        and spec.harbor_config.get("dataset_only") is True
                        and effective_image_mode(spec.harbor_config) == "managed"
                    ):
                        imports = prepare_dataset_images(
                            conn,
                            datasets=dataset_configs(spec.harbor_config),
                            harbor_dir=str(spec.harbor_dir or settings.harbor_dir),
                            renew=lambda: self._renew_inline_claim(evaluation_id),
                        )
                        spec = spec.model_copy(
                            update={"harbor_dataset_image_imports": [asdict(item) for item in imports]}
                        )
                    should_launch = True
                    if shared_campaign_id is not None:
                        should_launch = self._wait_for_campaign_permit(
                            conn,
                            shared_campaign_id,
                            evaluation_id=evaluation_id,
                            execution_number=execution_number,
                            repair=repair_shared_campaign,
                        )
                    if should_launch:
                        if spec.image_ref:
                            # Tag-form runtime references are required by signed-image admission.
                            # Re-resolve at the last application-controlled boundary;
                            # the platform admission controller remains final authority.
                            verify_stored_task_image(spec.image_ref, spec.image_digest)
                        handle = backend.launch(spec)
                    else:
                        handle = LaunchHandle(backend=row["runtime"], external_id=execution_id)
                    try:
                        self._renew_inline_claim(evaluation_id)
                    except DispatchClaimLost:
                        if should_launch:
                            self._teardown_failed_runtime_warn(backend, handle)
                        raise
                except Exception as exc:  # noqa: BLE001 — record any launch failure
                    if isinstance(exc, DispatchClaimLost):
                        raise
                    detail = str(exc)
                    if shared_campaign_id is not None:
                        SwitchyardCampaignRepository(conn).acknowledge_cleanup(
                            evaluation_id,
                            not_launched=True,
                        )
                    self._capture_switchyard_warn(
                        row,
                        artifact_root,
                    )
                    self._write_switchyard_run_manifest_warn(
                        row,
                        status="failed",
                        artifact_root=artifact_root,
                        harbor_rc=1,
                    )
                    self._capture_and_drain_switchyard_warn(
                        conn,
                        row,
                        artifact_root,
                        status="failed",
                    )
                    self._write_provenance_warn(row, status="failed", artifact_root=artifact_root)
                    self._sync_artifacts_warn(
                        evaluation_id,
                        row["runtime"],
                        execution_id=execution_id,
                        execution_number=execution_number,
                    )
                    self._build_archive_warn(
                        evaluation_id,
                        runtime=row["runtime"],
                        artifact_root=artifact_root,
                        execution_number=execution_number,
                    )
                    self._set_status(
                        conn,
                        evaluation_id,
                        "failed",
                        execution_number=execution_number,
                        detail=detail,
                        failure_code=type(exc).__name__,
                    )
                    return
                if switchyard_lease is not None:
                    handle = _handle_with_switchyard(handle, switchyard_lease)
                if shared_campaign_id is not None:
                    SwitchyardCampaignRepository(conn).mark_launch_running(
                        evaluation_id,
                        handle,
                    )

            db_status = EvaluationRepository(conn).load_runtime_status(
                evaluation_id,
                expected_execution_number=execution_number,
            )
            if db_status is None:
                raise DispatchClaimLost(evaluation_id)
            if db_status == "cancelled":
                self._teardown_cancelled_runtime(
                    evaluation_id,
                    backend,
                    handle,
                    row=row,
                    artifact_root=artifact_root,
                )
                return
            if db_status in _DB_TERMINAL_STATUSES:
                return

            if row["status"] != "running":
                # --- running: poll backend until the Harbor job finishes ---
                self._set_status(
                    conn,
                    evaluation_id,
                    "running",
                    execution_number=execution_number,
                    detail=f"launched on {handle.backend}",
                    handle=json.dumps(handle.model_dump()),
                )
                row = {**row, "status": "running", "backend_handle": handle.model_dump()}

        try:
            status = self._await_terminal(
                evaluation_id,
                backend,
                handle,
                row=row,
                artifact_root=artifact_root,
            )
        except DispatchClaimLost:
            self._teardown_failed_runtime_warn(backend, handle)
            raise
        except Exception as exc:  # noqa: BLE001 — record any status-read failure
            with self.connect() as conn:
                switchyard_note = self._capture_and_drain_switchyard_warn(
                    conn,
                    row,
                    artifact_root,
                    handle=handle,
                    status="failed",
                )
                switchyard_manifest_note = self._write_switchyard_run_manifest_warn(
                    row,
                    status="failed",
                    artifact_root=artifact_root,
                    handle=handle,
                    harbor_rc=1,
                )
                self._write_provenance_warn(row, status="failed", artifact_root=artifact_root, handle=handle)
                self._sync_artifacts_warn(
                    evaluation_id,
                    row["runtime"],
                    execution_id=execution_id,
                    execution_number=execution_number,
                )
                self._build_archive_warn(
                    evaluation_id,
                    runtime=row["runtime"],
                    artifact_root=artifact_root,
                    execution_number=execution_number,
                )
                detail = f"status read failed: {exc}"
                self._mark_campaign_cleanup_pending(conn, row)
                teardown_note = self._teardown_failed_runtime_warn(backend, handle)
                if teardown_note is None:
                    self._acknowledge_campaign_cleanup(conn, row)
                if switchyard_note:
                    detail = f"{detail}; {switchyard_note}"
                if switchyard_manifest_note:
                    detail = f"{detail}; {switchyard_manifest_note}"
                if teardown_note:
                    detail = f"{detail}; {teardown_note}"
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=detail,
                    failure_code=type(exc).__name__,
                )
            return

        with self.connect() as conn:
            self._renew_inline_claim(evaluation_id)
            if status.phase == "cancelled":
                self._teardown_cancelled_runtime(
                    evaluation_id,
                    backend,
                    handle,
                    row=row,
                    artifact_root=artifact_root,
                )
                return

            if status.phase != "succeeded":
                raw_result = dict(status.raw)
                summary: ResultSummary | None = None
                try:
                    summary = backend.summarize(status.raw)
                except Exception as exc:  # noqa: BLE001 - preserve the original failure
                    LOG.warning(
                        "failed to summarize terminal error result for %s: %s",
                        evaluation_id,
                        exc,
                    )
                failure_code = status.failure_code
                if not failure_code and summary is not None and summary.exception_counts:
                    failure_code = next(
                        (code for code in summary.exception_counts if is_retryable_failure(code)),
                        next(iter(summary.exception_counts)),
                    )
                failure_code = failure_code or "unknown"
                switchyard_capture_note = self._capture_switchyard_warn(
                    row,
                    artifact_root,
                    handle=handle,
                )
                switchyard_manifest_note = self._write_switchyard_run_manifest_warn(
                    row,
                    status="failed",
                    artifact_root=artifact_root,
                    handle=handle,
                    harbor_rc=1,
                )
                detail = status.detail or "run reported failure"
                if switchyard_capture_note:
                    detail = f"{detail}; {switchyard_capture_note}"
                if switchyard_manifest_note:
                    detail = f"{detail}; {switchyard_manifest_note}"
                self._mark_campaign_cleanup_pending(conn, row)
                teardown_note = self._teardown_failed_runtime_warn(backend, handle)
                if teardown_note is None:
                    self._acknowledge_campaign_cleanup(conn, row)
                else:
                    detail = f"{detail}; {teardown_note}"
                if is_retryable_failure(failure_code, detail):
                    switchyard_retry_note = self._capture_and_drain_switchyard_warn(
                        conn,
                        row,
                        artifact_root,
                        handle=handle,
                        status="failed",
                    )
                    if switchyard_retry_note:
                        detail = f"{detail}; {switchyard_retry_note}"
                    scheduled = EvaluationRepository(conn).schedule_retry(
                        evaluation_id,
                        execution_number=execution_number,
                        failure_code=failure_code,
                        failure_category=failure_category_for_code(failure_code, detail),
                        delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
                        expected_dispatch_owner=self._expected_dispatch_owner,
                    )
                    if scheduled is not None:
                        return

                switchyard_note = self._capture_and_drain_switchyard_warn(
                    conn,
                    row,
                    artifact_root,
                    handle=handle,
                    status="failed",
                )
                provenance_note = self._write_provenance_warn(
                    row, status="failed", artifact_root=artifact_root, handle=handle
                )
                intake_note = self._maybe_upload_atif(conn, row)
                artifact_note = self._sync_artifacts_warn(
                    evaluation_id,
                    row["runtime"],
                    execution_id=execution_id,
                    execution_number=execution_number,
                )
                self._build_archive_warn(
                    evaluation_id,
                    runtime=row["runtime"],
                    artifact_root=artifact_root,
                    execution_number=execution_number,
                )
                for note in (switchyard_note, provenance_note, artifact_note, intake_note):
                    if note:
                        detail = f"{detail}; {note}"
                if raw_result:
                    raw_result, viewer_note = self._maybe_upload_harbor_viewer(
                        row,
                        result=raw_result,
                        artifact_root=artifact_root,
                    )
                    if viewer_note:
                        detail = f"{detail}; {viewer_note}"
                if summary is not None and raw_result:
                    self._persist_result(
                        conn,
                        evaluation_id,
                        raw_result,
                        summary,
                        terminal_status="failed",
                        status_detail=detail,
                        failure_code=failure_code,
                        execution_number=execution_number,
                    )
                    return
                self._set_status(
                    conn,
                    evaluation_id,
                    "failed",
                    execution_number=execution_number,
                    detail=detail,
                    failure_code=failure_code,
                )
                return

            # --- terminal success: artifact sync, optional Intake ATIF, then write-back ---
            switchyard_note = self._capture_and_drain_switchyard_warn(
                conn,
                row,
                artifact_root,
                handle=handle,
                status="succeeded",
            )
            switchyard_manifest_note = self._write_switchyard_run_manifest_warn(
                row,
                status="succeeded",
                artifact_root=artifact_root,
                handle=handle,
                harbor_rc=0,
            )
            provenance_note = self._write_provenance_warn(
                row, status="succeeded", artifact_root=artifact_root, handle=handle
            )
            intake_note = self._maybe_upload_atif(conn, row)
            artifact_note = self._sync_artifacts_warn(
                evaluation_id,
                row["runtime"],
                execution_id=execution_id,
                execution_number=execution_number,
            )
            self._build_archive_warn(
                evaluation_id,
                runtime=row["runtime"],
                artifact_root=artifact_root,
                execution_number=execution_number,
            )
            raw_result, harbor_viewer_note = self._maybe_upload_harbor_viewer(
                row,
                result=status.raw,
                artifact_root=artifact_root,
            )
            # The backend reduces its own framework-typed result to the generic
            # summary — the worker stays framework-agnostic.
            summary = backend.summarize(status.raw)
            self._mark_campaign_cleanup_pending(conn, row)
            self._acknowledge_campaign_cleanup(conn, row)
            cleanup_note = self._teardown_succeeded_sandbox_warn(
                evaluation_id,
                str(row["runtime"]),
                backend,
                handle,
            )
            # Result envelope + summary; artifact/intake notes ride in status_detail.
            extra_detail = "; ".join(
                note
                for note in [
                    switchyard_note,
                    switchyard_manifest_note,
                    provenance_note,
                    artifact_note,
                    intake_note,
                    harbor_viewer_note,
                    cleanup_note,
                ]
                if note
            )
            self._persist_result(
                conn,
                evaluation_id,
                raw_result,
                summary,
                execution_number=execution_number,
                extra_detail=extra_detail or None,
            )

    def _await_terminal(
        self,
        evaluation_id: str,
        backend: RuntimeBackend,
        handle: LaunchHandle,
        *,
        row: dict,
        artifact_root: Path,
    ) -> RuntimeStatus:
        """Poll ``backend.status`` until the run is terminal or polls run out.

        Returns the terminal :class:`RuntimeStatus`; a fake backend that reports
        a terminal phase on the first call never sleeps. The immutable framework
        profile may extend the polling window through its sandbox lifecycle
        timeout, but the wait remains finite. Exhausting the effective poll limit
        yields a synthetic ``failed`` status so the run can't stick at running.
        """
        max_polls = self._terminal_max_polls(row)
        live_log_signature: tuple[int, int] | None = None
        last_resource_sample_at = float("-inf")
        for _ in range(max_polls):
            execution_number = int(row.get("execution_number") or 1)
            db_status = self._heartbeat_and_load_runtime_status(
                evaluation_id,
                execution_number=execution_number,
            )
            if db_status == "cancelled":
                return RuntimeStatus(phase="cancelled", detail="cancelled")
            if db_status in _DB_TERMINAL_STATUSES:
                return RuntimeStatus(
                    phase="failed",
                    failure_code="evaluation_already_terminal",
                    detail=f"evaluation already terminal: {db_status}",
                )
            status = backend.status(handle)
            live_log_signature = _sync_live_log_warn(
                evaluation_id,
                str(row["runtime"]),
                handle.external_id,
                execution_number,
                live_log_signature,
            )
            self._capture_switchyard_warn(row, artifact_root, handle=handle)
            if status.phase in _TERMINAL_PHASES:
                return status
            now = time.monotonic()
            if now - last_resource_sample_at >= settings.resource_usage_sample_interval_seconds:
                self._capture_resource_usage(
                    evaluation_id,
                    execution_number=execution_number,
                    backend=backend,
                    handle=handle,
                )
                last_resource_sample_at = now
            self.sleep(self.poll_interval)
        return RuntimeStatus(
            phase="failed",
            failure_code="poll_timeout",
            detail="timed out waiting for the run to finish",
        )

    def _capture_resource_usage(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        backend: RuntimeBackend,
        handle: LaunchHandle,
    ) -> None:
        sampler = getattr(backend, "sample_resources", None)
        if not callable(sampler):
            return
        try:
            samples = list(sampler(handle))
            if not samples:
                return
            with self.connect() as conn:
                ResourceUsageRepository(conn).record_samples(
                    evaluation_id,
                    execution_number=execution_number,
                    samples=samples,
                )
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail the evaluation
            LOG.warning(
                "resource telemetry unavailable for %s execution %s: %s",
                evaluation_id,
                execution_number,
                redact_secret_text(str(exc)),
            )

    def _terminal_max_polls(self, row: Mapping[str, Any]) -> int:
        lifecycle_timeout = _profile_lifecycle_timeout_seconds(row)
        if lifecycle_timeout is None or self.poll_interval <= 0:
            return self.max_polls
        lifecycle_polls = math.ceil((lifecycle_timeout + _LIFECYCLE_TIMEOUT_GRACE_SECONDS) / self.poll_interval)
        return max(self.max_polls, lifecycle_polls)

    def _heartbeat_and_load_runtime_status(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
    ) -> str:
        with self.connect() as conn:
            repo = EvaluationRepository(conn)
            if self._expected_dispatch_owner is not None and not repo.heartbeat_claim(
                evaluation_id,
                worker_id=self._expected_dispatch_owner,
                expected_execution_number=execution_number,
            ):
                if self._claim_lost is not None:
                    self._claim_lost.set()
                raise DispatchClaimLost(evaluation_id)
            status = repo.load_runtime_status(
                evaluation_id,
                expected_execution_number=execution_number,
            )
            if status is None:
                raise DispatchClaimLost(evaluation_id)
            return status

    def _provision_shared_campaign(
        self,
        conn: psycopg.Connection,
        campaign: Mapping[str, Any],
        *,
        evaluation_id: str,
        profile_id: str,
        raw_config: Mapping[str, Any],
        credential_env: Mapping[str, str],
        artifact_root: Path,
    ) -> tuple[dict[str, Any], SwitchyardRender | None]:
        benchmark_run_id = str(campaign["benchmark_run_id"])
        claim_attempt = int(campaign["claim_attempt"])
        repo = SwitchyardCampaignRepository(conn)

        def persist_lease(lease: SwitchyardLease) -> None:
            if not repo.record_provisioning_lease(
                benchmark_run_id,
                worker_id=self.worker_id,
                claim_attempt=claim_attempt,
                lease=lease,
            ):
                raise RuntimeError("Switchyard campaign provisioning ownership was lost before apply")

        try:
            with self._maintain_campaign_provisioning_claim(
                benchmark_run_id,
                claim_attempt=claim_attempt,
            ):
                render = self.switchyard.provision(
                    evaluation_id=benchmark_run_id,
                    benchmark_run_id=benchmark_run_id,
                    profile_id=profile_id,
                    raw_config=raw_config,
                    credential_env=credential_env,
                    artifact_root=artifact_root,
                    persist_lease=persist_lease,
                )
        except Exception as exc:
            failed_lease = _switchyard_lease_from_artifact_root(artifact_root)
            if repo.mark_provision_failed(
                benchmark_run_id,
                worker_id=self.worker_id,
                claim_attempt=claim_attempt,
                detail=str(exc),
                lease=failed_lease,
            ):
                raise
            return (
                self._wait_for_campaign_ready(
                    conn,
                    benchmark_run_id,
                    evaluation_id=evaluation_id,
                ),
                None,
            )

        resource_row = repo.mark_ready(
            benchmark_run_id,
            worker_id=self.worker_id,
            claim_attempt=claim_attempt,
            lease=render.lease,
        )
        if resource_row is not None:
            return resource_row, render

        # Ownership may have moved while an external rollout was completing.
        # The database row is authoritative: a stale owner must never delete a
        # deterministic shared resource name that a newer owner may have adopted.
        current = repo.get(benchmark_run_id)
        if current is not None and current.get("cancel_requested_at") is not None:
            self.switchyard.delete(render.lease, artifact_root=artifact_root)
            raise RuntimeError("Switchyard campaign was cancelled during provisioning")
        if current is not None and current["status"] == "ready":
            self._ensure_shared_campaign_ready(current)
            return current, None
        return (
            self._wait_for_campaign_ready(
                conn,
                benchmark_run_id,
                evaluation_id=evaluation_id,
            ),
            None,
        )

    def _repair_shared_campaign(
        self,
        conn: psycopg.Connection,
        benchmark_run_id: str,
        *,
        detail: str,
        evaluation_id: str,
        profile_id: str,
        raw_config: Mapping[str, Any],
        credential_env: Mapping[str, str],
        artifact_root: Path,
    ) -> dict[str, Any] | None:
        claimed = SwitchyardCampaignRepository(conn).claim_ready_reprovisioning(
            benchmark_run_id,
            worker_id=self.worker_id,
            claim_seconds=self.claim_timeout,
            detail=detail,
        )
        if claimed is None:
            return None
        repaired, _render = self._provision_shared_campaign(
            conn,
            claimed,
            evaluation_id=evaluation_id,
            profile_id=profile_id,
            raw_config=raw_config,
            credential_env=credential_env,
            artifact_root=artifact_root,
        )
        return repaired

    def _wait_for_campaign_ready(
        self,
        conn: psycopg.Connection,
        benchmark_run_id: str,
        *,
        evaluation_id: str,
        repair: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        repo = SwitchyardCampaignRepository(conn)
        last_ready_error: Exception | None = None
        for _ in range(self.max_polls):
            campaign = repo.get(benchmark_run_id)
            if campaign is None:
                raise RuntimeError("Switchyard campaign disappeared during provisioning")
            if campaign["status"] == "ready" and campaign.get("cancel_requested_at") is None:
                try:
                    self._ensure_shared_campaign_ready(campaign)
                except RuntimeError as exc:
                    last_ready_error = exc
                    if (
                        repair is not None
                        and _switchyard_resources_are_missing(exc)
                        and (repaired := repair(str(exc))) is not None
                    ):
                        return repaired
                else:
                    return campaign
            if campaign.get("cancel_requested_at") is not None:
                raise RuntimeError("Switchyard campaign was cancelled during provisioning")
            if campaign["status"] == "provision_failed":
                raise RuntimeError(f"Switchyard campaign provisioning failed: {campaign.get('evidence_error')}")
            self._renew_inline_claim(evaluation_id)
            self.sleep(self.poll_interval)
        detail = "timed out waiting for shared Switchyard readiness"
        if last_ready_error is not None:
            detail = f"{detail}: {last_ready_error}"
        raise TimeoutError(detail)

    def _wait_for_campaign_permit(
        self,
        conn: psycopg.Connection,
        benchmark_run_id: str,
        *,
        evaluation_id: str,
        execution_number: int,
        repair: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> bool:
        repo = SwitchyardCampaignRepository(conn)
        last_ready_error: Exception | None = None
        for _ in range(self.max_polls):
            campaign = repo.get(benchmark_run_id)
            if campaign is None or campaign["status"] != "ready" or campaign.get("cancel_requested_at") is not None:
                if campaign is not None and campaign.get("cancel_requested_at") is not None:
                    raise RuntimeError("Switchyard campaign cancelled before member launch")
                raise RuntimeError("Switchyard campaign is not ready for member launch")
            try:
                self._ensure_shared_campaign_ready(campaign)
            except RuntimeError as exc:
                last_ready_error = exc
                if repair is not None and _switchyard_resources_are_missing(exc) and repair(str(exc)) is not None:
                    continue
                self._renew_inline_claim(evaluation_id)
                self.sleep(self.poll_interval)
                continue
            decision = repo.acquire_launch_permit(
                benchmark_run_id=benchmark_run_id,
                evaluation_id=evaluation_id,
                worker_id=self.worker_id,
                lease_seconds=max(self.claim_timeout, self.poll_interval * 3),
            )
            if decision == "launch":
                return True
            if decision == "resume":
                return False
            status = EvaluationRepository(conn).load_runtime_status(
                evaluation_id,
                expected_execution_number=execution_number,
            )
            if status is None:
                raise DispatchClaimLost(evaluation_id)
            campaign = repo.get(benchmark_run_id)
            if status == "cancelled" or (campaign is not None and campaign.get("cancel_requested_at") is not None):
                raise RuntimeError("Switchyard campaign cancelled before member launch")
            if campaign is None or campaign["status"] != "ready":
                raise RuntimeError("Switchyard campaign is not ready for member launch")
            self._renew_inline_claim(evaluation_id)
            self.sleep(self.poll_interval)
        detail = "timed out waiting for a Switchyard campaign member permit"
        if last_ready_error is not None:
            detail = f"{detail}: {last_ready_error}"
        raise TimeoutError(detail)

    def _ensure_shared_campaign_ready(
        self,
        campaign: Mapping[str, Any],
    ) -> None:
        lease = _campaign_lease_from_row(campaign)
        if lease is None:
            raise RuntimeError("shared Switchyard campaign lease metadata missing or invalid")
        try:
            self.switchyard.ensure_ready(lease)
        except Exception as exc:  # noqa: BLE001 — preserve the provider's readiness detail
            raise RuntimeError(f"shared Switchyard campaign resources unavailable: {exc}") from exc

    def _teardown_cancelled_runtime(
        self,
        evaluation_id: str,
        backend: RuntimeBackend,
        handle: LaunchHandle,
        *,
        row: dict,
        artifact_root: Path,
    ) -> None:
        with self.connect() as conn:
            self._mark_campaign_cleanup_pending(conn, row)
        try:
            backend.teardown(handle)
        except Exception as exc:  # noqa: BLE001 — cancellation must stay durable
            with self.connect() as conn:
                EvaluationRepository(conn).record_cancel_teardown_failure(
                    evaluation_id,
                    f"cancelled; evaluation-runtime cleanup failed: {exc}",
                )
        else:
            with self.connect() as conn:
                self._acknowledge_campaign_cleanup(conn, row)
                self._capture_and_drain_switchyard_warn(
                    conn,
                    row,
                    artifact_root,
                    handle=handle,
                    status="cancelled",
                )
                self._write_switchyard_run_manifest_warn(
                    row,
                    status="cancelled",
                    artifact_root=artifact_root,
                    handle=handle,
                    harbor_rc=1,
                )
                self._write_provenance_warn(
                    row,
                    status="cancelled",
                    artifact_root=artifact_root,
                    handle=handle,
                )
                self._sync_artifacts_warn(
                    evaluation_id,
                    handle.backend,
                    execution_id=str(row.get("execution_id") or handle.external_id),
                    execution_number=int(row.get("execution_number") or 1),
                )
                EvaluationRepository(conn).record_cancel_teardown_succeeded(evaluation_id)
                self._build_archive_warn(
                    evaluation_id,
                    runtime=handle.backend,
                    artifact_root=artifact_root,
                    execution_number=int(row.get("execution_number") or 1),
                )

    @staticmethod
    def _teardown_failed_runtime_warn(
        backend: RuntimeBackend,
        handle: LaunchHandle,
    ) -> str | None:
        """Best-effort cleanup after a launched runtime fails or times out."""
        try:
            backend.teardown(handle)
        except Exception as exc:  # noqa: BLE001 — preserve the original failure
            detail = f"evaluation-runtime cleanup failed: {exc}"
            LOG.warning("%s for %s", detail, handle.external_id)
            return detail
        return None

    def _teardown_succeeded_sandbox_warn(
        self,
        evaluation_id: str,
        runtime: str,
        backend: RuntimeBackend,
        handle: LaunchHandle,
    ) -> str | None:
        """Best-effort bounded cleanup for completed Kubernetes sandboxes."""
        if runtime != "sandbox_k8s":
            return None
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                backend.teardown(handle)
                return None
            except Exception as exc:  # noqa: BLE001 — success must remain terminal
                detail = (
                    "evaluation-runtime cleanup failed "
                    f"for {evaluation_id} resource {handle.external_id} "
                    f"(attempt {attempt}/{attempts}): {exc}"
                )
                LOG.warning("%s", detail)
                if attempt < attempts:
                    self.sleep(float(attempt))
        return detail

    def _resume_handle(self, row: dict) -> LaunchHandle:
        raw_handle = row.get("backend_handle")
        if raw_handle:
            try:
                data = json.loads(raw_handle) if isinstance(raw_handle, str) else raw_handle
                return LaunchHandle(
                    backend=data.get("backend", row["runtime"]),
                    external_id=data.get("external_id", row["id"]),
                    raw=data.get("raw") if isinstance(data.get("raw"), dict) else {},
                )
            except Exception:  # noqa: BLE001 — fall back to legacy id-as-handle rows
                LOG.warning("invalid backend_handle for %s; falling back to evaluation id", row["id"])
        return LaunchHandle(backend=row["runtime"], external_id=row["id"])

    @staticmethod
    def _load_switchyard_config(conn: psycopg.Connection, row: dict) -> dict:
        """Load the selected Switchyard profile's non-secret config."""
        snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
        return Dispatcher._profile_config(conn, row, snapshot=snapshot, role="switchyard")

    @staticmethod
    def _load_intake_config(conn: psycopg.Connection, row: dict) -> dict:
        """Load the selected Intake profile's non-secret config."""
        snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
        return Dispatcher._profile_config(conn, row, snapshot=snapshot, role="intake")

    def _capture_switchyard_warn(
        self,
        row: dict,
        artifact_root: Path,
        *,
        handle: LaunchHandle | None = None,
        final: bool = False,
        session_ids: tuple[str, ...] = (),
    ) -> str | None:
        lease = _switchyard_lease_from(row, handle=handle)
        if lease is None:
            return None
        try:
            return self.switchyard.capture(
                lease,
                artifact_root,
                final=final,
                session_ids=session_ids,
            )
        except Exception as exc:  # noqa: BLE001 — logs are best-effort
            LOG.warning("switchyard capture failed for %s: %s", row["id"], exc)
            return f"switchyard capture failed: {exc}"

    def _capture_and_drain_switchyard_warn(
        self,
        conn: psycopg.Connection,
        row: dict,
        artifact_root: Path,
        *,
        handle: LaunchHandle | None = None,
        status: str,
        drain_seconds_override: float | None = None,
        capture: bool = True,
    ) -> str | None:
        lease = _switchyard_lease_from(row, handle=handle)
        if lease is None:
            return None

        session_ids = _switchyard_capture_session_ids(artifact_root, str(row["id"]))
        # Shared campaign evidence and draining are owned by the durable
        # benchmark finalizer after every member runtime has stopped. Capture
        # this member's session before its post-run Intake upload, without
        # draining the shared gateway.
        if row.get("switchyard_topology") == "shared_campaign":
            return self._capture_switchyard_warn(
                row,
                artifact_root,
                handle=handle,
                final=True,
                session_ids=session_ids,
            )

        notes: list[str] = []
        if capture and (
            capture_note := self._capture_switchyard_warn(
                row,
                artifact_root,
                handle=handle,
                final=True,
                session_ids=session_ids,
            )
        ):
            notes.append(capture_note)
            self._append_switchyard_event(
                conn,
                row["id"],
                status=status,
                detail=capture_note,
            )

        if lease.mode == "external":
            return "; ".join(notes) if notes else None

        drain_seconds = (
            drain_seconds_override
            if drain_seconds_override is not None
            else (lease.drain_seconds if lease.drain_seconds is not None else settings.switchyard_drain_seconds)
        )
        try:
            resource_row = RuntimeResourceRepository(conn).mark_switchyard_draining(
                row["id"],
                int(row.get("execution_number") or row.get("current_execution") or 1),
                drain_seconds=drain_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — terminal status must still be recorded
            detail = f"switchyard drain mark failed: {exc}"
            LOG.warning("%s for %s", detail, row["id"])
            self._append_switchyard_event(conn, row["id"], status=status, detail=detail)
            notes.append(detail)
            return "; ".join(notes)

        if resource_row is None:
            return "; ".join(notes) if notes else None

        row["switchyard_resource"] = resource_row
        row["switchyard_drain_until"] = resource_row.get("drain_until")
        detail = _switchyard_drain_detail(lease, resource_row)
        self._append_switchyard_event(conn, row["id"], status=status, detail=detail)
        notes.append(detail)
        return "; ".join(notes)

    @staticmethod
    def _mark_campaign_cleanup_pending(conn: psycopg.Connection, row: Mapping[str, Any]) -> None:
        if row.get("switchyard_topology") == "shared_campaign":
            SwitchyardCampaignRepository(conn).mark_cleanup_pending(str(row["id"]))

    @staticmethod
    def _acknowledge_campaign_cleanup(conn: psycopg.Connection, row: Mapping[str, Any]) -> None:
        if row.get("switchyard_topology") == "shared_campaign":
            SwitchyardCampaignRepository(conn).acknowledge_cleanup(str(row["id"]))

    @staticmethod
    def _append_switchyard_event(
        conn: psycopg.Connection,
        evaluation_id: str,
        *,
        status: str,
        detail: str | None,
    ) -> None:
        try:
            EvaluationRepository(conn).append_event(
                evaluation_id,
                status=status,
                detail=detail,
                type="switchyard",
            )
        except Exception:  # noqa: BLE001 — observability event must not break dispatch
            LOG.exception("failed to append switchyard event for %s", evaluation_id)

    def _load_status_runtime(self, evaluation_id: str) -> dict | None:
        with self.connect() as conn:
            return EvaluationRepository(conn).load_status_runtime(evaluation_id)

    @staticmethod
    def _event_status_for(conn: psycopg.Connection, evaluation_id: str) -> str:
        row = EvaluationRepository(conn).load_status_runtime(evaluation_id)
        if row is None:
            return "succeeded"
        return str(row["status"])

    @staticmethod
    def _load_harbor_config(conn: psycopg.Connection, row: dict) -> dict:
        """Load the selected Harbor profile's non-secret config for runtime render."""
        if row.get("framework") != "harbor":
            return {}
        snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
        return Dispatcher._profile_config(conn, row, snapshot=snapshot, role="harbor")

    @staticmethod
    def _load_framework_config(conn: psycopg.Connection, row: dict) -> dict:
        """Load generic framework configuration from its immutable snapshot."""
        snapshot = validate_execution_snapshot(row.get("execution_snapshot"))
        return Dispatcher._profile_config(conn, row, snapshot=snapshot, role="framework")

    @staticmethod
    def _profile_config(
        conn: psycopg.Connection,
        row: Mapping[str, Any],
        *,
        snapshot: Mapping[str, Any] | None,
        role: str,
    ) -> dict[str, Any]:
        profile_id = row.get(f"{role}_profile_id")
        if role == "framework" or (role == "harbor" and not profile_id and row.get("framework") == "harbor"):
            profile_id = row.get("framework_profile_id")
        if not profile_id:
            return {}
        if snapshot is not None:
            snap_role = role
            profiles = snapshot.get("profiles") or {}
            if role == "framework" and snap_role not in profiles and row.get("framework") == "harbor":
                snap_role = "harbor"
            if role == "harbor" and snap_role not in profiles and row.get("framework") == "harbor":
                snap_role = "framework"
            config = snapshot_profile_config(snapshot, snap_role)
            if not config and snap_role not in profiles:
                raise RuntimeError(f"{role} profile missing from execution snapshot")
            return config
        loader = getattr(EvaluationRepository(conn), f"load_{role}_profile")
        profile_row = loader(str(profile_id))
        if profile_row is None:
            raise RuntimeError(f"{role} profile not found: {profile_id}")
        config = profile_row.get("config") or {}
        if not isinstance(config, dict):
            raise RuntimeError(f"{role} profile {profile_id} config must be a JSON object")
        return config

    @staticmethod
    def _maybe_upload_harbor_viewer(
        row: Mapping[str, Any],
        *,
        result: Mapping[str, Any],
        artifact_root: Path,
    ) -> tuple[dict[str, Any], str | None]:
        """Persist a Viewer archive and best-effort upload the finished job."""
        if (row.get("framework") or "harbor") != "harbor":
            return dict(result), None
        try:
            publication = publish_harbor_job_archive(
                job_name=str(row["id"]),
                job_dir=artifact_root,
            )
        except Exception as exc:  # noqa: BLE001 — viewer upload must not fail the run
            LOG.warning("Harbor Viewer archive publication failed for %s: %s", row["id"], exc)
            return dict(result), f"harbor viewer archive publication failed: {exc}"
        if publication is None:
            return dict(result), None
        published_result = result_with_harbor_viewer_publication(result, publication)
        if publication.viewer_url:
            note = f"harbor viewer: {publication.viewer_url}"
        elif publication.upload_error:
            LOG.warning(
                "Harbor Viewer automatic upload failed for %s: %s",
                row["id"],
                publication.upload_error,
            )
            note = f"harbor viewer automatic upload failed: {publication.upload_error}; manual archive ready"
        else:
            note = "harbor viewer archive ready for manual upload"
        return published_result, note

    def _maybe_upload_atif(self, conn: psycopg.Connection, row: dict) -> str | None:
        """Upload Harbor ATIF trajectories when ``intake_profile_id`` is set.

        Reads ``trajectory.json`` files from the finished Harbor job dir and POSTs
        them to NMP Intake. Returns a human-readable note for ``status_detail``,
        or ``None`` on success/no-op. Failures are warnings unless
        ``INTAKE_FAIL_ON_ERROR`` is set.
        """
        profile_id = row.get("intake_profile_id")
        if not profile_id:
            return None

        config = row.get("intake_config")
        if not isinstance(config, dict):
            profile_row = EvaluationRepository(conn).load_intake_profile(profile_id)
            if profile_row is None:
                LOG.warning("intake profile %s not found for %s", profile_id, row["id"])
                return f"intake profile not found: {profile_id}"
            config = profile_row["config"] or {}

        target = resolve_intake_target(
            config,
            task_slug=row.get("task_slug"),
            base_url=settings.intake_base_url,
            source=settings.intake_source,
        )
        return upload_job_atif_warn(
            _artifact_root(row.get("execution_id") or row["id"], row["runtime"]),
            target,
            evaluation_run_id=row["id"],
            experiment=self._experiment_request(conn, row),
            test_case_id=row.get("task_slug"),
            timeout=settings.intake_timeout_seconds,
            fail_on_error=settings.intake_fail_on_error,
        )

    @staticmethod
    def _experiment_request(conn: psycopg.Connection, row: Mapping[str, Any]) -> ExperimentRequest:
        """Build the run's intake Experiment identity.

        Experiment = benchmark run (group + dataset = the benchmark); a standalone
        evaluation (no ``benchmark_run_id``) is its own Experiment keyed by the task.
        """
        benchmark_run_id = row.get("benchmark_run_id")
        metadata: dict[str, Any] = {
            "evaluation_id": row.get("id"),
            "framework": row.get("framework"),
            "framework_version": row.get("framework_version"),
            "task_slug": row.get("task_slug"),
            "benchmark_run_id": benchmark_run_id,
        }
        if benchmark_run_id:
            identity = BenchmarkRunRepository(conn).load_experiment_identity(benchmark_run_id) or {}
            benchmark = identity.get("benchmark_slug") or row.get("task_slug") or "eval"
            metadata["benchmark_run_name"] = identity.get("run_name")
            run_key = str(benchmark_run_id)
        else:
            benchmark = row.get("task_slug") or "eval"
            run_key = str(row["id"])
        return ExperimentRequest(
            benchmark=str(benchmark),
            run_key=run_key,
            metadata=str_metadata(metadata),
            group_metadata=str_metadata({"benchmark": benchmark}),
        )

    @staticmethod
    def _write_provenance_warn(
        row: dict,
        *,
        status: str,
        artifact_root: Path,
        handle: LaunchHandle | None = None,
    ) -> str | None:
        """Write per-run provenance into the local artifact tree, warning on errors."""
        try:
            write_run_provenance_manifest(
                artifact_root,
                row,
                status=status,
                artifact_prefix=s3.evaluation_artifact_prefix(row["id"]),
                backend=None if handle is None else handle.backend,
                handle=None if handle is None else handle.external_id,
            )
        except Exception as exc:  # noqa: BLE001 — terminal status must still be recorded
            LOG.warning("provenance manifest failed for %s: %s", row["id"], exc)
            return f"provenance manifest failed: {exc}"
        return None

    @staticmethod
    def _write_switchyard_run_manifest_warn(
        row: dict,
        *,
        status: str,
        artifact_root: Path,
        handle: LaunchHandle | None = None,
        harbor_rc: int | None = None,
    ) -> str | None:
        """Write the Switchyard benchmark-compatible manifest, warning on errors."""
        try:
            write_switchyard_run_manifest(
                artifact_root,
                row,
                status=status,
                backend=None if handle is None else handle.backend,
                handle=None if handle is None else handle.external_id,
                harbor_rc=harbor_rc,
            )
        except Exception as exc:  # noqa: BLE001 — terminal status must still be recorded
            LOG.warning("switchyard run manifest failed for %s: %s", row["id"], exc)
            return f"switchyard run manifest failed: {exc}"
        return None

    def _sync_artifacts_warn(
        self,
        evaluation_id: str,
        runtime: str,
        *,
        execution_id: str | None = None,
        execution_number: int | None = None,
        replace: bool = True,
    ) -> str | None:
        """Upload local evaluation artifacts to object storage, warning on errors."""
        source_id = execution_id or evaluation_id
        try:
            artifact_root = _artifact_root(source_id, runtime)
            if execution_number is not None:
                self._record_portable_telemetry_warn(
                    evaluation_id,
                    execution_number=execution_number,
                    artifact_root=artifact_root,
                )
            _copy_runtime_logs_to_artifact_root(source_id, runtime)
            artifact_prefix = s3.evaluation_artifact_prefix(evaluation_id)
            if execution_number is None:
                count = s3.sync_directory_to_prefix(artifact_root, artifact_prefix)
            else:
                with self.connect() as conn, conn.transaction():
                    if not EvaluationRepository(conn).lock_current_execution(
                        evaluation_id,
                        expected_execution_number=execution_number,
                    ):
                        raise DispatchClaimLost(evaluation_id)
                    sync = (
                        s3.replace_directory_at_prefix
                        if replace and execution_number > 1
                        else s3.sync_directory_to_prefix
                    )
                    count = sync(artifact_root, artifact_prefix)
                    ExecutionTelemetryRepository(conn).record_artifact_sync(
                        evaluation_id,
                        execution_number=execution_number,
                        status="succeeded",
                        file_count=count,
                    )
        except DispatchClaimLost:
            raise
        except Exception as exc:  # noqa: BLE001 — terminal status must still be recorded
            LOG.warning("artifact sync failed for %s: %s", evaluation_id, exc)
            if execution_number is not None:
                try:
                    with self.connect() as conn, conn.transaction():
                        ExecutionTelemetryRepository(conn).record_artifact_sync(
                            evaluation_id,
                            execution_number=execution_number,
                            status="failed",
                            error=str(exc),
                        )
                except Exception:  # noqa: BLE001 - preserve the primary sync warning
                    LOG.exception("failed to persist artifact sync telemetry for %s", evaluation_id)
            return f"artifact sync failed: {exc}"
        return f"artifacts: uploaded {count} files" if count else None

    def _record_portable_telemetry_warn(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        artifact_root: Path,
    ) -> None:
        """Persist facts derived from raw artifacts and the Intake handoff diagnostic."""
        try:
            summary = summarize_job_telemetry(
                artifact_root,
                evaluation_run_id=evaluation_id,
            )
            diagnostic: dict[str, Any] = {}
            diagnostic_path = artifact_root / "intake-upload.json"
            if diagnostic_path.exists():
                loaded = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                diagnostic = loaded if isinstance(loaded, dict) else {}
            with self.connect() as conn, conn.transaction():
                row = EvaluationRepository(conn).load_for_dispatch(evaluation_id)
                if row is None:
                    return
                repository = ExecutionTelemetryRepository(conn)
                repository.record_summary(
                    evaluation_id,
                    execution_number=execution_number,
                    summary=summary,
                )
                profile_id = row.get("intake_profile_id")
                if profile_id:
                    experiment = self._experiment_request(conn, row)
                    experiment_ref = build_experiment_name(
                        experiment.benchmark,
                        experiment.run_key,
                    )
                    diagnostic_status = str(diagnostic.get("status") or "pending")
                    if diagnostic_status not in {"succeeded", "failed", "no_records"}:
                        diagnostic_status = "pending"
                    repository.record_intake(
                        evaluation_id,
                        execution_number=execution_number,
                        experiment_ref=experiment_ref,
                        run_refs=list(summary["intake_run_refs"]),
                        status=diagnostic_status,
                        expected_records=int(summary["intake_expected_records"]),
                        uploaded_records=(
                            int(diagnostic["uploaded"]) if isinstance(diagnostic.get("uploaded"), int) else None
                        ),
                        error=(str(diagnostic["error"]) if diagnostic.get("error") else None),
                    )
                else:
                    repository.record_intake(
                        evaluation_id,
                        execution_number=execution_number,
                        experiment_ref=None,
                        run_refs=[],
                        status="disabled",
                        expected_records=None,
                        uploaded_records=None,
                        error=None,
                    )
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail the evaluation
            LOG.warning("portable telemetry unavailable for %s: %s", evaluation_id, exc)

    def _build_archive_warn(
        self,
        evaluation_id: str,
        *,
        runtime: str | None = None,
        already_claimed: bool = False,
        artifact_root: Path | None = None,
        execution_number: int | None = None,
    ) -> str | None:
        """Build the downloadable results archive, warning on errors."""
        try:
            if runtime is not None and not get_backend_capabilities(runtime).supports_archive:
                with self.connect() as conn:
                    EvaluationRepository(conn).mark_archive_missing(
                        evaluation_id,
                        detail=f"runtime {runtime} does not support result archives",
                        expected_execution_number=execution_number,
                    )
                return None
            if not already_claimed:
                with self.connect() as conn:
                    EvaluationRepository(conn).mark_archive_building(
                        evaluation_id,
                        worker_id=self.worker_id,
                        expected_execution_number=execution_number,
                    )
            if execution_number is None:
                if artifact_root is not None:
                    archive = s3.build_evaluation_archive_from_directory(
                        evaluation_id,
                        artifact_root,
                    )
                else:
                    archive = s3.build_evaluation_archive(evaluation_id)
                with self.connect() as conn:
                    EvaluationRepository(conn).mark_archive_ready(
                        evaluation_id,
                        object_key=archive["object_key"],
                        size_bytes=archive["size_bytes"],
                    )
            else:
                with self.connect() as conn, conn.transaction():
                    if not EvaluationRepository(conn).lock_current_execution(
                        evaluation_id,
                        expected_execution_number=execution_number,
                    ):
                        raise DispatchClaimLost(evaluation_id)
                    if artifact_root is not None:
                        archive = s3.build_evaluation_archive_from_directory(
                            evaluation_id,
                            artifact_root,
                        )
                    else:
                        archive = s3.build_evaluation_archive(evaluation_id)
                    EvaluationRepository(conn).mark_archive_ready(
                        evaluation_id,
                        object_key=archive["object_key"],
                        size_bytes=archive["size_bytes"],
                        expected_execution_number=execution_number,
                    )
        except DispatchClaimLost:
            raise
        except Exception as exc:  # noqa: BLE001 — terminal status must still be recorded
            detail = str(exc)
            LOG.warning("archive build failed for %s: %s", evaluation_id, detail)
            try:
                with self.connect() as conn:
                    EvaluationRepository(conn).mark_archive_missing(
                        evaluation_id,
                        detail=detail,
                        expected_execution_number=execution_number,
                    )
            except Exception:  # noqa: BLE001 — keep original archive failure as the warning
                LOG.exception("failed to record archive build failure for %s", evaluation_id)
            return f"archive build failed: {detail}"
        return f"archive: uploaded {s3.ARCHIVE_FILE_NAME}"

    @staticmethod
    def _load(conn: psycopg.Connection, evaluation_id: str) -> dict | None:
        return EvaluationRepository(conn).load_for_dispatch(evaluation_id)

    def _set_status(
        self,
        conn: psycopg.Connection,
        evaluation_id: str,
        status: str,
        *,
        execution_number: int,
        detail: str | None = None,
        handle: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        if self._claim_lost is not None and self._claim_lost.is_set():
            raise DispatchClaimLost(evaluation_id)
        telemetry = ExecutionTelemetryRepository(conn)
        if status in {"provisioning", "running"}:
            telemetry.record_phase(
                evaluation_id,
                execution_number=execution_number,
                phase=status,
            )
        elif status in _DB_TERMINAL_STATUSES:
            telemetry.record_phase(
                evaluation_id,
                execution_number=execution_number,
                phase="terminal",
                terminal_status=status,
            )
        updated = EvaluationRepository(conn).set_status(
            evaluation_id,
            status,
            detail=detail,
            handle=handle,
            failure_code=failure_code,
            expected_dispatch_owner=self._expected_dispatch_owner,
            expected_execution_number=execution_number,
        )
        if not updated:
            if self._claim_lost is not None:
                self._claim_lost.set()
            raise DispatchClaimLost(evaluation_id)

    def _persist_result(
        self,
        conn: psycopg.Connection,
        evaluation_id: str,
        result: dict,
        summary: ResultSummary,
        *,
        execution_number: int,
        extra_detail: str | None = None,
        terminal_status: str = "succeeded",
        status_detail: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        """Write a trustworthy terminal envelope + derived summary onto the row.

        ``result`` is the framework-typed result stored verbatim as ``result``
        JSONB; ``summary`` is the generic reduction the backend produced (see
        :meth:`RuntimeBackend.summarize`). ``extra_detail`` (e.g. an Intake upload
        note from :meth:`_maybe_upload_atif`) is appended to the trial-count
        ``status_detail`` when present.
        """
        if self._claim_lost is not None and self._claim_lost.is_set():
            raise DispatchClaimLost(evaluation_id)
        ExecutionTelemetryRepository(conn).record_phase(
            evaluation_id,
            execution_number=execution_number,
            phase="terminal",
            terminal_status=terminal_status,
        )
        updated = EvaluationRepository(conn).persist_result(
            evaluation_id,
            EvaluationResultWrite(result=result, summary=summary, extra_detail=extra_detail),
            terminal_status=terminal_status,
            status_detail=status_detail,
            failure_code=failure_code,
            expected_dispatch_owner=self._expected_dispatch_owner,
            expected_execution_number=execution_number,
        )
        if not updated:
            if self._claim_lost is not None:
                self._claim_lost.set()
            raise DispatchClaimLost(evaluation_id)


def _switchyard_config_for_network_policy(config: Mapping[str, Any], network_policy: str) -> dict[str, Any]:
    """Render Switchyard reachability without overriding unrestricted egress."""
    rendered = dict(config)
    if rendered.get("mode", "managed") == "external":
        return rendered
    rendered["sandbox_egress_network_policy"] = network_policy != "unrestricted"
    return rendered


def _runner_env_with_switchyard(
    credential_env: Mapping[str, str],
    switchyard_env: Mapping[str, str],
) -> dict[str, str]:
    """Preserve non-model env while routing all model calls through Switchyard."""
    env = {key: value for key, value in credential_env.items() if key not in _MODEL_ROUTE_ENV_KEYS}
    env.update({str(key): str(value) for key, value in switchyard_env.items()})
    return env


def _row_with_switchyard(
    row: Mapping[str, Any],
    lease: SwitchyardLease | None,
    resource_row: Mapping[str, Any] | None = None,
) -> dict:
    updated = dict(row)
    if lease is not None:
        updated["switchyard"] = lease.model_dump(mode="json", exclude_none=True)
    if resource_row is not None:
        updated["switchyard_resource"] = dict(resource_row)
        updated["switchyard_drain_until"] = resource_row.get("drain_until")
    return updated


def _handle_with_switchyard(handle: LaunchHandle, lease: SwitchyardLease) -> LaunchHandle:
    raw = dict(handle.raw)
    raw["switchyard"] = lease.model_dump(mode="json", exclude_none=True)
    return handle.model_copy(update={"raw": raw})


def _switchyard_lease_from(
    row: Mapping[str, Any],
    *,
    handle: LaunchHandle | None = None,
) -> SwitchyardLease | None:
    for value in (
        row.get("switchyard"),
        row.get("switchyard_lease"),
        (row.get("switchyard_resource") or {}).get("metadata")
        if isinstance(row.get("switchyard_resource"), Mapping)
        else None,
        None if handle is None else handle.raw.get("switchyard"),
    ):
        lease = _parse_switchyard_lease(value)
        if lease is not None:
            return lease
    return None


def _parse_switchyard_lease(value: Any) -> SwitchyardLease | None:
    if value is None:
        return None
    if isinstance(value, SwitchyardLease):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    try:
        return SwitchyardLease.model_validate(value)
    except Exception:  # noqa: BLE001 — invalid persisted metadata is handled by callers
        return None


def _campaign_lease_from_row(row: Mapping[str, Any] | None) -> SwitchyardLease | None:
    if row is None:
        return None
    return _parse_switchyard_lease(row.get("metadata"))


def _switchyard_resources_are_missing(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "notfound" in detail or "not found" in detail or "lease metadata missing" in detail


def _switchyard_lease_from_artifact_root(artifact_root: Path) -> SwitchyardLease | None:
    lease_path = artifact_root / "switchyard" / "lease.json"
    try:
        return SwitchyardLease.model_validate_json(lease_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stable_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _switchyard_drain_detail(lease: SwitchyardLease, resource_row: Mapping[str, Any]) -> str:
    drain_until = resource_row.get("drain_until")
    if hasattr(drain_until, "isoformat"):
        drain_text = drain_until.isoformat()
    elif drain_until is not None:
        drain_text = str(drain_until)
    else:
        drain_text = "unknown"
    return f"switchyard draining until {drain_text}: {lease.name}"


def get_dispatcher() -> Dispatcher:
    """FastAPI dependency yielding the production dispatcher.

    Overridden in tests (via ``dependency_overrides``) to inject a fake
    backend and a fake connection.
    """
    return Dispatcher()
