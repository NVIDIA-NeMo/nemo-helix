# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the evals plugin."""

from __future__ import annotations

import hashlib
import math
import pickle
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from nemo_evals.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    MetricInline,
    MetricRef,
    TaskInput,
    TaskInputs,
)
from nemo_evals.api.service.task_service import TaskService
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash, ResolvedHarborTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
from nemo_evals.config import HarborTaskJobsConfig, get_config
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.harbor.remote.preparation import (
    ImagePreparation,
    ImageRole,
    PreparationClients,
    PreparationErrorCode,
    PreparationErrorInfo,
    TaskPreparation,
)
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy
from nemo_evals.harbor.remote.sandbox_policy import resolve_sandbox_policy
from nemo_evals.jobs.agent_spec import AgentEvalSpec, HarborBuiltinAgentSource, HarborRunnerTarget, ResolvedTask
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.revisions import get_revision
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager, CloudpickleMetricPayload
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError, NemoHTTPError, PermissionDeniedError
from nemo_helix_plugin.config import NemoHelixConfig, get_platform_config
from nemo_helix_plugin.entities import (
    DEFAULT_WORKSPACE,
    EntityBase,
    EntityClient,
    ListResponse,
    PaginationInfo,
    parse_qualified_name,
)
from nemo_helix_plugin.entities.types import DeleteResponse
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.in_memory_filter import InMemoryFilterRepository
from nemo_helix_plugin.intake.types import EvaluationCreateRequest, EvaluationPatchRequest, EvaluationResponse
from nhx.testing.files import InMemoryFilesService
from nhx.testing.secrets import InMemorySecretsService
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric


@pytest.fixture
def files_service() -> InMemoryFilesService:
    """Return fresh isolated Files storage for each test's resource assertions."""
    return InMemoryFilesService()


@pytest.fixture
def files_client(files_service: InMemoryFilesService) -> AsyncFilesClient:
    """Return a real Files client backed by the test's in-memory Files service."""
    return files_service.client()


def matches_filter(entity, operation) -> bool:
    """Evaluate a filter the way the entity store does: base fields as columns, the rest under ``data``.

    Filters are *evaluated*, not ignored: a fake that returned every row would make a revision
    lookup that forgot its ``parent`` predicate look correct, and cross-record confusion is exactly
    the bug that predicate prevents.
    """
    if operation is None:
        return True
    row = {
        "name": entity.name,
        "workspace": entity.workspace,
        "project": entity.project,
        "parent": entity.parent,
        "created_by": entity.created_by,
        "data": entity.model_dump(mode="json", exclude=set(entity.__base_fields__)),
    }
    return operation.apply(InMemoryFilterRepository(row))


class FakeEntityStore(EntityClient):
    """In-memory entity store standing in for ``NemoEntitiesClient``.

    Two behaviors are reproduced deliberately, because service logic depends on them:

    - **Parent-scoped uniqueness.** Records key on ``(entity_type, workspace, name, parent)``, which
      is what lets revision children of different tasks both be named ``rev.1`` while making a
      duplicate ordinal under one parent conflict.
    - **Optimistic locking.** ``update`` rejects a write whose ``db_version`` is stale. A fake that
      accepted every update would make correct retry logic untestable and broken retry logic look
      fine.
    - **Copy-on-read and copy-on-write.** Every record crosses the boundary as a deep copy, because
      the real client serializes over HTTP and cannot share objects with its caller. A fake that
      handed back the stored instance would let a service mutate the store just by touching an
      entity it read — so staging changes in memory and writing them later would be
      indistinguishable from writing them immediately, and a missing write would still pass.

    Signatures follow the *concrete* client (positional ``name``, optional ``parent``) rather than
    the narrower shared protocols, since that is what services actually call.
    """

    def __init__(self) -> None:
        # The job transformer now accepts only the concrete typed-client boundary. This fake
        # implements that boundary directly while keeping all storage in memory.
        self.entities: dict[tuple[str, str, str, str | None], EntityBase] = {}
        #: Principal stamped as ``created_by``, as the real store does from the request identity.
        self.caller = "user:alice"
        #: Monotonic tick for creation timestamps. Wall-clock ``now()`` can repeat within a test,
        #: which would make ``-created_at`` ordering non-deterministic — the real store's inserts
        #: are genuinely ordered, so the fake must be too.
        self._tick = 0

    def _now(self) -> datetime:
        self._tick += 1
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._tick)

    def _key(self, entity_type, name: str, workspace: str, parent: str | None):
        return (entity_type.__entity_type__, workspace, name, parent)

    async def create(self, entity):
        key = self._key(type(entity), entity.name, entity.workspace, entity._parent)
        if key in self.entities:
            raise NemoEntityConflictError(f"{key} exists")
        now = self._now()
        entity._id = f"{entity.__entity_type__}-{self._tick}"
        entity._created_at = now
        entity._created_by = self.caller
        entity._updated_at = now
        entity._db_version = 0
        self.entities[key] = entity.model_copy(deep=True)
        return entity.model_copy(deep=True)

    async def get(self, entity_type, name: str, *, workspace: str | None = None, parent: str | None = None):
        workspace, name = parse_qualified_name(name, default_workspace=workspace)
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return self.entities[key].model_copy(deep=True)

    async def get_by_id(self, entity_type, entity_id):
        for entity in self.entities.values():
            if entity.id == entity_id and isinstance(entity, entity_type):
                return entity.model_copy(deep=True)
        raise NemoEntityNotFoundError("Task not found")

    async def update(self, entity, *, original_name: str | None = None):
        key = self._key(type(entity), original_name or entity.name, entity.workspace, entity._parent)
        stored = self.entities.get(key)
        if stored is not None and entity._db_version != stored._db_version:
            raise NemoEntityConflictError(f"stale update for {entity.name}")
        entity._db_version = (stored._db_version if stored is not None else 0) + 1
        entity._updated_at = self._now()
        self.entities[key] = entity.model_copy(deep=True)
        return entity.model_copy(deep=True)

    async def delete(
        self,
        entity_type,
        name: str,
        *,
        workspace: str | None = None,
        parent: str | None = None,
        expected_db_version: int | None = None,
    ) -> DeleteResponse:
        # ``parent`` is part of the key, not decoration: revision children of different heads share
        # the name ``rev.1``, so ignoring it here would delete whichever one was inserted first.
        workspace, name = parse_qualified_name(name, default_workspace=workspace)
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        deleted = self.entities.pop(key)
        return DeleteResponse(id=deleted.id)

    async def list(
        self,
        entity_type,
        *,
        workspace: str = DEFAULT_WORKSPACE,
        filter_operation=None,
        filter_str: str | None = None,
        sort: str | None = None,
        filter_obj: dict[str, Any] | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse:
        if filter_str is not None or filter_obj is not None:
            raise NotImplementedError("the fake entity store filters by filter_operation only")
        items = [
            entity
            for (entity_type_name, record_workspace, _, _), entity in self.entities.items()
            if entity_type_name == entity_type.__entity_type__ and workspace in ("-", record_workspace)  # "-": all
        ]
        items = [entity for entity in items if matches_filter(entity, filter_operation)]
        if sort and items:
            # Honour ``sort`` rather than ignoring it: services rely on server-side ordering, and a
            # fake that returned insertion order would make a wrong ``sort`` argument invisible.
            # Nothing to order when the result is empty, and probing the field on a bare ``object``
            # would reject a legitimate sort over an empty page.
            field = sort.lstrip("-")
            if not hasattr(items[0], field):
                raise NotImplementedError(f"fake cannot sort on {field!r}")
            items = sorted(items, key=lambda entity: getattr(entity, field), reverse=sort.startswith("-"))
        # Totals describe the whole result set, not the page — that distinction is the only way a
        # caller can tell a truncated history from a complete one, which is what ``list_revisions``
        # documents.
        total_results = len(items)
        start = (page - 1) * page_size
        items = [entity.model_copy(deep=True) for entity in items[start : start + page_size]]
        return ListResponse(
            data=items,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                current_page_size=len(items),
                total_pages=math.ceil(total_results / page_size),
                total_results=total_results,
            ),
        )


@pytest.fixture
def entity_store() -> FakeEntityStore:
    """A fresh in-memory entity store.

    Exposed as a fixture rather than an importable name because ``from conftest import ...`` is
    ambiguous once more than one test root is collected in the same run — the wrong ``conftest``
    module wins and the import fails.
    """
    return FakeEntityStore()


# ---- judge secret resolution ----------------------------------------------


@pytest.fixture
def allow_cloudpickle_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a deployment whose operator has enabled cloudpickle metrics."""
    monkeypatch.setattr(get_config(), "allow_insecure_cloudpickle_metrics", True)


class _Detonator:
    def __reduce__(self):
        return (pytest.fail, ("a cloudpickle metric payload was deserialized",))


@pytest.fixture
def detonating_cloudpickle_metric() -> dict:
    """A cloudpickle ``MetricInline`` whose payload fails the test if anything deserializes it."""
    bundle = bundle_metric(ExactMatchMetric(reference="a", candidate="a"), CloudpickleMetricBundlePackager())
    payload = CloudpickleMetricPayload.from_blob(pickle.dumps(_Detonator()))
    detonating = bundle.model_copy(update={"payload": payload})
    return MetricInline.model_validate_json(detonating.model_dump_json()).model_dump(mode="json")


@pytest.fixture
def make_secrets_service() -> type[InMemorySecretsService]:
    """Return a factory for in-memory Secrets services seeded with ``(workspace, name) -> value``."""
    return InMemorySecretsService


@pytest.fixture(autouse=True)
def _guard_codex_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``CODEX_HOME`` and the temp root at a pytest temp dir.

    Codex-adapter trials link the base ``auth.json`` into a temp Codex home; without this guard, unit
    tests would link the developer's real ``~/.codex`` login.
    """
    guard = tmp_path_factory.mktemp("codex-home-guard")
    monkeypatch.setenv("CODEX_HOME", str(guard / "base-codex-home"))
    temp_root = guard / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))


# --- Taskset runs on Kubernetes (M1 story 5 contracts, M2 story 0) -----------------------------------------

DISPATCHER_IMAGE = "registry.invalid/nhx-tasks@sha256:" + "1" * 64
SANDBOX_DOMAIN = "opensandbox.opensandbox-system.svc.cluster.local"


def harbor_task_jobs_config(**overrides: object) -> HarborTaskJobsConfig:
    """An operator ``evals.runs.harbor`` block with the MVP's two profiles."""
    data: dict[str, object] = {
        "dispatcher_image": DISPATCHER_IMAGE,
        "dispatcher_profiles": ["local-k8s", "k8s-cpu"],
        "sandbox_profiles": {
            "harbor-standard": {"egress_allow": ["igw.svc.cluster.local"]},
            "harbor-open": {"egress_allow": ["igw.svc.cluster.local", "inference-api.nvidia.com"]},
        },
    }
    data.update(overrides)
    return HarborTaskJobsConfig.model_validate(data)


def sandbox_platform_config(**overrides: object) -> NemoHelixConfig:
    """Platform settings naming an in-cluster OpenSandbox server."""
    fields: dict[str, object] = {
        "sandbox_cluster_capable": True,
        "sandbox_server_domain": SANDBOX_DOMAIN,
        "sandbox_server_protocol": "http",
    }
    fields.update(overrides)
    return get_platform_config().model_copy(update=fields)


@pytest.fixture
def sandbox_platform() -> NemoHelixConfig:
    return sandbox_platform_config()


@pytest.fixture
def harbor_config() -> HarborTaskJobsConfig:
    return harbor_task_jobs_config()


@pytest.fixture
def sandbox_policy() -> ResolvedHarborSandboxPolicy:
    return resolve_sandbox_policy(harbor_task_jobs_config(), sandbox_platform_config())


class FakePreparer:
    """Stand-in for ``prepare_task_images``: prebuilt images only, deterministic, no network.

    Satisfies ``ImagePreparer``. Keep it in step with the real preparer when M3 story 7 lands.
    ``reject`` scripts ``check``; ``fail`` scripts the preparation call.
    ``ready_at_once=True`` skips step A and returns ``ready`` on the first call, as the MVP body does.
    """

    def __init__(
        self,
        *,
        fail: Mapping[str, PreparationErrorCode] | None = None,
        reject: Mapping[str, PreparationErrorCode] | None = None,
        stay_pending: AbstractSet[str] = frozenset(),
        with_verifier: AbstractSet[str] = frozenset(),
        ready_at_once: bool = False,
    ) -> None:
        self.fail = dict(fail or {})
        self.reject = dict(reject or {})
        self.checked: list[str] = []
        self.stay_pending = stay_pending
        self.with_verifier = with_verifier
        self.ready_at_once = ready_at_once
        self.calls: list[tuple[str, TaskPreparation | None]] = []

    async def check(
        self, task: ResolvedTask, *, policy: ResolvedHarborSandboxPolicy, clients: PreparationClients
    ) -> PreparationErrorInfo | None:
        self.checked.append(task.id)
        if task.id in self.reject:
            return PreparationErrorInfo(code=self.reject[task.id], detail="scripted")
        return None

    async def __call__(
        self,
        task: ResolvedTask,
        recorded: TaskPreparation | None,
        *,
        policy: ResolvedHarborSandboxPolicy,
        caller: str,
        clients: PreparationClients,
        deadline: datetime,
    ) -> TaskPreparation:
        self.calls.append((task.id, recorded))
        if task.id in self.fail:
            return TaskPreparation(
                status="failed", error=PreparationErrorInfo(code=self.fail[task.id], detail="scripted")
            )
        roles: tuple[ImageRole, ...] = ("agent", "verifier") if task.id in self.with_verifier else ("agent",)
        if recorded is None and not self.ready_at_once:
            return TaskPreparation(
                status="pending",
                resource_class="cpu",
                images={
                    r: ImagePreparation(role=r, preparation_key=_key(task.id, r), source="prebuilt") for r in roles
                },
            )
        if task.id in self.stay_pending:
            assert recorded is not None
            return recorded
        prepared = {r: _prepared(task.id, r) for r in roles}
        return TaskPreparation(
            status="ready",
            resource_class="cpu",
            images={
                r: ImagePreparation(role=r, preparation_key=_key(task.id, r), source="prebuilt", prepared=prepared[r])
                for r in roles
            },
            environment=prepared_environment(task.id, with_verifier="verifier" in roles),
        )


def prepared_environment(task_id: str, *, with_verifier: bool = False) -> PreparedHarborEnvironment:
    """The environment ``FakePreparer`` gives a ready task."""
    return PreparedHarborEnvironment(
        task_revision_digest=_key(task_id, "revision"),
        archive_digest=_key(task_id, "archive"),
        agent=_prepared(task_id, "agent"),
        verifier=_prepared(task_id, "verifier") if with_verifier else None,
    )


def _key(task_id: str, part: str) -> str:
    return hashlib.sha256(f"{task_id}:{part}".encode()).hexdigest()


def _prepared(task_id: str, role: ImageRole) -> PreparedImage:
    return PreparedImage(
        image_ref=f"registry.invalid/fake/{task_id}-{role}@sha256:{_key(task_id, role)}",
        preparation_key=_key(task_id, role),
    )


@pytest.fixture
def fake_preparer() -> FakePreparer:
    return FakePreparer()


class FakeTaskJobs:
    """In-memory ``TaskJobs``: ``subprocess_profiles`` are the profiles Jobs would run as host subprocesses."""

    def __init__(self) -> None:
        self.subprocess_profiles: set[str] = set()

    async def is_subprocess_profile(self, profile: str) -> bool:
        return profile in self.subprocess_profiles


def _http_error[ErrorT: NemoHTTPError](cls: type[ErrorT], status: int) -> ErrorT:
    return cls(httpx.Response(status, json={"detail": "scripted"}, request=httpx.Request("POST", "http://jobs.test")))


@pytest.fixture
def fake_task_jobs() -> FakeTaskJobs:
    return FakeTaskJobs()


class FakeRunEvaluations:
    """In-memory ``RunEvaluations``: Experiments by name, Evaluations by name, with scripted failures.

    - ``fail``: operations ("experiment", "create", "get", "patch") that raise a 500, or
      "create-denied" for a create that answers 403.
    - ``deny_experiments``: the Experiment lookup answers 403.
    """

    def __init__(self) -> None:
        self.experiments: dict[str, str] = {"swe-agent-tuning": "exp-1"}
        self.evaluations: dict[str, EvaluationResponse] = {}
        self.created: list[EvaluationCreateRequest] = []
        self.patches: list[tuple[str, EvaluationPatchRequest]] = []
        self.fail: set[str] = set()
        self.deny_experiments = False

    async def experiment_id(self, workspace: str, name: str) -> str | None:
        if self.deny_experiments:
            raise _http_error(PermissionDeniedError, 403)
        if "experiment" in self.fail:
            raise _http_error(InternalServerError, 500)
        return self.experiments.get(name)

    async def create(self, workspace: str, body: EvaluationCreateRequest) -> None:
        if "create" in self.fail:
            raise _http_error(InternalServerError, 500)
        if "create-denied" in self.fail:
            raise _http_error(PermissionDeniedError, 403)
        if body.name in self.evaluations:
            raise _http_error(ConflictError, 409)
        self.created.append(body)
        self.evaluations[body.name] = EvaluationResponse(
            id=f"eval-{len(self.evaluations)}", workspace=workspace, **body.model_dump(exclude={"experiment_group_id"})
        )

    async def get(self, workspace: str, name: str) -> EvaluationResponse | None:
        if "get" in self.fail:
            raise _http_error(InternalServerError, 500)
        return self.evaluations.get(name)

    async def patch(self, workspace: str, name: str, body: EvaluationPatchRequest) -> None:
        if "patch" in self.fail:
            raise _http_error(InternalServerError, 500)
        self.patches.append((name, body))
        self.evaluations[name] = self.evaluations[name].model_copy(update=body.model_dump(exclude_none=True))


@pytest.fixture
def fake_evaluations() -> FakeRunEvaluations:
    return FakeRunEvaluations()


def harbor_resolved_task(name: str = "sum-three", *, config: dict | None = None) -> ResolvedTask:
    """A resolved stored Harbor task snapshot, as submit freezes it into a plan."""
    digest = hashlib.sha256(name.encode()).hexdigest()
    return ResolvedTask(
        id=f"hello/{name}",
        spec=ResolvedHarborTaskDefinition(
            kind="harbor",
            native_task_id=f"hello/{name}",
            instruction="Solve the task.",
            source=HarborArchiveSource(
                fileset_ref=f"default/harbor-tasks#{name}/{digest}/task_archive", files_hash=digest
            ),
            harbor_hash=HarborTaskHash(digest=digest, harbor_version="0.20.0"),
            config=config if config is not None else {"environment": {}, "verifier": {"environment_mode": "separate"}},
            provenance=TaskProvenance(entity_name=f"default/debug-{name}", revision_digest=digest),
        ),
    )


@pytest.fixture(name="harbor_resolved_task")
def harbor_resolved_task_factory() -> Callable[..., ResolvedTask]:
    """``harbor_resolved_task`` for tests that build tasks with their own ``task.toml``."""
    return harbor_resolved_task


@pytest.fixture
def harbor_spec() -> AgentEvalSpec:
    """A resolved two-task Harbor spec running the built-in oracle agent."""
    return AgentEvalSpec(
        tasks=[harbor_resolved_task("greet-universe"), harbor_resolved_task("sum-three")],
        target=HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle")),
    )


@pytest.fixture
def harbor_plan(harbor_spec, sandbox_policy, harbor_config) -> TasksetRunPlan:
    """An accepted two-task Harbor run's frozen plan."""
    return TasksetRunPlan(
        run_id="run-0123456789abcdef",
        workspace="default",
        taskset_ref="default/debug#" + "e" * 64,
        spec=harbor_spec,
        max_concurrent_task_jobs=None,
        sandbox_policy=sandbox_policy,
        dispatcher_image=harbor_config.dispatcher_image,
        dispatcher_profile="k8s-cpu",
        task_environments=[prepared_environment(task.id) for task in harbor_spec.tasks],
    )


@pytest.fixture
def harbor_run() -> TasksetRunEntity:
    """The run record ``harbor_plan`` belongs to."""
    accepted = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    return TasksetRunEntity(
        name="run-0123456789abcdef",
        workspace="default",
        plan_ref="default/run-plan#plan.json",
        plan_digest="f" * 64,
        request_digest="0" * 64,
        taskset_ref="default/debug#" + "e" * 64,
        accepted_at=accepted,
        timeout_seconds=3600,
        execution_deadline_at=accepted + timedelta(hours=1),
        finalization_deadline_at=accepted + timedelta(hours=2),
    )


ARCHIVE_REF = "default/harbor-tasks#sort-task/abc/task_archive"
ARCHIVE_DIGEST = "a" * 64

StoreTask = Callable[..., Awaitable[str]]


class _StoreTaskMetrics:
    async def store_derived_metric(self, metric, *, workspace: str) -> MetricRef:
        return MetricRef(f"{workspace}/derived")

    async def get_metric(self, workspace: str, name: str) -> object | None:
        return object()


@pytest.fixture
def store_task(entity_store, monkeypatch) -> StoreTask:
    """Store a task in ``entity_store`` and return its revision digest."""
    from nhx_evals_sdk.agent_eval.runtimes.harbor.archive import NativeTask

    async def verify(spec, files_client):
        return NativeTask("sort-task", spec.instruction, spec.config)

    monkeypatch.setattr("nemo_evals.api.service.task_service.verify_definition", verify)
    service = TaskService(entity_store, _StoreTaskMetrics(), files_client=Mock(), authorize_metric_read=AsyncMock())

    async def store(name: str = "sort-task", *, config: dict | None = None, harbor: bool = True) -> str:
        spec = (
            HarborTaskDefinition(
                kind="harbor",
                native_task_id="sort-task",
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                source=HarborArchiveSource(fileset_ref=ARCHIVE_REF, files_hash=ARCHIVE_DIGEST),
                config=config or {},
            )
            if harbor
            else EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Answer.",
                inputs=TaskInputs(instruction="2+2?"),
                metrics=[MetricRef("default/m")],
            )
        )
        await service.create_task(name, TaskInput(spec=spec), workspace="default")
        head = await entity_store.get(TaskEntity, name=name, workspace="default")
        return (await get_revision(entity_store, TaskRevisionEntity, head)).content_hash

    return store
