# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real hardener routers against mixed persisted specs, with SQL list filtering."""

import json
from contextlib import asynccontextmanager
from math import ceil
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from nemo_agent_hardener_plugin.api.v2.jobs import router, synth_router
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_helix_plugin.agent_hardener.types import SynthBenignSpec, WarGameSpec
from nemo_helix_plugin.api.filter import ComparisonOperation, LogicalOperation, parse_json_filter
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client, get_request_authorizer
from nemo_helix_plugin.jobs import api_factory
from nemo_helix_plugin.jobs.schemas import FileStorageType, HelixJobResultResponse
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from sqlalchemy import JSON, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient


class _Base(DeclarativeBase):
    pass


class _StoredJob(_Base):
    __tablename__ = "jobs"
    name: Mapped[str] = mapped_column(primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)


def _job(name, spec, source="agent-hardener"):
    return SimpleNamespace(
        id=f"id-{name}",
        name=name,
        workspace="default",
        source=source,
        spec=spec,
        description=None,
        created_at=None,
        updated_at=None,
        status="created",
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
    )


def _response(value):
    return SimpleNamespace(data=lambda: value)


def _entity_filter(operation):
    # Same core Jobs namespace translation: source/spec live in entity data.
    if isinstance(operation, ComparisonOperation):
        return operation.model_copy(update={"field": f"data.{operation.field}"})
    assert isinstance(operation, LogicalOperation)
    return operation.model_copy(update={"operations": [_entity_filter(op) for op in operation.operations]})


@pytest.fixture
def mixed_jobs(monkeypatch):
    war = WarGameSpec(config="/project/agent-hardener.yaml").model_dump(mode="json")
    service_war = WarGameSpec(manifest_id="manifest", driver="service").model_dump(mode="json")
    synth = SynthBenignSpec(manifest_id="manifest").model_dump(mode="json")
    records = {
        "old-war": _job("old-war", war),
        "old-service-war": _job("old-service-war", service_war),
        "old-synth": _job("old-synth", synth),
        "new-war": _job("new-war", {**war, "_nemo_job_route": "war-game"}),
        "new-synth": _job("new-synth", {**synth, "_nemo_job_route": "synth-benign"}),
        "foreign": _job("foreign", war, "other"),
        "ambiguous": _job("ambiguous", {**war, **synth}),
        "unknown": _job("unknown", {"manifest_id": "manifest"}),
        "unknown-marker": _job("unknown-marker", {**war, "_nemo_job_route": "future"}),
    }
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    _Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            _StoredJob(name=name, data={"source": job.source, "spec": job.spec}) for name, job in records.items()
        )
        session.commit()
    jobs = AsyncMock()
    jobs.get_job.side_effect = lambda *, name, workspace: _response(records[name])
    jobs.with_retry = Mock(return_value=jobs)

    async def list_jobs(*, workspace, query_params):
        operation = _entity_filter(parse_json_filter(query_params["filter"]))
        condition = operation.apply(SQLAlchemyFilterRepository(_StoredJob, dialect_name="sqlite"))
        page, size = query_params["page"], query_params["page_size"]
        with Session(engine) as session:
            total = session.scalar(select(func.count()).select_from(_StoredJob).where(condition))
            names = session.scalars(
                select(_StoredJob.name).where(condition).order_by(_StoredJob.name).offset((page - 1) * size).limit(size)
            ).all()
        return SimpleNamespace(
            page=lambda: SimpleNamespace(
                items=[records[name] for name in names],
                metadata={
                    "page": page,
                    "page_size": size,
                    "current_page_size": len(names),
                    "total_pages": ceil(total / size),
                    "total_results": total,
                },
            )
        )

    jobs.list_jobs.side_effect = list_jobs
    monkeypatch.setattr(api_factory.AsyncJobsClient, "from_client", lambda _: jobs)

    @asynccontextmanager
    async def internal_client(**kwargs):
        assert kwargs == {"as_service": "agent-hardener", "internal": True, "workspace": "default"}
        yield SimpleNamespace()

    monkeypatch.setattr(api_factory, "get_async_nemo_client", internal_client)
    app = FastAPI()
    app.include_router(router, prefix="/v2/{workspace}")
    app.include_router(synth_router, prefix="/v2/{workspace}/synth-benign")
    app.dependency_overrides[get_nemo_client] = lambda: SimpleNamespace()
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    authorize = AsyncMock()
    app.dependency_overrides[get_request_authorizer] = lambda: authorize
    with TestClient(app) as client:
        yield client, jobs, records
    engine.dispose()


@pytest.mark.parametrize(
    "collection,expected",
    [
        ("", {"old-war", "old-service-war", "new-war"}),
        ("/synth-benign", {"old-synth", "new-synth"}),
    ],
)
@pytest.mark.parametrize(
    "user_filter",
    [None, {"$or": [{"name": {"$eq": "old-war"}}, {"name": {"$like": "%"}}]}, {"$not": {"name": {"$eq": "missing"}}}],
)
def test_mixed_sql_lists_filter_before_pagination(mixed_jobs, collection, expected, user_filter):
    client, _, _ = mixed_jobs
    seen = set()
    for page in range(1, len(expected) + 1):
        params = {"page": page, "page_size": 1}
        if user_filter is not None:
            params["filter"] = json.dumps(user_filter)
        response = client.get(f"/v2/default{collection}/jobs", params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pagination"]["total_results"] == len(expected)
        assert body["pagination"]["total_pages"] == len(expected)
        assert len(body["data"]) == 1
        assert "_nemo_job_route" not in body["data"][0]["spec"]
        seen.add(body["data"][0]["name"])
    assert seen == expected


@pytest.mark.parametrize(
    "collection,own_names",
    [("", {"old-war", "old-service-war", "new-war"}), ("/synth-benign", {"old-synth", "new-synth"})],
)
def test_named_access_preserves_legacy_and_rejects_siblings(mixed_jobs, collection, own_names):
    client, _, records = mixed_jobs
    for name in records:
        response = client.get(f"/v2/default{collection}/jobs/{name}")
        assert response.status_code == (200 if name in own_names else 404), response.text


@pytest.mark.parametrize(
    "collection,name",
    [
        ("", "old-synth"),
        ("", "new-synth"),
        ("/synth-benign", "old-war"),
        ("/synth-benign", "new-war"),
        ("", "ambiguous"),
        ("/synth-benign", "unknown"),
        ("", "foreign"),
    ],
)
@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", "/status"),
        ("GET", "/logs"),
        ("GET", "/results"),
        ("GET", "/results/report"),
        ("GET", "/results/report/download"),
        ("POST", "/cancel"),
        ("DELETE", ""),
    ],
)
def test_cross_type_routes_do_not_read_results_or_mutate(mixed_jobs, collection, name, method, suffix):
    client, jobs, _ = mixed_jobs
    response = client.request(method, f"/v2/default{collection}/jobs/{name}{suffix}")
    assert response.status_code == 404, response.text
    assert [call[0] for call in jobs.method_calls] == ["get_job"] or (
        method == "DELETE" and [call[0] for call in jobs.method_calls] == ["with_retry", "get_job"]
    )


@pytest.mark.parametrize(
    "collection,name",
    [
        ("", "old-war"),
        ("", "old-service-war"),
        ("", "new-war"),
        ("/synth-benign", "old-synth"),
        ("/synth-benign", "new-synth"),
    ],
)
@pytest.mark.parametrize(
    "method,suffix", [("GET", "/status"), ("GET", "/results"), ("POST", "/cancel"), ("DELETE", "")]
)
def test_own_legacy_lifecycle_and_results(mixed_jobs, collection, name, method, suffix):
    client, jobs, records = mixed_jobs
    jobs.cancel_job.return_value = _response(records[name])
    jobs.get_job_status.return_value = _response(
        SimpleNamespace(
            model_dump=lambda: {
                "id": f"id-{name}",
                "name": name,
                "status": "created",
                "status_details": {},
                "error_details": None,
                "steps": [],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    jobs.list_job_results.return_value = _response(SimpleNamespace(data=[]))
    response = client.request(method, f"/v2/default{collection}/jobs/{name}{suffix}")
    assert response.status_code == (204 if method == "DELETE" else 200), response.text


@pytest.mark.parametrize("collection,name", [("", "old-war"), ("/synth-benign", "old-synth")])
def test_legacy_logs_result_metadata_and_download(mixed_jobs, monkeypatch, tmp_path, collection, name):
    client, jobs, _ = mixed_jobs
    jobs.list_job_logs.return_value = SimpleNamespace(
        page=lambda: SimpleNamespace(
            items=[],
            metadata={"total": 0, "next_page": None, "prev_page": None},
        )
    )
    result = HelixJobResultResponse(
        name="report",
        job=name,
        workspace="default",
        artifact_url="fileset://report",
        artifact_storage_type=FileStorageType.FILESET,
    )
    jobs.get_job_result.return_value = _response(result)
    report = tmp_path / "report.txt"
    report.write_text("own result")
    download = AsyncMock(return_value=(None, SimpleNamespace(path=report, cleanup_tmp_dir=lambda: None)))
    monkeypatch.setattr(api_factory, "download_from_result_info", download)
    monkeypatch.setattr(api_factory.AsyncFilesClient, "from_client", lambda _: SimpleNamespace())
    prefix = f"/v2/default{collection}/jobs/{name}"
    assert client.get(f"{prefix}/logs").status_code == 200
    metadata = client.get(f"{prefix}/results/report")
    assert metadata.status_code == 200, metadata.text
    assert metadata.json()["job"] == name
    response = client.get(f"{prefix}/results/report/download")
    assert response.status_code == 200, response.text
    assert response.text == "own result"
    assert download.await_args is not None
    assert download.await_args.kwargs["job_name"] == name


@pytest.mark.parametrize(
    "collection,job_class,spec,marker",
    [
        ("", AgentHardenerRunJob, {"config": "/project/config.yaml"}, "war-game"),
        ("/synth-benign", AgentHardenerSynthBenignJob, {"manifest_id": "manifest"}, "synth-benign"),
    ],
)
def test_creation_stamps_kind_without_changing_public_or_task_spec(
    mixed_jobs, monkeypatch, collection, job_class, spec, marker
):
    client, jobs, records = mixed_jobs
    compiler = AsyncMock(
        return_value={"steps": [{"name": "step", "executor": {"provider": "cpu", "container": {"image": "test"}}}]}
    )
    monkeypatch.setattr(job_class, "compile", compiler)

    async def create(*, workspace, body):
        assert workspace == "default"
        assert body.source == "agent-hardener"
        records[body.name] = _job(body.name, body.spec)
        return _response(records[body.name])

    jobs.create_job.side_effect = create
    response = client.post(
        f"/v2/default{collection}/jobs", json={"name": "created", "spec": {**spec, "_nemo_job_route": "spoof"}}
    )
    assert response.status_code == 201, response.text
    assert "_nemo_job_route" not in response.json()["spec"]
    assert compiler.await_args is not None
    assert "_nemo_job_route" not in compiler.await_args.kwargs["spec"].model_dump()
    assert records["created"].spec["_nemo_job_route"] == marker
    assert jobs.create_job.await_args.kwargs["body"].platform_spec.steps[0].config == {}
    sibling = "" if collection else "/synth-benign"
    assert client.get(f"/v2/default{sibling}/jobs/created").status_code == 404


@pytest.mark.parametrize("collection,sibling", [("", "synth-benign"), ("/synth-benign", "war-game")])
def test_caller_cannot_select_sibling_marker(mixed_jobs, collection, sibling):
    client, _, _ = mixed_jobs
    response = client.get(
        f"/v2/default{collection}/jobs", params={"filter": json.dumps({"spec._nemo_job_route": {"$eq": sibling}})}
    )
    assert response.status_code == 200, response.text
    assert response.json()["pagination"]["total_results"] == 0
    assert response.json()["data"] == []
