# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditor plugin SDK CRUD sub-resources and ``run`` helper.

Each CRUD test stubs ``platform._client`` with a ``MagicMock(spec=httpx.Client)``
(or ``AsyncMock(spec=httpx.AsyncClient)``) so we can assert on the URL and
JSON body the SDK actually sends — same pattern the evaluator plugin uses in
``plugins/nemo-evaluator/tests/test_sdk.py``.

``test_run_*`` patches ``nemo_auditor.sdk.AuditJob`` so the test never actually
shells out to garak; we just verify the SDK builds the right ``AuditInputSpec``
payload and calls the job with an explicit local ``JobContext``.
"""

from __future__ import annotations

import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_auditor.entities import (
    AuditConfig,
    AuditPluginsData,
    AuditReportData,
    AuditRunData,
    AuditSystemData,
    AuditTarget,
)
from nemo_auditor.sdk import AsyncAuditorPluginResource, AuditorPluginResource
from nemo_auditor.sdk_resources.job_resources import AsyncAuditorJobResource, AuditorJobResource
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job_context import JobContext

NOW = datetime.now(timezone.utc)


class _SyncPlatform:
    def __init__(self) -> None:
        self.base_url = "http://test:8000"
        self._client = MagicMock(spec=httpx.Client)


class _AsyncPlatform:
    def __init__(self) -> None:
        self.base_url = "http://test:8000"
        self._client = AsyncMock(spec=httpx.AsyncClient)


def _config_payload(name: str = "cfg-1", workspace: str = "default", **overrides: Any) -> dict:
    """Return the wire shape the FastAPI route returns for an AuditConfig."""
    base = {
        "id": f"auditor-audit-config-{name}-id",
        "name": name,
        "workspace": workspace,
        "entity_type": "auditor_audit_config",
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "description": None,
        "system": AuditSystemData().model_dump(mode="json"),
        "run": AuditRunData().model_dump(mode="json"),
        "plugins": AuditPluginsData().model_dump(mode="json"),
        "reporting": AuditReportData().model_dump(mode="json"),
    }
    base.update(overrides)
    return base


def _target_payload(name: str = "tgt-1", workspace: str = "default", **overrides: Any) -> dict:
    base = {
        "id": f"auditor-audit-target-{name}-id",
        "name": name,
        "workspace": workspace,
        "entity_type": "auditor_audit_target",
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "description": None,
        "type": "nim",
        "model": "meta/llama-3.1-8b-instruct",
        "options": {},
    }
    base.update(overrides)
    return base


def _ok_response(payload: dict, *, status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


# ---------------------------------------------------------------------------
# Sync CRUD: configs
# ---------------------------------------------------------------------------


class TestSyncConfigs:
    def test_create_posts_to_workspace_route_with_full_body(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_config_payload(name="cfg-1"), status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        cfg = resource.configs.create(
            workspace="default",
            name="cfg-1",
            description="hello",
            system=AuditSystemData(lite=True, parallel_attempts=4),
            run=AuditRunData(generations=3),
        )

        assert isinstance(cfg, AuditConfig)
        assert cfg.name == "cfg-1"
        assert cfg.workspace == "default"
        platform._client.post.assert_called_once()
        url, kwargs = platform._client.post.call_args[0], platform._client.post.call_args.kwargs
        assert url == ("http://test:8000/apis/auditor/v2/workspaces/default/configs",)
        body = kwargs["json"]
        assert body["name"] == "cfg-1"
        assert body["description"] == "hello"
        assert body["system"]["lite"] is True
        assert body["system"]["parallel_attempts"] == 4
        assert body["run"]["generations"] == 3

    def test_create_fills_default_subblocks_when_omitted(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_config_payload(), status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        resource.configs.create(workspace="default", name="cfg-1")

        body = platform._client.post.call_args.kwargs["json"]
        # CreateAuditConfigRequest defaults must round-trip even when caller omits everything.
        assert body["system"] == AuditSystemData().model_dump(mode="json")
        assert body["run"] == AuditRunData().model_dump(mode="json")
        assert body["plugins"] == AuditPluginsData().model_dump(mode="json")
        assert body["reporting"] == AuditReportData().model_dump(mode="json")

    def test_list_forwards_pagination_params(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(
            {
                "data": [],
                "pagination": {"page": 2, "page_size": 5, "total_pages": 0, "total_results": 0},
                "sort": "name",
            }
        )
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        body = resource.configs.list(workspace="prod", page=2, page_size=5, sort="name")

        assert body["pagination"]["page"] == 2
        url = platform._client.get.call_args.args[0]
        params = platform._client.get.call_args.kwargs["params"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/prod/configs"
        assert params == {"page": 2, "page_size": 5, "sort": "name"}

    def test_get_hits_named_route_and_returns_entity(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(_config_payload(name="cfg-1"))
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        cfg = resource.configs.get(workspace="default", name="cfg-1")

        assert isinstance(cfg, AuditConfig)
        assert cfg.name == "cfg-1"
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/configs/cfg-1",
        )

    def test_update_puts_full_body(self) -> None:
        platform = _SyncPlatform()
        platform._client.put.return_value = _ok_response(_config_payload(name="cfg-1", description="new"))
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        cfg = resource.configs.update(workspace="default", name="cfg-1", description="new")

        assert cfg.description == "new"
        url = platform._client.put.call_args.args[0]
        body = platform._client.put.call_args.kwargs["json"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/default/configs/cfg-1"
        assert body["description"] == "new"

    def test_delete_hits_named_route(self) -> None:
        platform = _SyncPlatform()
        response = MagicMock(spec=httpx.Response)
        response.status_code = 204
        response.raise_for_status.return_value = None
        platform._client.delete.return_value = response
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        result = resource.configs.delete(workspace="default", name="cfg-1")

        assert result is None
        platform._client.delete.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/configs/cfg-1",
        )


# ---------------------------------------------------------------------------
# Sync CRUD: targets
# ---------------------------------------------------------------------------


class TestSyncTargets:
    def test_create_posts_to_workspace_route(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_target_payload(name="tgt-1"), status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        tgt = resource.targets.create(
            workspace="default",
            name="tgt-1",
            type="nim",
            model="meta/llama-3.1-8b-instruct",
            options={"uri": "http://localhost:9000/v1"},
            description="local nim",
        )

        assert isinstance(tgt, AuditTarget)
        assert tgt.name == "tgt-1"
        platform._client.post.assert_called_once()
        url = platform._client.post.call_args.args[0]
        body = platform._client.post.call_args.kwargs["json"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/default/targets"
        assert body == {
            "name": "tgt-1",
            "description": "local nim",
            "type": "nim",
            "model": "meta/llama-3.1-8b-instruct",
            "options": {"uri": "http://localhost:9000/v1"},
        }

    def test_get_and_delete_hit_named_route(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(_target_payload(name="tgt-1"))
        delete_response = MagicMock(spec=httpx.Response)
        delete_response.status_code = 204
        delete_response.raise_for_status.return_value = None
        platform._client.delete.return_value = delete_response
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        tgt = resource.targets.get(workspace="default", name="tgt-1")
        resource.targets.delete(workspace="default", name="tgt-1")

        assert isinstance(tgt, AuditTarget)
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/targets/tgt-1",
        )
        platform._client.delete.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/targets/tgt-1",
        )

    def test_list_and_update_round_trip(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(
            {"data": [_target_payload(name="a"), _target_payload(name="b")], "pagination": None, "sort": "-created_at"}
        )
        platform._client.put.return_value = _ok_response(_target_payload(name="tgt-1", model="new-model"))
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        listed = resource.targets.list(workspace="default")
        assert [t["name"] for t in listed["data"]] == ["a", "b"]

        updated = resource.targets.update(
            workspace="default",
            name="tgt-1",
            type="nim",
            model="new-model",
        )
        assert updated.model == "new-model"


# ---------------------------------------------------------------------------
# Sub-resource caching
# ---------------------------------------------------------------------------


def test_configs_and_targets_properties_are_cached() -> None:
    platform = _SyncPlatform()
    resource = AuditorPluginResource(cast(NeMoPlatform, platform))

    cached_configs = resource.configs
    cached_targets = resource.targets
    assert resource.configs is cached_configs
    assert resource.targets is cached_targets


# ---------------------------------------------------------------------------
# Sync run()
# ---------------------------------------------------------------------------


class TestSyncRun:
    def test_resolves_name_strings_via_get_then_calls_job(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.side_effect = [
            _ok_response(_config_payload(name="my-cfg", workspace="default")),
            _ok_response(_target_payload(name="my-tgt", workspace="default")),
        ]
        job = MagicMock()
        job.run.return_value = {"status": "completed", "returncode": 0, "results": {}}

        with patch("nemo_auditor.sdk.AuditJob", return_value=job):
            resource = AuditorPluginResource(cast(NeMoPlatform, platform))
            result = resource.run(config="my-cfg", target="my-tgt", workspace="default")

        assert result["status"] == "completed"
        # Resolved both entities via two sync GETs.
        assert platform._client.get.call_count == 2
        platform._client.get.assert_any_call(
            "http://test:8000/apis/auditor/v2/workspaces/default/configs/my-cfg",
        )
        platform._client.get.assert_any_call(
            "http://test:8000/apis/auditor/v2/workspaces/default/targets/my-tgt",
        )

        # Spec handed to the job must carry inline entities (not strings).
        job.run.assert_called_once()
        args, kwargs = job.run.call_args
        spec_dict = args[0]
        assert isinstance(spec_dict["config"], dict)
        assert spec_dict["config"]["name"] == "my-cfg"
        assert isinstance(spec_dict["target"], dict)
        assert spec_dict["target"]["name"] == "my-tgt"
        assert kwargs["sdk"] is platform
        assert isinstance(kwargs["ctx"], JobContext)
        assert kwargs["ctx"].workspace == "default"

    def test_inline_entities_skip_http_resolution(self) -> None:
        platform = _SyncPlatform()
        job = MagicMock()
        job.run.return_value = {"status": "completed", "returncode": 0, "results": {}}

        inline_config = AuditConfig(name="inline-cfg", workspace="default")
        inline_target = AuditTarget(name="inline-tgt", workspace="default", type="nim", model="m")

        with patch("nemo_auditor.sdk.AuditJob", return_value=job):
            resource = AuditorPluginResource(cast(NeMoPlatform, platform))
            resource.run(config=inline_config, target=inline_target)

        # No HTTP roundtrip — inline entities go straight to the job.
        platform._client.get.assert_not_called()
        spec_dict = job.run.call_args.args[0]
        assert spec_dict["config"]["name"] == "inline-cfg"
        assert spec_dict["target"]["name"] == "inline-tgt"
        ctx = job.run.call_args.kwargs["ctx"]
        assert isinstance(ctx, JobContext)
        assert ctx.workspace == "default"

    def test_workspace_qualified_name_parses_workspace_from_string(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.side_effect = [
            _ok_response(_config_payload(name="cfg-1", workspace="prod")),
            _ok_response(_target_payload(name="tgt-1", workspace="staging")),
        ]
        job = MagicMock()
        job.run.return_value = {"status": "completed", "returncode": 0, "results": {}}

        with patch("nemo_auditor.sdk.AuditJob", return_value=job):
            resource = AuditorPluginResource(cast(NeMoPlatform, platform))
            resource.run(config="prod/cfg-1", target="staging/tgt-1", workspace="default")

        # GETs must use the workspace from the qualified name, not the default.
        platform._client.get.assert_any_call(
            "http://test:8000/apis/auditor/v2/workspaces/prod/configs/cfg-1",
        )
        platform._client.get.assert_any_call(
            "http://test:8000/apis/auditor/v2/workspaces/staging/targets/tgt-1",
        )
        # The run workspace used by JobContext still comes from the kwarg.
        ctx = job.run.call_args.kwargs["ctx"]
        assert isinstance(ctx, JobContext)
        assert ctx.workspace == "default"


# ---------------------------------------------------------------------------
# Job submission: submit / list_jobs / get_job
# ---------------------------------------------------------------------------

_JOB_PAYLOAD = {
    "name": "audit-job-abc123",
    "workspace": "default",
    "status": "created",
}

_JOBS_LIST_PAYLOAD = {
    "data": [_JOB_PAYLOAD],
    "pagination": {"page": 1, "page_size": 20, "total_pages": 1, "total_results": 1},
}


class TestSyncJobMethods:
    def test_submit_with_string_refs_posts_correct_url_and_body(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_JOB_PAYLOAD, status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        result = resource.submit(config="ws/my-cfg", target="ws/my-tgt", workspace="ws")

        assert isinstance(result, AuditorJobResource)
        assert result.name == "audit-job-abc123"
        platform._client.post.assert_called_once()
        url = platform._client.post.call_args.args[0]
        body = platform._client.post.call_args.kwargs["json"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit"
        assert body["spec"]["config"] == "ws/my-cfg"
        assert body["spec"]["target"] == "ws/my-tgt"
        assert body["spec"]["max_probe_retries"] == 0
        assert body["spec"]["fail_job_on_retries_exhausted"] is True

    def test_submit_with_inline_entities_serialises_full_dict(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_JOB_PAYLOAD, status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        cfg = AuditConfig(name="cfg-1", workspace="default")
        tgt = AuditTarget(name="tgt-1", workspace="default", type="nim", model="llama")
        result = resource.submit(config=cfg, target=tgt, workspace="default")

        assert isinstance(result, AuditorJobResource)
        body = platform._client.post.call_args.kwargs["json"]
        assert isinstance(body["spec"]["config"], dict)
        assert body["spec"]["config"]["name"] == "cfg-1"
        assert isinstance(body["spec"]["target"], dict)
        assert body["spec"]["target"]["model"] == "llama"

    def test_submit_defaults_workspace_to_default(self) -> None:
        platform = _SyncPlatform()
        platform._client.post.return_value = _ok_response(_JOB_PAYLOAD, status_code=201)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        result = resource.submit(config="my-cfg", target="my-tgt")

        assert isinstance(result, AuditorJobResource)
        url = platform._client.post.call_args.args[0]
        assert "/workspaces/default/" in url

    def test_list_jobs_hits_collection_url_with_pagination(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(_JOBS_LIST_PAYLOAD)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        result = resource.list_jobs(workspace="ws", page=2, page_size=5)

        assert result == _JOBS_LIST_PAYLOAD
        url = platform._client.get.call_args.args[0]
        params = platform._client.get.call_args.kwargs["params"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit"
        assert params == {"page": 2, "page_size": 5}

    def test_get_job_hits_named_url(self) -> None:
        platform = _SyncPlatform()
        platform._client.get.return_value = _ok_response(_JOB_PAYLOAD)
        resource = AuditorPluginResource(cast(NeMoPlatform, platform))

        result = resource.get_job("audit-job-abc123", workspace="ws")

        assert result == _JOB_PAYLOAD
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit/audit-job-abc123",
        )


@pytest.mark.asyncio
class TestAsyncJobMethods:
    async def test_submit_posts_correct_url_and_body(self) -> None:
        platform = _AsyncPlatform()
        platform._client.post.return_value = _ok_response(_JOB_PAYLOAD, status_code=201)
        resource = AsyncAuditorPluginResource(cast(AsyncNeMoPlatform, platform))

        result = await resource.submit(config="ws/my-cfg", target="ws/my-tgt", workspace="ws")

        assert isinstance(result, AsyncAuditorJobResource)
        assert result.name == "audit-job-abc123"
        url = platform._client.post.call_args.args[0]
        body = platform._client.post.call_args.kwargs["json"]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit"
        assert body["spec"]["config"] == "ws/my-cfg"
        assert body["spec"]["target"] == "ws/my-tgt"

    async def test_list_jobs_hits_collection_url(self) -> None:
        platform = _AsyncPlatform()
        platform._client.get.return_value = _ok_response(_JOBS_LIST_PAYLOAD)
        resource = AsyncAuditorPluginResource(cast(AsyncNeMoPlatform, platform))

        result = await resource.list_jobs(workspace="ws")

        assert result == _JOBS_LIST_PAYLOAD
        url = platform._client.get.call_args.args[0]
        assert url == "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit"

    async def test_get_job_hits_named_url(self) -> None:
        platform = _AsyncPlatform()
        platform._client.get.return_value = _ok_response(_JOB_PAYLOAD)
        resource = AsyncAuditorPluginResource(cast(AsyncNeMoPlatform, platform))

        result = await resource.get_job("audit-job-abc123", workspace="ws")

        assert result == _JOB_PAYLOAD
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/ws/jobs/audit/audit-job-abc123",
        )


# ---------------------------------------------------------------------------
# Async smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_configs_create_posts_to_workspace_route() -> None:
    platform = _AsyncPlatform()
    platform._client.post.return_value = _ok_response(_config_payload(name="cfg-1"), status_code=201)
    resource = AsyncAuditorPluginResource(cast(AsyncNeMoPlatform, platform))

    cfg = await resource.configs.create(workspace="default", name="cfg-1", description="hi")

    assert isinstance(cfg, AuditConfig)
    assert cfg.name == "cfg-1"
    platform._client.post.assert_called_once()
    url = platform._client.post.call_args.args[0]
    assert url == "http://test:8000/apis/auditor/v2/workspaces/default/configs"


@pytest.mark.asyncio
async def test_async_run_resolves_names_and_calls_job_in_thread() -> None:
    platform = _AsyncPlatform()
    platform._client.get.side_effect = [
        _ok_response(_config_payload(name="my-cfg")),
        _ok_response(_target_payload(name="my-tgt")),
    ]
    job = MagicMock()
    job.run.return_value = {"status": "completed", "returncode": 0, "results": {}}

    with (
        patch("nemo_auditor.sdk.AuditJob", return_value=job) as audit_job_cls,
        patch("nemo_auditor.sdk.asyncio.to_thread", new=AsyncMock(return_value=job.run.return_value)) as to_thread,
    ):
        resource = AsyncAuditorPluginResource(cast(AsyncNeMoPlatform, platform))
        result = await resource.run(config="my-cfg", target="my-tgt", workspace="default")

    assert result["status"] == "completed"
    audit_job_cls.assert_called_once_with()
    # Job call is dispatched via asyncio.to_thread so the caller's loop stays free.
    to_thread.assert_awaited_once()
    call = to_thread.await_args
    assert call is not None
    assert call.args[0] is job.run
    spec_dict = call.args[1]
    assert spec_dict["config"]["name"] == "my-cfg"
    assert spec_dict["target"]["name"] == "my-tgt"
    assert call.kwargs["async_sdk"] is platform
    assert isinstance(call.kwargs["ctx"], JobContext)
    assert call.kwargs["ctx"].workspace == "default"


# ---------------------------------------------------------------------------
# AuditorJobResource
# ---------------------------------------------------------------------------

_STATUS_PAYLOAD = {"name": "audit-job-abc123", "status": "active", "workspace": "default"}


def _make_tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"report data"
        info = tarfile.TarInfo(name="report.jsonl")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class TestAuditorJobResource:
    def _make_resource(self) -> tuple[_SyncPlatform, AuditorJobResource]:
        platform = _SyncPlatform()
        resource = AuditorJobResource(
            job_name="audit-job-abc123",
            platform=cast(NeMoPlatform, platform),
            workspace="default",
        )
        return platform, resource

    def test_name_property(self) -> None:
        _, resource = self._make_resource()
        assert resource.name == "audit-job-abc123"

    def test_get_job_hits_named_url(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response(_JOB_PAYLOAD)

        result = resource.get_job()

        assert result == _JOB_PAYLOAD
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/jobs/audit/audit-job-abc123"
        )

    def test_get_job_status_hits_status_url(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "active"})

        status = resource.get_job_status()

        assert status == "active"
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/jobs/audit/audit-job-abc123/status"
        )

    def test_check_if_complete_returns_true_when_completed(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "completed"})
        assert resource.check_if_complete() is True

    def test_check_if_complete_returns_false_when_active(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "active"})
        assert resource.check_if_complete() is False

    def test_check_if_complete_raises_when_requested(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "error"})
        with pytest.raises(RuntimeError):
            resource.check_if_complete(raise_if_not_complete=True)

    def test_wait_until_done_polls_until_completed(self) -> None:
        platform, resource = self._make_resource()
        # Status sequence: active → completed; logs return empty pages each time.
        status_responses = [
            _ok_response({"status": "active"}),
            _ok_response({"status": "active"}),
            _ok_response({"status": "completed"}),
        ]
        logs_response = _ok_response({"data": [], "next_page": None})
        platform._client.get.side_effect = (
            status_responses[:1] + [logs_response, status_responses[1]] + [logs_response, status_responses[2]]
        )

        with patch("nemo_auditor.sdk_resources.job_resources._pause"):
            resource.wait_until_done()

        # Final status call resolves to "completed" — no RuntimeError raised.

    def test_wait_until_done_exits_on_terminal_failure(self) -> None:
        platform, resource = self._make_resource()
        status_responses = [
            _ok_response({"status": "active"}),
            _ok_response({"status": "error"}),
        ]
        logs_response = _ok_response({"data": [], "next_page": None})
        platform._client.get.side_effect = [
            status_responses[0],
            logs_response,
            status_responses[1],
        ]

        with patch("nemo_auditor.sdk_resources.job_resources._pause"):
            resource.wait_until_done()  # must not raise — just logs error and returns

    def test_download_artifacts_raises_when_not_completed(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "active"})

        with pytest.raises(RuntimeError, match="status 'active'"):
            resource.download_artifacts()

    def test_download_artifacts_extracts_tarball(self, tmp_path: Path) -> None:
        platform, resource = self._make_resource()
        status_resp = _ok_response({"status": "completed"})
        tar_resp = MagicMock(spec=httpx.Response)
        tar_resp.raise_for_status.return_value = None
        tar_resp.content = _make_tar_bytes()
        platform._client.get.side_effect = [status_resp, tar_resp]

        result = resource.download_artifacts(path=tmp_path)

        assert result == tmp_path
        platform._client.get.assert_called_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/jobs/audit/audit-job-abc123/results/artifacts/download"
        )

    def test_poll_safe_returns_fallback_on_transient_error(self) -> None:
        _, resource = self._make_resource()

        def failing() -> str:
            raise ConnectionError("blip")

        result = resource._poll_safe(failing, "cached")
        assert result == "cached"
        assert resource._consecutive_poll_errors == 1

    def test_poll_safe_raises_after_max_consecutive_errors(self) -> None:
        _, resource = self._make_resource()
        resource._consecutive_poll_errors = 4

        def failing() -> str:
            raise ConnectionError("blip")

        with pytest.raises(ConnectionError):
            resource._poll_safe(failing, "cached")
        assert resource._consecutive_poll_errors == 0


@pytest.mark.asyncio
class TestAsyncAuditorJobResource:
    def _make_resource(self) -> tuple[_AsyncPlatform, AsyncAuditorJobResource]:
        platform = _AsyncPlatform()
        resource = AsyncAuditorJobResource(
            job_name="audit-job-abc123",
            platform=cast(AsyncNeMoPlatform, platform),
            workspace="default",
        )
        return platform, resource

    async def test_name_property(self) -> None:
        _, resource = self._make_resource()
        assert resource.name == "audit-job-abc123"

    async def test_get_job_status_hits_status_url(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "completed"})

        status = await resource.get_job_status()

        assert status == "completed"
        platform._client.get.assert_called_once_with(
            "http://test:8000/apis/auditor/v2/workspaces/default/jobs/audit/audit-job-abc123/status"
        )

    async def test_check_if_complete_returns_true_when_completed(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "completed"})
        assert await resource.check_if_complete() is True

    async def test_check_if_complete_raises_when_requested(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "cancelled"})
        with pytest.raises(RuntimeError):
            await resource.check_if_complete(raise_if_not_complete=True)

    async def test_wait_until_done_polls_until_completed(self) -> None:
        platform, resource = self._make_resource()
        status_responses = [
            _ok_response({"status": "active"}),
            _ok_response({"status": "active"}),
            _ok_response({"status": "completed"}),
        ]
        logs_response = _ok_response({"data": [], "next_page": None})
        platform._client.get.side_effect = (
            status_responses[:1] + [logs_response, status_responses[1]] + [logs_response, status_responses[2]]
        )

        with patch("nemo_auditor.sdk_resources.job_resources._async_pause", new=AsyncMock()):
            await resource.wait_until_done()

    async def test_download_artifacts_raises_when_not_completed(self) -> None:
        platform, resource = self._make_resource()
        platform._client.get.return_value = _ok_response({"status": "pending"})

        with pytest.raises(RuntimeError, match="status 'pending'"):
            await resource.download_artifacts()

    async def test_download_artifacts_extracts_tarball(self, tmp_path: Path) -> None:
        platform, resource = self._make_resource()
        status_resp = _ok_response({"status": "completed"})
        tar_resp = MagicMock(spec=httpx.Response)
        tar_resp.raise_for_status.return_value = None
        tar_resp.content = _make_tar_bytes()
        platform._client.get.side_effect = [status_resp, tar_resp]

        result = await resource.download_artifacts(path=tmp_path)

        assert result == tmp_path
