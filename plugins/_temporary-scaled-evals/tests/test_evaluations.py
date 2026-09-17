# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the evaluations router and the dispatch worker.

In-process FastAPI TestClient with a mocked psycopg connection, plus direct
unit tests of the Dispatcher/backend with an injected fake backend. No cluster,
no compose — end-to-end coverage lives in tests/integration/.
"""

import json
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")
from api_test_fixture import client, v1
from scaled_evals.api.db import Database, get_conn, get_stream_database_factory
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.routers import evaluations as evaluations_router
from scaled_evals.api.schemas.common import decode_cursor
from scaled_evals.api.schemas.evaluations import CreateEvaluationRequest
from scaled_evals.api.settings import settings
from scaled_evals.dispatch import (
    LaunchHandle,
    LaunchSpec,
    RuntimeStatus,
    summarize_harbor_result,
)
from scaled_evals.dispatch.switchyard import (
    SwitchyardProfileConfig,
    SwitchyardRender,
    render_switchyard,
    write_switchyard_artifacts,
)
from scaled_evals.dispatch.worker import (
    Dispatcher,
)

NOW = datetime(2026, 6, 6, tzinfo=UTC)


# A Harbor framework-typed result, shaped like
# examples/agent-sandbox/results/astra-oracle-result.json: one eval, one trial,
# reward 1.0. The write-back path summarizes and stores this verbatim.
SAMPLE_HARBOR_RESULT = {
    "id": "49e59139-bb0c-4699-aa6f-17e018f0eb52",
    "started_at": "2026-06-06T07:45:22.289428",
    "updated_at": "2026-06-06T07:46:03.430270",
    "finished_at": "2026-06-06T07:46:03.430270",
    "n_total_trials": 1,
    "stats": {
        "n_completed_trials": 1,
        "n_errored_trials": 0,
        "evals": {
            "oracle__adhoc": {
                "n_trials": 1,
                "n_errors": 0,
                "metrics": [{"mean": 1.0}],
                "reward_stats": {"reward": {"1.0": ["tasks-hello-skills__CqhqmY7"]}},
            }
        },
    },
}


def _eval_row(**overrides) -> dict:
    """A full evaluations row as the DB would return it (dict_row)."""
    row = {
        "id": "ev_test123",
        "name": "run 7",
        "framework": "harbor",
        "task_id": "task_abc",
        "task_revision": 2,
        "framework_profile_id": None,
        "harbor_profile_id": None,
        "switchyard_profile_id": None,
        "intake_profile_id": None,
        "credentials": {},
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "image_ref": "registry.example/task_abc:rev2",
        "n_attempts": 1,
        "parallelism": 4,
        "visibility": "private",
        "status": "queued",
        "status_detail": None,
        "cancel_teardown_status": "not_requested",
        "cancel_teardown_error": None,
        "cancel_teardown_updated_at": None,
        "backend_handle": None,
        "dispatch_job_name": None,
        "dispatch_job_uid": None,
        "current_execution": 1,
        "max_executions": 3,
        "next_retry_at": None,
        "last_failure_code": None,
        "last_failure_category": None,
        "reward": None,
        "n_trials": None,
        "n_errored": None,
        "finished_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    row.update(overrides)
    return row


def _archive_row(**overrides) -> dict:
    row = {
        "id": "ev_test123",
        "archive_status": "missing",
        "archive_object_key": None,
        "archive_size_bytes": None,
        "archive_built_at": None,
        "archive_error": None,
    }
    row.update(overrides)
    return row


def _conn_with_fetchone(*results) -> MagicMock:
    """Mock connection whose successive fetchone() calls return `results`."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = list(results)
    cur.fetchall.return_value = []
    return conn


def _conn(fetchone=(), fetchall=()) -> MagicMock:  # noqa: ANN001
    """Mock connection with explicit fetchone() and fetchall() sequences.

    For the POST path: fetchone serves the revision-status check then the
    INSERT ... RETURNING; fetchall serves the config_profiles then credentials
    existence queries (skipped when there are no such references).
    """
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = list(fetchone)
    cur.fetchall.side_effect = list(fetchall)
    return conn


def _override_conn(conn: MagicMock) -> None:
    def _gen() -> Iterator[MagicMock]:
        yield conn

    @contextmanager
    def _stream_gen() -> Iterator[Database]:
        yield Database(conn)

    v1.dependency_overrides[get_conn] = _gen
    v1.dependency_overrides[get_stream_database_factory] = lambda: _stream_gen


def _sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        parsed: dict = {"data": None}
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event: "):
                parsed["event"] = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        if data_lines:
            parsed["data"] = json.loads("\n".join(data_lines))
        events.append(parsed)
    return events


@pytest.fixture(autouse=True)
def _cleanup_overrides() -> Iterator[None]:
    evaluations_router._sse_active_connections = 0
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(get_stream_database_factory, None)
    evaluations_router._sse_active_connections = 0


@pytest.fixture(autouse=True)
def _stub_dispatch_archive_build(monkeypatch) -> None:  # noqa: ANN001
    def fake_build(evaluation_id: str) -> dict:
        return {
            "object_key": f"evaluations/{evaluation_id}/results.tar.gz",
            "size_bytes": 123,
            "file_count": 2,
            "source_bytes": 42,
        }

    monkeypatch.setattr("scaled_evals.dispatch.worker.s3.build_evaluation_archive", fake_build)
    monkeypatch.setattr(
        "scaled_evals.dispatch.worker.s3.build_evaluation_archive_from_directory",
        lambda evaluation_id, _root: fake_build(evaluation_id),
    )


@pytest.fixture(autouse=True)
def _disable_dispatch_artifact_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dispatcher unit tests independent from local object-store state."""
    monkeypatch.setattr(
        "scaled_evals.dispatch.worker.s3.sync_directory_to_prefix",
        lambda _root, _prefix: 0,
    )
    monkeypatch.setattr(
        "scaled_evals.dispatch.worker.s3.replace_directory_at_prefix",
        lambda _root, _prefix: 0,
    )


@pytest.fixture(autouse=True)
def _stub_execution_snapshot(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        EvaluationRepository,
        "_capture_execution_snapshot",
        MagicMock(
            return_value={
                "schema_version": "scaled-evals-execution-inputs-v1",
                "captured_at": NOW.isoformat(),
                "evaluation": {},
                "task": {},
                "profiles": {},
                "credentials": {},
                "submission_identity": {},
            }
        ),
    )


# ---------- POST happy path -----------------------------------------------


def test_create_commits_before_returning() -> None:
    """The evaluation row must be committed before the API responds.

    The dispatch worker opens its own connection to load the row; get_conn
    defers COMMIT to connection teardown (after the response) and the INSERT
    runs in a SAVEPOINT (the validation SELECTs already opened a txn), so
    without an explicit commit the worker can't see the queued row promptly.
    """
    conn = _conn_with_fetchone({"status": "ready"}, _eval_row())
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "run 8", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 202, response.text
    assert conn.commit.called, "router must commit the evaluation row"


def test_preflight_reports_runnable_without_creating_evaluation() -> None:
    conn = _conn_with_fetchone({"status": "ready"})
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations/preflight",
        json={"name": "check", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 200, response.text
    assert response.json()["runnable"] is True
    assert response.json()["kind"] == "evaluation"
    assert not any(
        "INSERT INTO evaluations" in call.args[0]
        for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    )
    assert not conn.commit.called


def test_preflight_reports_create_blocker_with_http_200() -> None:
    conn = _conn_with_fetchone({"status": "building"})
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations/preflight",
        json={"name": "check", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 200, response.text
    assert response.json()["runnable"] is False
    blocker = next(check for check in response.json()["checks"] if check["blocking"])
    assert blocker["code"] == "task_not_ready"


# ---------- POST guard rails ----------------------------------------------


def test_create_rejects_blank_initial_user_turn() -> None:
    with pytest.raises(ValueError, match="must not contain blank turns"):
        CreateEvaluationRequest(
            name="x",
            task_id="task_abc",
            task_revision=1,
            initial_user_turns=["   "],
        )


def test_create_rejects_initial_user_turns_for_non_harbor_framework() -> None:
    with pytest.raises(ValueError, match="only valid"):
        CreateEvaluationRequest(
            name="x",
            task_id="task_abc",
            task_revision=1,
            framework="nemo_gym",
            initial_user_turns=["initialize"],
        )


def test_create_409_when_revision_not_ready() -> None:
    conn = _conn_with_fetchone({"status": "building"})
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "task_not_ready"


def test_create_allows_ready_gcs_revision_when_pack_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "object_store_backend", "gcs")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "us-central1-docker.pkg.dev")
    checked: list[str] = []
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations.s3.object_exists",
        lambda key: checked.append(key) or True,
    )
    conn = _conn_with_fetchone(
        {
            "status": "ready",
            "image_ref": "us-central1-docker.pkg.dev/project/repo/task:rev1",
            "image_digest": "sha256:" + "a" * 64,
            "tarball_object_key": "task_abc/rev/1/tarball.tar.gz",
        },
        _eval_row(),
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 202, response.text
    assert checked == ["task_abc/rev/1/tarball.tar.gz"]


def test_create_409_when_ready_revision_pack_missing_local_rustfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "object_store_backend", "s3")
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.object_exists", lambda _key: False)
    conn = _conn_with_fetchone(
        {
            "status": "ready",
            "image_ref": "registry.example/task_abc:rev1",
            "image_digest": "sha256:" + "b" * 64,
            "tarball_object_key": "task_abc/rev/1/tarball.tar.gz",
        }
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_abc", "task_revision": 1},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "task_object_missing"
    assert not any(
        "INSERT INTO evaluations" in call.args[0]
        for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    )


def test_create_404_when_revision_missing() -> None:
    conn = _conn_with_fetchone(None)
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_missing", "task_revision": 9},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_create_rejects_task_image_from_unapproved_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "us-central1-docker.pkg.dev")
    conn = _conn_with_fetchone(
        {
            "status": "ready",
            "image_ref": "artifactory.nvidia.com/team/task:rev2",
            "image_digest": "sha256:" + "a" * 64,
        }
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_abc", "task_revision": 2},
    )

    assert response.status_code == 422
    error = response.json()["detail"]["error"]
    assert error["code"] == "invalid_task_image"
    assert error["message"] == (
        "task image registry 'artifactory.nvidia.com' is not approved; allowed registries: us-central1-docker.pkg.dev"
    )
    assert not any(
        "INSERT INTO evaluations" in call.args[0]
        for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    )


def test_create_422_on_malformed_profile_id() -> None:
    _override_conn(_conn_with_fetchone({"status": "ready"}, _eval_row()))

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "task_id": "task_abc",
            "task_revision": 1,
            "harbor_profile_id": "not-a-cfg-id",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_reference"


def test_create_records_default_deny_without_switchyard() -> None:
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            _eval_row(network_policy="default_deny"),
        ],
        fetchall=[[]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "offline",
            "task_id": "task_abc",
            "task_revision": 1,
            "network_policy": "default_deny",
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["network_policy"] == "default_deny"
    execute_calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    insert_call = next(call for call in execute_calls if "INSERT INTO evaluations" in call.args[0])
    assert insert_call.args[1][24] == "default_deny"
    assert insert_call.args[1][25].obj == {}


def test_create_resolves_harbor_version_before_queueing() -> None:
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            _eval_row(
                requested_framework_version="stable",
                framework_version="0.13.2",
                runner_image_ref="scaled-evals-api:dev",
                framework_adapter_version="scaled-evals-overlay-v5",
                sandbox_k8s_version="0.1.13",
            ),
        ],
        fetchall=[[]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "versioned",
            "task_id": "task_abc",
            "task_revision": 1,
            "framework_version": "stable",
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["framework_version"] == "0.13.2"
    execute_calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    insert_call = next(call for call in execute_calls if "INSERT INTO evaluations" in call.args[0])
    assert insert_call.args[1][4:10] == (
        "stable",
        "0.13.2",
        "scaled-evals-api:dev",
        None,
        "nemo-platform-plugin-overlay-v1",
        "0.1.13",
    )
    runner_metadata = insert_call.args[1][10].obj
    assert runner_metadata["qualification"]["release"]["version"] == "0.13.2"
    assert runner_metadata["qualification"]["release"]["wheel_sha256"]
    assert runner_metadata["artifact"]["image_ref"] == "scaled-evals-api:dev"


def test_create_gym_runtime_persists_runtime_runner_identity(monkeypatch) -> None:  # noqa: ANN001
    source_revision = "a" * 40
    digest = "sha256:" + "b" * 64
    monkeypatch.setattr(settings, "gym_runner_image", "registry.example/gym:0.4.0")
    monkeypatch.setattr(settings, "gym_runner_image_digest", digest)
    monkeypatch.setattr(settings, "gym_source_revision", source_revision)
    monkeypatch.setattr(settings, "gym_package_version", "0.4.0")
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            _eval_row(
                runtime="gym_daytona",
                runner_image_ref="registry.example/gym:0.4.0",
                runner_image_digest=digest,
            ),
        ],
        fetchall=[[]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "gym-runtime",
            "task_id": "task_abc",
            "task_revision": 1,
            "runtime": "gym_daytona",
        },
    )

    assert response.status_code == 202, response.text
    execute_calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    insert_call = next(call for call in execute_calls if "INSERT INTO evaluations" in call.args[0])
    assert insert_call.args[1][6:8] == ("registry.example/gym:0.4.0", digest)
    runner_metadata = insert_call.args[1][10].obj
    assert runner_metadata["artifact"]["source_revision"] == source_revision
    assert runner_metadata["gym"]["runtime"] == "gym_daytona"
    assert runner_metadata["gym"]["provider"] == "daytona"
    assert runner_metadata["framework_artifact"]["image_ref"] == "scaled-evals-api:dev"


def test_create_resolves_and_persists_agent_bundle(monkeypatch) -> None:
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            {"config": {}},
            _eval_row(framework_profile_id="cfg_h", harbor_profile_id="cfg_h"),
        ],
        fetchall=[[{"id": "cfg_h", "type": "harbor"}]],
    )
    _override_conn(conn)
    resolved = {
        "bundle_id": "ab_claude",
        "agent_name": "claude-code",
        "agent_version": "2.1.133",
        "image_ref": "artifactory.example/agent:2.1.133",
        "image_digest": "artifactory.example/agent@sha256:" + "a" * 64,
        "entrypoint": "bin/claude",
    }
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations.accessible_bundle_for_run",
        lambda db, principal, bundle_id: resolved,
    )

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "bundle",
            "task_id": "task_abc",
            "task_revision": 1,
            "framework_profile_id": "cfg_h",
            "agent_bundle_id": "ab_claude",
        },
    )

    assert response.status_code == 202, response.text
    execute_calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    insert_call = next(call for call in execute_calls if "INSERT INTO evaluations" in call.args[0])
    assert insert_call.args[1][10].obj["agent_bundle"] == resolved


def test_create_rejects_unsupported_harbor_version_with_choices() -> None:
    _override_conn(_conn())

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "unsupported",
            "task_id": "task_abc",
            "task_revision": 1,
            "framework_version": "latest",
        },
    )

    assert response.status_code == 422
    error = response.json()["detail"]["error"]
    assert error["code"] == "unsupported_framework_version"
    assert "0.6.3" in error["message"]
    assert "0.6.4" in error["message"]
    assert "0.13.2" in error["message"]


def test_create_rejects_scoped_egress_without_rules() -> None:
    _override_conn(MagicMock())
    response = client.post(
        "/v1/evaluations",
        json={
            "name": "scoped",
            "task_id": "task_abc",
            "task_revision": 1,
            "network_policy": "scoped_egress",
        },
    )

    assert response.status_code == 422


def test_create_rejects_zero_parallelism() -> None:
    _override_conn(_conn_with_fetchone({"status": "ready"}))
    response = client.post(
        "/v1/evaluations",
        json={"name": "x", "task_id": "task_abc", "task_revision": 1, "parallelism": 0},
    )
    assert response.status_code == 422


# ---------- POST reference existence/type validation (real tables) --------


def test_create_202_with_valid_profile_and_credential() -> None:
    # fetchone: ready revision, then the inserted row.
    # fetchall: config_profiles existence, then credentials existence.
    conn = _conn(
        fetchone=[{"status": "ready"}, {"config": {}}, _eval_row()],
        fetchall=[[{"id": "cfg_h", "type": "harbor"}], [{"id": "cred_a"}]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "run",
            "task_id": "task_abc",
            "task_revision": 2,
            "harbor_profile_id": "cfg_h",
            "credentials": {"anthropic": "cred_a"},
        },
    )

    assert response.status_code == 202, response.text


def test_create_202_with_nemo_gym_framework_profile() -> None:
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            _eval_row(framework="nemo_gym", framework_profile_id="cfg_g"),
        ],
        fetchall=[[{"id": "cfg_g", "type": "gym"}]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "run",
            "framework": "nemo_gym",
            "task_id": "task_abc",
            "task_revision": 2,
            "framework_profile_id": "cfg_g",
        },
    )

    assert response.status_code == 202, response.text


def test_create_202_treats_framework_profile_as_harbor_alias() -> None:
    conn = _conn(
        fetchone=[
            {"status": "ready"},
            {"config": {}},
            _eval_row(framework_profile_id="cfg_h", harbor_profile_id="cfg_h"),
        ],
        fetchall=[[{"id": "cfg_h", "type": "harbor"}]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "run",
            "task_id": "task_abc",
            "task_revision": 2,
            "framework_profile_id": "cfg_h",
        },
    )

    assert response.status_code == 202, response.text


def test_create_422_when_profile_missing() -> None:
    # Revision ready, but the config_profiles lookup returns no row.
    conn = _conn(fetchone=[{"status": "ready"}], fetchall=[[]])
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "task_id": "task_abc",
            "task_revision": 1,
            "harbor_profile_id": "cfg_ghost",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_reference"


def test_create_422_when_profile_wrong_type() -> None:
    # harbor_profile_id points at a switchyard-typed profile.
    conn = _conn(
        fetchone=[{"status": "ready"}],
        fetchall=[[{"id": "cfg_sw", "type": "switchyard"}]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "task_id": "task_abc",
            "task_revision": 1,
            "harbor_profile_id": "cfg_sw",
        },
    )

    assert response.status_code == 422
    assert "expected 'harbor'" in response.json()["detail"]["error"]["message"]


def test_create_422_when_nemo_gym_profile_wrong_type() -> None:
    conn = _conn(
        fetchone=[{"status": "ready"}],
        fetchall=[[{"id": "cfg_h", "type": "harbor"}]],
    )
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "framework": "nemo_gym",
            "task_id": "task_abc",
            "task_revision": 1,
            "framework_profile_id": "cfg_h",
        },
    )

    assert response.status_code == 422
    assert "expected 'gym'" in response.json()["detail"]["error"]["message"]


def test_create_422_when_harbor_aliases_conflict() -> None:
    _override_conn(_conn_with_fetchone({"status": "ready"}))

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "task_id": "task_abc",
            "task_revision": 1,
            "framework_profile_id": "cfg_one",
            "harbor_profile_id": "cfg_two",
        },
    )

    assert response.status_code == 422


def test_create_422_when_harbor_alias_used_for_nemo_gym() -> None:
    _override_conn(_conn_with_fetchone({"status": "ready"}))

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "framework": "nemo_gym",
            "task_id": "task_abc",
            "task_revision": 1,
            "harbor_profile_id": "cfg_h",
        },
    )

    assert response.status_code == 422


def test_create_422_when_credential_missing() -> None:
    # No profiles, so the only existence query is credentials, which misses.
    conn = _conn(fetchone=[{"status": "ready"}], fetchall=[[]])
    _override_conn(conn)

    response = client.post(
        "/v1/evaluations",
        json={
            "name": "x",
            "task_id": "task_abc",
            "task_revision": 1,
            "credentials": {"anthropic": "cred_ghost"},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_reference"


# ---------- list / get ----------------------------------------------------


def test_list_returns_envelope_shape() -> None:
    _override_conn(_conn_with_fetchone())
    response = client.get("/v1/evaluations")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_get_404_when_missing() -> None:
    _override_conn(_conn_with_fetchone(None))
    response = client.get("/v1/evaluations/ev_nope")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_get_returns_result_envelope_for_terminal_run() -> None:
    # A succeeded row carries the summary columns plus the full result JSONB.
    row = _eval_row(
        status="succeeded",
        reward=1.0,
        n_trials=1,
        n_errored=0,
        finished_at=NOW,
        result=SAMPLE_HARBOR_RESULT,
    )
    _override_conn(_conn_with_fetchone(row))

    response = client.get("/v1/evaluations/ev_test123")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["reward"] == 1.0
    assert body["n_trials"] == 1
    assert body["n_errored"] == 0
    # The framework-typed result envelope is returned verbatim.
    assert body["result"] == SAMPLE_HARBOR_RESULT
    assert body["current_execution"] == 1
    assert body["max_executions"] == 3
    assert body["links"]["retry"] == "/evaluations/ev_test123/retry"


def test_get_projects_boolean_reward_value_without_changing_result_envelope() -> None:
    raw_result = {"verifier_result": {"score": True}}
    row = _eval_row(
        status="succeeded",
        reward=None,
        reward_value=True,
        n_trials=1,
        n_errored=0,
        finished_at=NOW,
        result=raw_result,
    )
    _override_conn(_conn_with_fetchone(row))

    response = client.get("/v1/evaluations/ev_test123")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reward"] is True
    assert body["result"] == raw_result


def test_get_result_is_none_before_terminal() -> None:
    # A still-running row has no result yet; the field is present and null.
    _override_conn(_conn_with_fetchone(_eval_row(status="running")))
    response = client.get("/v1/evaluations/ev_test123")
    assert response.status_code == 200
    body = response.json()
    assert body["result"] is None
    assert body["reward"] is None


def test_get_telemetry_returns_attempt_aware_usage_and_raw_handoff_links() -> None:
    execution = {
        "execution_number": 2,
        "provisioning_started_at": NOW,
        "running_started_at": NOW,
        "terminal_at": NOW,
        "terminal_status": "succeeded",
        "failure_phase": None,
        "input_tokens": 120,
        "output_tokens": 30,
        "cached_tokens": 10,
        "cache_creation_tokens": 5,
        "usage_source": "switchyard-session-stats",
        "turn_count": 4,
        "tool_call_count": 2,
        "cost_usd": 0.25,
        "cost_source": "estimated",
        "raw_artifact_refs": [{"relation": "trajectory", "path": "trial/agent/trajectory.json"}],
        "intake_experiment_ref": "task-ev-test123",
        "intake_run_refs": ["scaled-evals:ev_test123:trial-1"],
        "intake_status": "succeeded",
        "intake_expected_records": 1,
        "intake_uploaded_records": 1,
        "intake_error": None,
        "artifact_sync_status": "succeeded",
        "artifact_sync_file_count": 12,
        "artifact_sync_error": None,
    }
    usage = {
        "execution_number": 2,
        "component": "sandbox",
        "source": "kubernetes_metrics_api",
        "collection_status": "sampled",
        "collection_error": None,
        "sample_count": 3,
        "first_observed_at": NOW,
        "last_observed_at": NOW,
        "cpu_sample_count": 3,
        "avg_cpu_cores": 0.25,
        "peak_cpu_cores": 0.5,
        "memory_sample_count": 0,
        "avg_memory_bytes": None,
        "peak_memory_bytes": None,
        "cpu_request_cores": 1.0,
        "cpu_limit_cores": 2.0,
        "memory_request_bytes": None,
        "memory_limit_bytes": None,
        "gpu_request": None,
        "gpu_sample_count": 0,
        "avg_gpu_usage_percent": None,
        "peak_gpu_usage_percent": None,
        "gpu_memory_sample_count": 0,
        "avg_gpu_memory_usage_bytes": None,
        "peak_gpu_memory_usage_bytes": None,
    }
    conn = _conn(fetchone=[_eval_row(current_execution=2, intake_profile_id="cfg_intake")])
    conn.cursor.return_value.__enter__.return_value.fetchall.side_effect = [
        [execution],
        [],
        [usage],
    ]
    _override_conn(conn)

    response = client.get("/v1/evaluations/ev_test123/telemetry")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "scaled-evals-evaluation-telemetry-v1"
    assert body["resource_usage"][0]["execution_number"] == 2
    assert body["resource_usage"][0]["avg_cpu_cores"] == 0.25
    assert body["executions"][0]["usage"] == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cached_tokens": 10,
        "cache_creation_tokens": 5,
        "source": "switchyard-session-stats",
    }
    assert body["executions"][0]["interactions"] == {"turns": 4, "tool_calls": 2}
    assert body["executions"][0]["cost"] == {
        "value_usd": 0.25,
        "source": "estimated",
    }
    assert body["executions"][0]["raw_artifacts"][0]["download"].endswith("/trial/agent/trajectory.json")
    assert body["intake"]["status"] == "succeeded"
    assert body["intake"]["complete"] is True
    assert body["intake"]["experiment_ref"] == "task-ev-test123"
    assert body["artifacts"]["artifact_sync_status"] == "succeeded"
    assert body["cleanup"] == {
        "cancellation_status": "not_requested",
        "error": None,
        "updated_at": None,
        "executions": [],
    }
    assert body["intake"] == {
        "enabled": True,
        "profile_id": "cfg_intake",
        "status": "succeeded",
        "experiment_ref": "task-ev-test123",
        "run_refs": ["scaled-evals:ev_test123:trial-1"],
        "expected_records": 1,
        "uploaded_records": 1,
        "complete": True,
        "error": None,
        "diagnostic_artifact": ("/evaluations/ev_test123/artifacts/intake-upload.json"),
    }
    assert body["artifacts"]["listing"] == "/evaluations/ev_test123/artifacts"
    assert body["artifacts"]["provenance"].endswith("scaled-evals-provenance.json")


def test_reproduce_returns_safe_rerun_request_and_command() -> None:
    _override_conn(
        _conn_with_fetchone(
            _eval_row(
                status="failed",
                framework_profile_id="cfg_h",
                harbor_profile_id="cfg_h",
                switchyard_profile_id="cfg_s",
                intake_profile_id="cfg_i",
                credentials={"openai": "cred_openai"},
                n_attempts=3,
                extra_skill_object_keys=["skills/review/SKILL.md"],
                instruction_prefix="inspect first",
                instruction_postfix="summarize last",
                initial_user_turns=["Initialize", "/review"],
                image_ref="registry.example/task_abc@sha256:task",
                status_detail="backend failed with api_key=sk-do-not-return",
            )
        )
    )

    response = client.get("/v1/evaluations/ev_test123/reproduce")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evaluation_id"] == "ev_test123"
    assert body["source_status"] == "failed"
    assert body["request"] == {
        "name": "rerun of run 7",
        "task_id": "task_abc",
        "task_revision": 2,
        "framework": "harbor",
        "framework_version": None,
        "framework_profile_id": "cfg_h",
        "harbor_profile_id": "cfg_h",
        "switchyard_profile_id": "cfg_s",
        "intake_profile_id": "cfg_i",
        "credentials": {"openai": "cred_openai"},
        "agent_bundle_id": None,
        "extra_skill_object_keys": ["skills/review/SKILL.md"],
        "instruction_prefix": "inspect first",
        "instruction_postfix": "summarize last",
        "initial_user_turns": ["Initialize", "/review"],
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "n_attempts": 3,
        "parallelism": 4,
        "visibility": "private",
    }
    assert body["cli_command"][:4] == ["scaled-evals", "evaluation", "create", "--name"]
    assert "--credential" in body["cli_command"]
    assert "openai=cred_openai" in body["cli_command"]
    assert "--extra-skill-object-key" in body["cli_command"]
    assert "--instruction-prefix" in body["cli_command"]
    assert "--instruction-postfix" in body["cli_command"]
    assert body["cli_command"].count("--initial-user-turn") == 2
    assert body["cli_command"][body["cli_command"].index("--n-attempts") + 1] == "3"
    assert "sk-do-not-return" not in response.text
    assert any("Secret material is not exported" in note for note in body["notes"])


def test_reproduce_404_when_missing() -> None:
    _override_conn(_conn_with_fetchone(None))
    response = client.get("/v1/evaluations/ev_nope/reproduce")
    assert response.status_code == 404


# ---------- retry ---------------------------------------------------------


def test_retry_queues_next_execution_on_same_benchmark_member() -> None:
    row = _eval_row(
        status="queued",
        status_detail="manual retry scheduled; execution 2",
        benchmark_run_id="bmr_1",
        current_execution=2,
        max_executions=3,
    )
    conn = _conn_with_fetchone(row)
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/retry")

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["id"] == "ev_test123"
    assert body["benchmark_run_id"] == "bmr_1"
    assert body["status"] == "queued"
    assert body["current_execution"] == 2
    assert body["links"]["retry"] == "/evaluations/ev_test123/retry"
    cur = conn.cursor.return_value.__enter__.return_value
    retry_sql = cur.execute.call_args_list[0].args[0]
    assert "UPDATE evaluations" in retry_sql
    assert "status = 'failed'" in retry_sql
    assert "current_execution = current_execution + 1" in retry_sql
    assert "INSERT INTO evaluations" not in retry_sql
    event_call = cur.execute.call_args_list[1]
    assert event_call.args[1][1:3] == ("retry", "queued")
    conn.commit.assert_called_once()


def test_retry_rejects_non_failed_evaluation() -> None:
    conn = _conn_with_fetchone(None, _eval_row(status="running"))
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/retry")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "evaluation_not_failed"


def test_retry_reports_terminal_artifact_finalization() -> None:
    conn = _conn_with_fetchone(
        None,
        _eval_row(status="failed"),
        {"reason": "terminal_artifacts_finalizing"},
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/retry")

    assert response.status_code == 409
    error = response.json()["detail"]["error"]
    assert error["code"] == "evaluation_retry_pending"
    assert "terminal evidence or archive generation is in progress" in error["message"]


def test_retry_reports_unavailable_benchmark() -> None:
    conn = _conn_with_fetchone(
        None,
        _eval_row(status="failed", benchmark_run_id="bmr_cancelled"),
        {"reason": "benchmark_unavailable"},
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/retry")

    assert response.status_code == 409
    error = response.json()["detail"]["error"]
    assert error["code"] == "evaluation_not_retryable"
    assert "benchmark run is cancelled or unavailable" in error["message"]


def test_retry_404_when_missing() -> None:
    _override_conn(_conn_with_fetchone(None, None))

    response = client.post("/v1/evaluations/ev_nope/retry")

    assert response.status_code == 404


# ---------- cancel --------------------------------------------------------


class _CancelBackend:
    name = "sandbox_k8s"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.handles: list[LaunchHandle] = []

    def launch(self, spec):  # noqa: ANN001
        raise NotImplementedError

    def status(self, handle):  # noqa: ANN001
        raise NotImplementedError

    def teardown(self, handle: LaunchHandle) -> None:
        self.handles.append(handle)
        if self.fail:
            raise RuntimeError("remote job already gone")


def test_cancel_not_yet_launched_marks_cancelled_without_teardown() -> None:
    conn = _conn_with_fetchone(
        _eval_row(
            status="cancelled",
            backend_handle=None,
            cancel_teardown_status="pending",
        ),
        _eval_row(
            status="cancelled",
            backend_handle=None,
            cancel_teardown_status="succeeded",
        ),
    )
    _override_conn(conn)
    response = client.post("/v1/evaluations/ev_test123/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancel_teardown_status"] == "succeeded"
    cancel_sql = conn.cursor.return_value.__enter__.return_value.execute.call_args_list[0].args[0]
    assert "evidence_status = 'building'" in cancel_sql
    assert "finished_at = COALESCE(finished_at, NOW())" in cancel_sql


def test_cancel_launched_tears_down_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _CancelBackend()
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    conn = _conn_with_fetchone(
        _eval_row(
            status="cancelled",
            runtime="sandbox_k8s",
            backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
            cancel_teardown_status="pending",
        ),
        _eval_row(
            status="cancelled",
            runtime="sandbox_k8s",
            backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
            cancel_teardown_status="succeeded",
        ),
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancel_teardown_status"] == "succeeded"
    assert [handle.external_id for handle in backend.handles] == ["job-123"]


def test_cancel_job_backed_evaluation_defers_teardown_to_owning_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _CancelBackend()
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    conn = _conn_with_fetchone(
        _eval_row(
            status="cancelled",
            runtime="gym_sandbox_opensandbox",
            dispatch_job_name="scaled-evals-eval-123",
            cancel_teardown_status="pending",
            backend_handle={
                "backend": "gym_sandbox_opensandbox",
                "external_id": "ev_test123",
                "raw": {"process_pid": 42, "process_owner_pod": "old-pod"},
            },
        )
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancel_teardown_status"] == "pending"
    assert backend.handles == []


def test_cancel_terminal_returns_unchanged_without_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _CancelBackend()
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    _override_conn(
        _conn_with_fetchone(
            None,
            _eval_row(
                status="succeeded",
                backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
            ),
        )
    )

    response = client.post("/v1/evaluations/ev_test123/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "succeeded"
    assert backend.handles == []


def test_cancel_records_teardown_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _CancelBackend(fail=True)
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    conn = _conn_with_fetchone(
        _eval_row(
            status="cancelled",
            runtime="sandbox_k8s",
            backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
            cancel_teardown_status="pending",
        ),
        _eval_row(
            status="cancelled",
            status_detail="cancelled; evaluation-runtime cleanup failed: remote job already gone",
            runtime="sandbox_k8s",
            backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
            cancel_teardown_status="failed",
            cancel_teardown_error=("cancelled; evaluation-runtime cleanup failed: remote job already gone"),
        ),
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert response.json()["status_detail"] == ("cancelled; evaluation-runtime cleanup failed: remote job already gone")
    assert response.json()["cancel_teardown_status"] == "failed"
    assert [handle.external_id for handle in backend.handles] == ["job-123"]


def test_cancel_404_when_missing() -> None:
    # UPDATE matches nothing, then the existence SELECT also returns nothing.
    _override_conn(_conn_with_fetchone(None, None))
    response = client.post("/v1/evaluations/ev_nope/cancel")
    assert response.status_code == 404


# ---------- delete --------------------------------------------------------


def test_delete_soft_deletes() -> None:
    _override_conn(_conn_with_fetchone({"id": "ev_test123"}))
    response = client.delete("/v1/evaluations/ev_test123")
    assert response.status_code == 200
    assert response.json() == {"id": "ev_test123", "deleted": True}


def test_delete_404_when_missing() -> None:
    _override_conn(_conn_with_fetchone(None))
    response = client.delete("/v1/evaluations/ev_nope")
    assert response.status_code == 404


# ---------- logs/events ---------------------------------------------------


def test_logs_tail_result_lines() -> None:
    _override_conn(_conn_with_fetchone(_eval_row(status="succeeded", result={"logs": ["a", "b", "c"]})))

    response = client.get("/v1/evaluations/ev_test123/logs", params={"tail_lines": 2})

    assert response.status_code == 200
    assert response.json() == {
        "evaluation_id": "ev_test123",
        "lines": ["b", "c"],
        "status": "succeeded",
        "complete": True,
    }


def test_logs_tail_backend_log_file(tmp_path) -> None:
    log_path = tmp_path / "harbor.log"
    log_path.write_text("one\ntwo\nthree\n")
    _override_conn(
        _conn_with_fetchone(
            _eval_row(
                status="running",
                backend_handle={"raw": {"log": str(log_path)}},
            )
        )
    )

    response = client.get("/v1/evaluations/ev_test123/logs", params={"tail_lines": 2})

    assert response.status_code == 200
    assert response.json()["lines"] == ["two", "three"]
    assert response.json()["complete"] is False


def test_logs_redact_api_keys_from_result_lines() -> None:
    _override_conn(
        _conn_with_fetchone(
            _eval_row(
                status="succeeded",
                result={
                    "logs": [
                        "Running: ++policy_api_key=sk-do-not-return",
                        "OPENAI_API_KEY=sk-also-do-not-return",
                    ]
                },
            )
        )
    )

    response = client.get("/v1/evaluations/ev_test123/logs")

    assert response.status_code == 200
    text = response.text
    assert "sk-do-not-return" not in text
    assert "sk-also-do-not-return" not in text
    assert "<redacted>" in text


def test_events_returns_persisted_transition_history() -> None:
    events = [
        {"type": "status", "status": "queued", "detail": None, "created_at": NOW},
        {"type": "status", "status": "running", "detail": "launched", "created_at": NOW},
    ]
    _override_conn(_conn(fetchone=[{"id": "ev_test123"}], fetchall=[events]))

    response = client.get("/v1/evaluations/ev_test123/events", params={"limit": 2, "offset": 0})

    assert response.status_code == 200
    body = response.json()
    assert [event["status"] for event in body["data"]] == ["queued", "running"]
    assert all(event["evaluation_id"] == "ev_test123" for event in body["data"])
    assert body["data"][1]["detail"] == "launched"


def test_events_returns_next_cursor_for_more_history() -> None:
    events = [
        {"id": 1, "type": "status", "status": "queued", "detail": None, "created_at": NOW},
        {
            "id": 2,
            "type": "status",
            "status": "provisioning",
            "detail": "claimed",
            "created_at": NOW + timedelta(microseconds=1),
        },
        {
            "id": 3,
            "type": "status",
            "status": "running",
            "detail": "launched",
            "created_at": NOW + timedelta(microseconds=2),
        },
    ]
    _override_conn(_conn(fetchone=[{"id": "ev_test123"}], fetchall=[events]))

    response = client.get("/v1/evaluations/ev_test123/events", params={"limit": 2})

    assert response.status_code == 200
    body = response.json()
    assert [event["status"] for event in body["data"]] == ["queued", "provisioning"]
    assert decode_cursor(body["next_cursor"]).id == "2"  # type: ignore[union-attr]


def test_events_404_when_evaluation_missing() -> None:
    _override_conn(_conn_with_fetchone(None))

    response = client.get("/v1/evaluations/ev_nope/events")

    assert response.status_code == 404


def test_events_stream_emits_live_persisted_events_until_terminal(monkeypatch) -> None:  # noqa: ANN001
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("scaled_evals.api.routers.evaluations.asyncio.sleep", no_sleep)
    queued = {"id": 1, "type": "status", "status": "queued", "detail": None, "created_at": NOW}
    running = {
        "id": 2,
        "type": "status",
        "status": "running",
        "detail": "launched",
        "created_at": NOW + timedelta(microseconds=1),
    }
    succeeded = {
        "id": 3,
        "type": "status",
        "status": "succeeded",
        "detail": "1/1 trials completed",
        "created_at": NOW + timedelta(microseconds=2),
    }
    conn = _conn(
        fetchone=[{"id": "ev_test123"}, {"status": "running"}],
        fetchall=[[queued], [], [running], [succeeded]],
    )
    _override_conn(conn)

    response = client.get("/v1/evaluations/ev_test123/events/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert [(event["event"], event["data"].get("status")) for event in events] == [
        ("status", "queued"),
        ("ping", None),
        ("status", "running"),
        ("status", "succeeded"),
    ]
    status_events = [event["data"] for event in events if event["event"] == "status"]
    assert all(set(event) == {"evaluation_id", "type", "status", "detail", "at"} for event in status_events)
    assert status_events[-1] == {
        "evaluation_id": "ev_test123",
        "type": "status",
        "status": "succeeded",
        "detail": "1/1 trials completed",
        "at": (NOW + timedelta(microseconds=2)).isoformat(),
    }


def test_events_stream_emits_terminal_event_after_status_flip(monkeypatch) -> None:  # noqa: ANN001
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("scaled_evals.api.routers.evaluations.asyncio.sleep", no_sleep)
    succeeded = {
        "id": 1,
        "type": "status",
        "status": "succeeded",
        "detail": "1/1 trials completed",
        "created_at": NOW,
    }
    conn = _conn(
        fetchone=[{"id": "ev_test123"}, {"status": "succeeded"}],
        fetchall=[[], [succeeded]],
    )
    _override_conn(conn)

    response = client.get("/v1/evaluations/ev_test123/events/stream")

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [(event["event"], event["data"].get("status")) for event in events] == [("status", "succeeded")]


def test_events_stream_does_not_stop_on_switchyard_lifecycle_event(monkeypatch) -> None:  # noqa: ANN001
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("scaled_evals.api.routers.evaluations.asyncio.sleep", no_sleep)
    switchyard = {
        "id": 1,
        "type": "switchyard",
        "status": "succeeded",
        "detail": "switchyard draining until later",
        "created_at": NOW,
    }
    succeeded = {
        "id": 2,
        "type": "status",
        "status": "succeeded",
        "detail": "1/1 trials completed",
        "created_at": NOW + timedelta(microseconds=1),
    }
    conn = _conn(fetchone=[{"id": "ev_test123"}], fetchall=[[switchyard], [succeeded]])
    _override_conn(conn)

    response = client.get("/v1/evaluations/ev_test123/events/stream")

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [(event["event"], event["data"].get("status")) for event in events] == [
        ("switchyard", "succeeded"),
        ("status", "succeeded"),
    ]


def test_events_stream_404_when_evaluation_missing() -> None:
    _override_conn(_conn_with_fetchone(None))

    response = client.get("/v1/evaluations/ev_nope/events/stream")

    assert response.status_code == 404


def test_events_stream_returns_429_at_process_connection_limit(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "api_sse_max_connections", 1)
    evaluations_router._sse_active_connections = 1
    _override_conn(_conn_with_fetchone({"id": "ev_test123"}))

    response = client.get("/v1/evaluations/ev_test123/events/stream")

    assert response.status_code == 429
    assert response.json()["detail"]["error"]["code"] == "stream_limit_exceeded"


def test_events_stream_releases_database_between_polls(monkeypatch) -> None:  # noqa: ANN001
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("scaled_evals.api.routers.evaluations.asyncio.sleep", no_sleep)
    queued = {"id": 1, "type": "status", "status": "queued", "detail": None, "created_at": NOW}
    succeeded = {
        "id": 2,
        "type": "status",
        "status": "succeeded",
        "detail": "1/1 trials completed",
        "created_at": NOW + timedelta(microseconds=1),
    }
    conn = _conn(
        fetchone=[{"id": "ev_test123"}, {"status": "running"}],
        fetchall=[[queued], [], [succeeded]],
    )
    active = 0
    checkouts = 0

    @contextmanager
    def stream_db() -> Iterator[Database]:
        nonlocal active, checkouts
        assert active == 0, "an SSE poll retained its previous database checkout"
        active += 1
        checkouts += 1
        try:
            yield Database(conn)
        finally:
            active -= 1

    v1.dependency_overrides[get_stream_database_factory] = lambda: stream_db

    response = client.get("/v1/evaluations/ev_test123/events/stream")

    assert response.status_code == 200
    assert active == 0
    assert checkouts == 5
    assert evaluations_router._sse_active_connections == 0
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_get_artifact_rejects_traversal_path() -> None:
    _override_conn(_conn_with_fetchone({"id": "ev_test123"}))

    # Percent-encoded so the traversal reaches the route instead of being
    # normalized away by the HTTP client.
    response = client.get("/v1/evaluations/ev_test123/artifacts/..%2F..%2Fev_other%2Fsecret")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_list_artifacts_reads_s3_and_supports_prefix(monkeypatch) -> None:  # noqa: ANN001
    _override_conn(_conn_with_fetchone({"id": "ev_test123"}))

    listed_prefixes: list[str] = []

    def fake_list_objects(prefix: str) -> list[dict]:
        listed_prefixes.append(prefix)
        return [
            {
                "key": "evaluations/ev_test123/artifacts/trial/result.json",
                "size_bytes": 42,
                "updated_at": "2026-06-15T12:00:00+00:00",
            }
        ]

    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.list_objects", fake_list_objects)

    response = client.get("/v1/evaluations/ev_test123/artifacts", params={"prefix": "trial/"})

    assert response.status_code == 200
    assert listed_prefixes == ["evaluations/ev_test123/artifacts/trial/"]
    assert response.json() == {
        "data": [
            {
                "path": "trial/result.json",
                "size_bytes": 42,
                "updated_at": "2026-06-15T12:00:00+00:00",
                "links": {
                    "download": "/evaluations/ev_test123/artifacts/trial/result.json",
                },
            }
        ],
        "next_cursor": None,
    }


def test_list_artifacts_404_when_evaluation_unknown(monkeypatch) -> None:  # noqa: ANN001
    _override_conn(_conn_with_fetchone(None))
    list_objects = MagicMock(return_value=[])
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.list_objects", list_objects)

    response = client.get("/v1/evaluations/ev_missing/artifacts")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"
    assert list_objects.call_count == 0


def test_get_artifact_streams_content(monkeypatch) -> None:  # noqa: ANN001
    conn = _conn_with_fetchone({"id": "ev_test123"})
    _override_conn(conn)
    checkout_active = False

    @contextmanager
    def stream_db() -> Iterator[Database]:
        nonlocal checkout_active
        checkout_active = True
        try:
            yield Database(conn)
        finally:
            checkout_active = False

    def stream_object(_key: str) -> Iterator[bytes]:
        assert not checkout_active, "artifact download retained its database checkout"
        yield b"artifact content"

    v1.dependency_overrides[get_stream_database_factory] = lambda: stream_db
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.stream_object", stream_object)

    response = client.get(
        "/v1/evaluations/ev_test123/artifacts/trial/result.json",
    )

    assert response.status_code == 200
    assert response.content == b"artifact content"
    assert response.headers["content-disposition"] == 'attachment; filename="result.json"'


def test_get_archive_missing(monkeypatch) -> None:  # noqa: ANN001
    _override_conn(_conn_with_fetchone(_archive_row()))
    presign = MagicMock(return_value="http://signed.example/archive")
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.presign_get", presign)

    response = client.get("/v1/evaluations/ev_test123/archive")

    assert response.status_code == 200
    assert response.json() == {
        "evaluation_id": "ev_test123",
        "status": "missing",
        "format": "tar.gz",
        "size_bytes": None,
        "built_at": None,
        "error": None,
        "download": None,
    }
    assert presign.call_count == 0


def test_get_archive_building() -> None:
    _override_conn(_conn_with_fetchone(_archive_row(archive_status="building")))

    response = client.get("/v1/evaluations/ev_test123/archive")

    assert response.status_code == 200
    assert response.json()["status"] == "building"
    assert response.json()["download"] is None


def test_get_archive_ready_presigns_download(monkeypatch) -> None:  # noqa: ANN001
    built_at = datetime(2026, 6, 15, 12, tzinfo=UTC)
    _override_conn(
        _conn_with_fetchone(
            _archive_row(
                archive_status="ready",
                archive_object_key="evaluations/ev_test123/results.tar.gz",
                archive_size_bytes=2048,
                archive_built_at=built_at,
            )
        )
    )
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations.s3.presign_get",
        lambda key: f"http://signed.example/{key}",
    )

    response = client.get("/v1/evaluations/ev_test123/archive")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["size_bytes"] == 2048
    assert body["built_at"] == "2026-06-15T12:00:00Z"
    assert body["download"] == {
        "method": "GET",
        "url": "http://signed.example/evaluations/ev_test123/results.tar.gz",
    }


def test_get_archive_ready_uses_api_download_when_backend_cannot_presign(monkeypatch) -> None:  # noqa: ANN001
    _override_conn(
        _conn_with_fetchone(
            _archive_row(
                archive_status="ready",
                archive_object_key="evaluations/ev_test123/results.tar.gz",
            )
        )
    )
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.can_presign_get", lambda: False)
    presign = MagicMock()
    monkeypatch.setattr("scaled_evals.api.routers.evaluations.s3.presign_get", presign)

    response = client.get("/v1/evaluations/ev_test123/archive")

    assert response.status_code == 200
    assert response.json()["download"] == {
        "method": "GET",
        "url": "/evaluations/ev_test123/archive/download",
    }
    assert presign.call_count == 0


def test_get_archive_404_when_evaluation_unknown() -> None:
    _override_conn(_conn_with_fetchone(None))

    response = client.get("/v1/evaluations/ev_missing/archive")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_post_archive_404_when_evaluation_unknown() -> None:
    _override_conn(_conn_with_fetchone(None))

    response = client.post("/v1/evaluations/ev_missing/archive", json={})

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_post_archive_enqueues_build_and_commits() -> None:
    conn = _conn_with_fetchone(_archive_row(), _archive_row(archive_status="building"))
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/archive", json={})

    assert response.status_code == 202
    assert response.json()["status"] == "building"
    assert conn.commit.called
    update_sql = conn.cursor.return_value.__enter__.return_value.execute.call_args_list[1].args[0]
    assert "archive_status = 'building'" in update_sql


def test_post_archive_ready_without_force_returns_existing_download(monkeypatch) -> None:  # noqa: ANN001
    conn = _conn_with_fetchone(
        _archive_row(
            archive_status="ready",
            archive_object_key="evaluations/ev_test123/results.tar.gz",
            archive_size_bytes=99,
        )
    )
    _override_conn(conn)
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations.s3.presign_get",
        lambda _key: "http://signed.example/archive",
    )

    response = client.post("/v1/evaluations/ev_test123/archive", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["download"]["url"] == "http://signed.example/archive"
    assert len(conn.cursor.return_value.__enter__.return_value.execute.call_args_list) == 1


def test_post_archive_force_requeues_ready_archive() -> None:
    conn = _conn_with_fetchone(
        _archive_row(archive_status="ready", archive_object_key="old", archive_size_bytes=99),
        _archive_row(archive_status="building"),
    )
    _override_conn(conn)

    response = client.post("/v1/evaluations/ev_test123/archive", json={"force": True})

    assert response.status_code == 202
    assert response.json()["status"] == "building"
    params = conn.cursor.return_value.__enter__.return_value.execute.call_args_list[1].args[1]
    assert params[:3] == (True, True, True)


def test_logs_fallback_dispatch_log_path(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "gym_sandbox_daytona_work_dir", str(tmp_path))
    log_path = tmp_path / "ev_test123" / "gym.log"
    log_path.parent.mkdir()
    log_path.write_text("launched gym-runner\n")
    _override_conn(
        _conn_with_fetchone(
            _eval_row(
                status="running",
                runtime="gym_sandbox_daytona",
                backend_handle="ev_test123",
            )
        )
    )

    response = client.get("/v1/evaluations/ev_test123/logs")

    assert response.status_code == 200
    assert response.json()["lines"] == ["launched gym-runner"]


def test_logs_include_ng_run_for_gym(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "gym_sandbox_daytona_work_dir", str(tmp_path))
    work = tmp_path / "ev_test123"
    work.mkdir()
    (work / "gym.log").write_text("meta\n")
    (work / "ng_run.log").write_text("ray start\nharbor_agent ready\n")
    _override_conn(
        _conn_with_fetchone(_eval_row(status="running", runtime="gym_sandbox_daytona", backend_handle="ev_test123"))
    )

    response = client.get("/v1/evaluations/ev_test123/logs", params={"tail_lines": 10})

    assert response.status_code == 200
    lines = response.json()["lines"]
    assert "--- gym.log ---" in lines
    assert "meta" in lines
    assert "--- ng_run.log ---" in lines
    assert "harbor_agent ready" in lines


def test_logs_include_switchyard_capture_files(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "gym_daytona_work_dir", str(tmp_path))
    work = tmp_path / "ev_test123" / "switchyard"
    work.mkdir(parents=True)
    (work / "switchyard.log").write_text("switchyard accepted request\n")
    (work / "switchyard.previous.log").write_text("switchyard restarted once\n")
    (work / "status.json").write_text('{"availableReplicas":1}\n')
    _override_conn(_conn_with_fetchone(_eval_row(status="running", runtime="gym_daytona", backend_handle="ev_test123")))

    response = client.get("/v1/evaluations/ev_test123/logs", params={"tail_lines": 20})

    assert response.status_code == 200
    lines = response.json()["lines"]
    assert "--- switchyard.log ---" in lines
    assert "switchyard accepted request" in lines
    assert "--- switchyard.previous.log ---" in lines
    assert "switchyard restarted once" in lines
    assert "--- status.json ---" in lines


def test_log_stream_returns_sse_snapshot() -> None:
    row = _eval_row(status="succeeded", result={"logs": ["done"]})
    _override_conn(_conn_with_fetchone(row, row))

    response = client.get("/v1/evaluations/ev_test123/logs/stream")

    assert response.status_code == 200
    assert "event: log" in response.text
    assert '"line": "done"' in response.text
    assert "event: status" in response.text


def test_log_stream_returns_404_before_starting_unknown_stream() -> None:
    _override_conn(_conn_with_fetchone(None))

    response = client.get("/v1/evaluations/ev_missing/logs/stream")

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


# ---------- dispatch worker (fake backend, no cluster) --------------------


class _FakeBackend:
    """A cluster-free backend. ``launch`` records the spec; ``status`` returns a
    scripted sequence of RuntimeStatus values (last one repeats once exhausted),
    so a test can drive running -> succeeded/failed without any cluster.
    """

    name = "fake"

    def __init__(self, statuses: list[RuntimeStatus] | None = None) -> None:
        self.launched: list[LaunchSpec] = []
        self.status_handles: list[LaunchHandle] = []
        self.teardown_handles: list[LaunchHandle] = []
        self.fail = False
        self.status_error: Exception | None = None
        self.teardown_error: Exception | None = None
        self.teardown_errors: list[Exception | None] = []
        # Default: a single poll that finds the run finished with a result.
        self._statuses = statuses or [RuntimeStatus(phase="succeeded", raw=SAMPLE_HARBOR_RESULT)]
        self.status_calls = 0

    def launch(self, spec: LaunchSpec) -> LaunchHandle:
        self.launched.append(spec)
        if self.fail:
            raise RuntimeError("boom")
        return LaunchHandle(backend=self.name, external_id="hbr-ev-test123")

    def status(self, handle) -> RuntimeStatus:  # noqa: ANN001
        self.status_handles.append(handle)
        if self.status_error is not None:
            raise self.status_error
        idx = min(self.status_calls, len(self._statuses) - 1)
        self.status_calls += 1
        return self._statuses[idx]

    def teardown(self, handle):  # noqa: ANN001
        self.teardown_handles.append(handle)
        if self.teardown_errors:
            if error := self.teardown_errors.pop(0):
                raise error
            return
        if self.teardown_error is not None:
            raise self.teardown_error

    def summarize(self, result):  # noqa: ANN001, ANN201
        # Mirrors the real backends: reduce the (Harbor-shaped) result envelope.
        return summarize_harbor_result(result)


class _FakeSwitchyardProvisioner:
    def __init__(self) -> None:
        self.provisions: list[dict] = []
        self.captures: list[str] = []
        self.capture_session_ids: list[tuple[str, ...]] = []
        self.deletes: list[str] = []
        self.delete_artifact_roots: list[Path | None] = []
        self.ensure_ready_checks: list[str] = []
        self.ensure_ready_error: Exception | None = None
        self.ensure_ready_errors: list[Exception | None] = []
        self.invoke_persist_lease = False
        self.provision_error: Exception | None = None

    def provision(
        self,
        *,
        evaluation_id: str,
        profile_id: str,
        raw_config,
        credential_env,
        artifact_root: Path,
        benchmark_run_id=None,
        persist_lease=None,
    ) -> SwitchyardRender:
        self.provisions.append(
            {
                "evaluation_id": evaluation_id,
                "profile_id": profile_id,
                "raw_config": dict(raw_config),
                "credential_env": dict(credential_env),
                "benchmark_run_id": benchmark_run_id,
            }
        )
        render = render_switchyard(
            evaluation_id=evaluation_id,
            profile_id=profile_id,
            config=SwitchyardProfileConfig.model_validate(raw_config),
            credential_env=credential_env,
            artifact_root=artifact_root,
            benchmark_run_id=benchmark_run_id,
            external_allowed_hosts=("switchyard.example.com",),
        )
        write_switchyard_artifacts(render, artifact_root)
        if self.invoke_persist_lease and persist_lease is not None:
            persist_lease(render.lease)
        if self.provision_error is not None:
            raise self.provision_error
        return render

    def capture(  # noqa: ANN001, ANN201
        self, lease, artifact_root: Path, *, final=False, session_ids=()
    ):
        self.captures.append(lease.name)
        self.capture_session_ids.append(tuple(session_ids))
        root = artifact_root / "switchyard"
        root.mkdir(parents=True, exist_ok=True)
        (root / "switchyard.log").write_text("switchyard routed request\n")
        (root / "status.json").write_text('{"ready":true}\n')
        if final:
            sessions = {
                session_id: {
                    "session_id": session_id,
                    "total_calls": 1,
                    "total_prompt_tokens": 1,
                    "total_cached_tokens": 0,
                    "total_cache_creation_tokens": 0,
                    "total_completion_tokens": 1,
                    "models": {
                        "nvidia/unknown-test-model": {
                            "calls": 1,
                            "prompt_tokens": 1,
                            "cached_tokens": 0,
                            "cache_creation_tokens": 0,
                            "completion_tokens": 1,
                        }
                    },
                }
                for session_id in session_ids
            }
            (root / "routing_stats_final.json").write_text(json.dumps({"requests": 1, "sessions": sessions}) + "\n")
        return None

    def delete(self, lease, artifact_root: Path | None = None) -> None:  # noqa: ANN001
        self.deletes.append(lease.name)
        self.delete_artifact_roots.append(artifact_root)

    def ensure_ready(self, lease) -> None:  # noqa: ANN001
        self.ensure_ready_checks.append(lease.name)
        if self.ensure_ready_errors:
            if error := self.ensure_ready_errors.pop(0):
                raise error
            return
        if self.ensure_ready_error is not None:
            raise self.ensure_ready_error


def _worker_conn(row: dict | None) -> tuple[MagicMock, list]:
    """Connection whose load returns `row`; returns (conn, executed-calls)."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = row
    cur.fetchall.return_value = []
    return conn, cur.execute.call_args_list


def _worker_conn_fetches(*rows: dict | None) -> tuple[MagicMock, list]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    values = iter(rows)
    cur.fetchone.side_effect = lambda: next(values, {"status": "running"})
    cur.fetchall.return_value = []
    return conn, cur.execute.call_args_list


def _status_updates(execute_calls) -> list[str]:
    """Extract the status set by each UPDATE evaluations call."""
    updates = []
    for call in execute_calls:
        if hasattr(call, "args"):
            sql = call.args[0]
            params = call.args[1]
        else:
            sql, params = call
        if "UPDATE evaluations" in sql and "SET status" in sql:
            updates.append(params[0])
    return updates


def _status_update_params(execute_calls, status: str) -> tuple:
    for call in execute_calls:
        if hasattr(call, "args"):
            sql = call.args[0]
            params = call.args[1]
        else:
            sql, params = call
        if "UPDATE evaluations" in sql and "SET status" in sql and params[0] == status:
            return params
    raise AssertionError(f"no {status} status UPDATE was issued")


def _result_update_params(execute_calls) -> tuple:
    """Params of the UPDATE that writes the result envelope (the SET result one)."""
    for call in execute_calls:
        if hasattr(call, "args"):
            sql = call.args[0]
            params = call.args[1]
        else:
            sql, params = call
        if "UPDATE evaluations" in sql and "result = %s" in sql:
            return params
    raise AssertionError("no result write-back UPDATE was issued")


def _event_inserts(execute_calls) -> list[tuple]:
    inserts = []
    for call in execute_calls:
        if hasattr(call, "args"):
            sql = call.args[0]
            params = call.args[1]
        else:
            sql, params = call
        if "INSERT INTO evaluation_events" in sql:
            if len(params) == 4:
                inserts.append((params[0], params[2], params[3]))
            else:
                inserts.append(params)
    return inserts


def _typed_event_inserts(execute_calls) -> list[tuple]:
    inserts = []
    for call in execute_calls:
        if hasattr(call, "args"):
            sql = call.args[0]
            params = call.args[1]
        else:
            sql, params = call
        if "INSERT INTO evaluation_events" in sql:
            if len(params) == 4:
                inserts.append(params)
            else:
                inserts.append((params[0], "status", params[1], params[2]))
    return inserts


def _event_rows_from_inserts(inserts: list[tuple]) -> list[dict]:
    return [
        {
            "id": idx,
            "type": "status",
            "evaluation_id": evaluation_id,
            "status": status,
            "detail": detail,
            "created_at": NOW + timedelta(microseconds=idx),
        }
        for idx, (evaluation_id, status, detail) in enumerate(inserts, start=1)
    ]


def _dispatcher(backend: _FakeBackend, conn: MagicMock) -> Dispatcher:
    @contextmanager
    def connect():
        yield conn

    # Fail loudly if the poll loop ever sleeps in a unit test (it shouldn't —
    # the fake reaches a terminal status without waiting).
    def no_sleep(_seconds: float) -> None:
        raise AssertionError("unit tests must not sleep in the poll loop")

    return Dispatcher(resolve=lambda runtime: backend, connect=connect, sleep=no_sleep)


def _context(conn: MagicMock):
    @contextmanager
    def connect():
        yield conn

    return connect


# ---------- sandbox_k8s backend (no cluster) ------------------------------


def _spec() -> LaunchSpec:
    return LaunchSpec(
        evaluation_id="ev_test123",
        name="run 7",
        framework="harbor",
        image_ref="registry.example/task_abc:rev2",
        image_digest=f"sha256:{'a' * 64}",
        parallelism=4,
    )


def _spec_with_credential_env(credential_env: dict[str, str]) -> LaunchSpec:
    return LaunchSpec(**{**_spec().__dict__, "credential_env": credential_env})


def _spec_with_initial_user_turns(initial_user_turns: list[str]) -> LaunchSpec:
    return LaunchSpec(**{**_spec().__dict__, "initial_user_turns": initial_user_turns})


_USER_TOKEN_CONFIG = """\
job_name: x
environment:
  import_path: sandbox_k8s.harbor:K8sSandboxEnvironment
  kwargs:
    namespace: ${SANDBOX_NAMESPACE}
    kubeconfig_path: ${HOME}/.kube/config
    context: ${SANDBOX_CONTEXT}
    image: ${TASK_IMAGE}
    verify_ssl: ${VERIFY_SSL}
"""


# ---------- sandbox_k8s live submitter (no subprocess, no cluster) --------

_ORACLE_CONFIG = """\
job_name: astra-hello-skills-oracle
jobs_dir: jobs/astra
environment:
  import_path: sandbox_k8s.harbor:K8sSandboxEnvironment
  kwargs:
    namespace: ${SANDBOX_NAMESPACE}
    image: ${TASK_IMAGE}
    verify_ssl: ${VERIFY_SSL}
"""


_IMAGE_CONFIG = """\
job_name: astra-oracle
jobs_dir: jobs/astra
environment:
  import_path: sandbox_k8s.harbor:K8sSandboxEnvironment
  kwargs:
    namespace: ${SANDBOX_NAMESPACE}
    image: ${TASK_IMAGE}
    image_pull_policy: Always
    image_pull_secrets: regcred
    verify_ssl: ${VERIFY_SSL}
"""


# ---------- sandbox_k8s task tree staged from the task upload --------

_TASKS_CONFIG = _IMAGE_CONFIG + "tasks:\n  - path: ${BROKEN_PYTHON_TASK_PATH}\n"


def _spec_with_tarball(tarball_object_key: str) -> LaunchSpec:
    return LaunchSpec(**{**_spec().__dict__, "tarball_object_key": tarball_object_key})


def _make_task_pack(path: Path, *, root_prefix: str = "task", include_task: bool = True) -> Path:
    """Write a gzip pack like the one uploaded for a task revision.

    Always carries a root ``Dockerfile`` (the build context); the Harbor task
    tree (``task.toml`` + tests/solution/instruction) is included under
    ``root_prefix`` unless ``include_task`` is False (a Dockerfile-only pack).
    """
    import io

    with tarfile.open(path, "w:gz") as tar:

        def _add(name: str, content: str) -> None:
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add("Dockerfile", "FROM python:3.13-slim\n")
        if include_task:
            _add(f"{root_prefix}/task.toml", "version = '1.0'\n")
            _add(f"{root_prefix}/instruction.md", "fix it\n")
            _add(f"{root_prefix}/tests/test.sh", "echo ok\n")
            _add(f"{root_prefix}/solution/solve.sh", "echo solved\n")
    return path


def _fake_download(pack: Path):  # noqa: ANN202
    import shutil

    def _dl(object_key: str, dest_path: str) -> None:  # noqa: ANN001
        shutil.copyfile(pack, dest_path)

    return _dl


# ---------- gym_daytona backend (no Daytona account) ----------------------


# ---------- gym_sandbox_daytona backend (nemo_gym.sandbox) -----------------


# ---------- gym_sandbox_opensandbox backend (nemo_gym.sandbox) ------------


# ---------- result summary + status reader (no cluster) -------------------
