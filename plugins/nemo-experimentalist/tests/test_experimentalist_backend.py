# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend fetch/archive/publish tests.

Covers insight resolution, the git vs local dispatch in ``get_agent_code``, and the
``archive_candidate`` / ``publish_candidate`` verbs (which derive branch/base_ref/agent_path
from the captured ``AgentSource`` and delegate to ``PRPublisher``) — with the git/subprocess
boundary mocked.
"""

import asyncio
import json
import logging
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from doubles import fake_client, make_candidate
from nemo_experimentalist_plugin.entities import Candidate
from nemo_experimentalist_plugin.experimentalist import experimentalist_backend as beim
from nemo_experimentalist_plugin.experimentalist.components.repository import AgentCloneError, AgentSource
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import (
    CandidateStorageConfig,
    LocalExperimentalistBackend,
)
from nemo_insights_plugin.entities import Insight
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NotFoundError


def _local_backend(tmp_path: Path) -> LocalExperimentalistBackend:
    return LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")


# ---------------------------------------------------------------------------
# get_insight — local file fixture vs platform id fetch
# ---------------------------------------------------------------------------


class _StubInsightsClient:
    def __init__(self, insight: Insight | None, error: Exception | None = None) -> None:
        self._insight = insight
        self._error = error
        self.calls: list[dict[str, str]] = []

    async def get_insight(self, *, workspace: str, insight_id: str) -> Insight:
        self.calls.append({"workspace": workspace, "insight_id": insight_id})
        if self._error is not None:
            raise self._error
        assert self._insight is not None
        return self._insight


async def test_get_insight_reads_local_file(tmp_path: Path) -> None:
    insight_file = tmp_path / "insight.json"
    insight_file.write_text(
        json.dumps({"id": "insight-local", "workspace": "w", "title": "t", "description": "d", "agent": "a"})
    )
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")

    insight = await backend.get_insight(workspace="w", insight_id=str(insight_file))

    assert insight.id == "insight-local"


async def test_get_insight_fetches_platform_id_via_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    platform_insight = Insight(workspace="ws", title="platform", description="d", agent="a")
    insights_client = _StubInsightsClient(platform_insight)
    monkeypatch.setattr(beim.AsyncInsightsClient, "from_client", lambda _client: insights_client)
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")

    insight = await backend.get_insight(workspace="ws", insight_id="insight-remote-123")

    assert insight is platform_insight
    assert insights_client.calls == [{"workspace": "ws", "insight_id": "insight-remote-123"}]


async def test_get_insight_platform_404_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    error = NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://platform.test/insights/missing")))
    monkeypatch.setattr(beim.AsyncInsightsClient, "from_client", lambda _client: _StubInsightsClient(None, error))
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")

    with pytest.raises(ValueError, match="Insight not found on the platform"):
        await backend.get_insight(workspace="w", insight_id="missing")


# ---------------------------------------------------------------------------
# get_agent_code — local path (unchanged behavior, returns None)
# ---------------------------------------------------------------------------


async def test_get_agent_code_local_copies_returns_none(tmp_path: Path) -> None:
    src = tmp_path / "agent"
    src.mkdir()
    (src / "main.py").write_text("print('x')\n")
    dest = tmp_path / "dest"
    source = await _local_backend(tmp_path).get_agent_code(workspace="w", agent=src, dest=dest)
    assert source is None
    assert (dest / "main.py").read_text() == "print('x')\n"


# ---------------------------------------------------------------------------
# get_agent_code — git source (dispatches to clone_agent_repo, returns AgentSource)
# ---------------------------------------------------------------------------


def test_get_agent_code_git_dispatches_to_clone(tmp_path, monkeypatch):  # noqa: ANN001
    calls: list[tuple[str, Path]] = []

    def fake_clone(spec: str, dest: Path, *, clone_depth: int | None = None) -> AgentSource:
        calls.append((spec, Path(dest)))
        Path(dest).mkdir(parents=True, exist_ok=True)
        return AgentSource(repo_url="ssh://git@h/g/r.git", ref="main")

    monkeypatch.setattr(beim, "clone_agent_repo", fake_clone)

    backend = _local_backend(tmp_path)
    spec = "ssh://git@h/g/r.git@main"
    dest = tmp_path / "agent-src"

    source = asyncio.run(backend.get_agent_code(workspace="w", agent=spec, dest=dest))
    assert source == AgentSource(repo_url="ssh://git@h/g/r.git", ref="main")
    assert calls == [(spec, dest)]


async def test_get_agent_code_git_preserves_owned_clone_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "git clone failed for https://***@gitlab.com/org/repo.git (exit status 128)"
    failure = AgentCloneError(message)

    def fail_clone(spec: str, *_args, **_kwargs) -> AgentSource:  # noqa: ANN002, ANN003
        raise failure

    monkeypatch.setattr(beim, "clone_agent_repo", fail_clone)

    with pytest.raises(AgentCloneError) as exc_info:
        await _local_backend(tmp_path).get_agent_code(
            workspace="w",
            agent="https://gitlab.com/org/repo.git",
            dest=tmp_path / "dest",
        )

    assert exc_info.value is failure
    assert str(exc_info.value) == message
    assert exc_info.value.__cause__ is None


async def test_get_agent_code_git_suppresses_legacy_called_process_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = beim.subprocess.CalledProcessError(
        128,
        ["git", "clone", "command-secret"],
        output="stdout-secret",
        stderr="stderr-secret",
    )

    def fail_clone(spec: str, *_args, **_kwargs) -> AgentSource:  # noqa: ANN002, ANN003
        raise failure

    monkeypatch.setattr(beim, "clone_agent_repo", fail_clone)
    secrets = ("token-user", "secret-token")
    source = f"https://{secrets[0]}:{secrets[1]}@gitlab.com/org/repo.git@main"

    with pytest.raises(ValueError) as exc_info:
        await _local_backend(tmp_path).get_agent_code(workspace="w", agent=source, dest=tmp_path / "dest")

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert str(exc_info.value) == "failed to fetch --agent 'https://***@gitlab.com/org/repo.git'"
    assert not any(secret in formatted for secret in (*secrets, "command-secret", "stdout-secret", "stderr-secret"))
    assert exc_info.value.__cause__ is None


# ---------------------------------------------------------------------------
# get_ethos — local path copies to dest and returns dest
# ---------------------------------------------------------------------------


async def test_get_ethos_local_copies_to_dest_and_returns_dest(tmp_path: Path) -> None:
    ethos_file = tmp_path / "ETHOS.md"
    ethos_file.write_text("# My Agent\nDoes things.\n")
    dest = tmp_path / "workspace" / "ETHOS.md"

    result = await _local_backend(tmp_path).get_ethos(workspace="w", ethos=str(ethos_file), dest=dest)

    assert result == dest
    assert dest.read_text() == "# My Agent\nDoes things.\n"


async def test_get_ethos_local_missing_raises(tmp_path: Path) -> None:
    dest = tmp_path / "workspace" / "ETHOS.md"
    with pytest.raises(FileNotFoundError, match="Ethos not found"):
        await _local_backend(tmp_path).get_ethos(workspace="w", ethos=str(tmp_path / "nonexistent.md"), dest=dest)


async def test_get_ethos_directory_raises(tmp_path: Path) -> None:
    dest = tmp_path / "workspace" / "ETHOS.md"
    with pytest.raises(FileNotFoundError, match="Ethos not found"):
        await _local_backend(tmp_path).get_ethos(workspace="w", ethos=str(tmp_path), dest=dest)


# ---------------------------------------------------------------------------
# archive_candidate / publish_candidate — clean (workspace, candidate) verbs.
# The backend derives branch/base_ref/agent_path/title/body from the captured git
# provenance + storage config; the git/subprocess boundary (PRPublisher) is stubbed.
# ---------------------------------------------------------------------------


def _cand(label: str = "agent-2", run_id: str = "run-1") -> Candidate:
    return make_candidate(label=label, run_id=run_id, generation=1, ancestor="agent-0", description="edit")


def _git_backend(tmp_path: Path, storage: CandidateStorageConfig | None = None) -> LocalExperimentalistBackend:
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend", storage=storage)
    backend._agent_source = AgentSource(repo_url="ssh://git@h/g/r.git", ref="main", agent_path="pkg/agent")
    backend._agent_checkout = tmp_path / "clone"
    return backend


async def test_archive_candidate_git_pushes_and_records_source_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _StubPublisher:
        def __init__(self, *, agent_dir: Path) -> None:
            captured["agent_dir"] = agent_dir

        def push_branch(self, *, src_dir, branch, base_ref, message, agent_path="."):  # noqa: ANN001, ANN202
            captured.update(branch=branch, base_ref=base_ref, agent_path=agent_path)
            return True

    monkeypatch.setattr(beim, "PRPublisher", _StubPublisher)
    backend = _git_backend(tmp_path, CandidateStorageConfig(candidate_branch_prefix="optimizer"))

    # Clean verb: only (workspace, candidate); branch/base_ref/agent_path are derived internally.
    link = await backend.archive_candidate(workspace="w", candidate=_cand())
    assert captured == {
        "agent_dir": tmp_path / "clone",
        "branch": "optimizer/run-1/agent-2",
        "base_ref": "main",
        "agent_path": "pkg/agent",
    }
    assert link == "ssh://git@h/g/r.git@optimizer/run-1/agent-2"
    # The link is recorded so re-projection surfaces the real branch, not the pseudo placeholder.
    assert backend._candidate_source_links["agent-2"] == link


async def test_publish_candidate_git_delegates_and_stashes_pr_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _StubPublisher:
        def __init__(self, *, agent_dir: Path) -> None:
            captured["agent_dir"] = agent_dir

        def publish(
            self, *, winner_dir, branch, base_ref, title, body, draft, agent_path, base_branch_override, labels
        ):  # noqa: ANN001, ANN202
            captured.update(branch=branch, base_ref=base_ref, agent_path=agent_path, draft=draft, labels=labels)
            return "https://gitlab.example.com/g/r/-/merge_requests/7"

    monkeypatch.setattr(beim, "PRPublisher", _StubPublisher)
    storage = CandidateStorageConfig(candidate_branch_prefix="optimizer", pr_draft=False, pr_labels=["opt"])
    backend = _git_backend(tmp_path, storage)

    url = await backend.publish_candidate(workspace="w", candidate=_cand(label="agent-1"))
    assert url == "https://gitlab.example.com/g/r/-/merge_requests/7"
    assert captured["branch"] == "optimizer/run-1/agent-1" and captured["base_ref"] == "main"
    assert captured["agent_path"] == "pkg/agent" and captured["draft"] is False and captured["labels"] == ["opt"]
    # PR URL stashed so the mirror's finalize() records it on the winner's Experiment source_link.
    assert backend._pr_url == url


async def test_archive_and_publish_noop_for_non_git_source(tmp_path: Path) -> None:
    # No captured git source (local agent) -> both verbs no-op returning None.
    backend = _local_backend(tmp_path)
    assert await backend.archive_candidate(workspace="w", candidate=_cand()) is None
    assert await backend.publish_candidate(workspace="w", candidate=_cand()) is None


async def test_persist_result_writes_run_summary(tmp_path: Path) -> None:
    # persist_result must write result.summary into run.json, else publish_candidate's
    # _compose_pr_body falls back to "Experimentalist run complete." instead of the real summary.
    from nemo_experimentalist_plugin.entities import ExperimentRun
    from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult

    backend = _local_backend(tmp_path)
    run = ExperimentRun(workspace="w", agent="a")
    run._id = "run-1"  # type: ignore[attr-defined]
    (backend._eo / "run.json").write_text(run.model_dump_json(indent=2))

    result = ExperimentalistResult(summary="the real run summary", run_id="run-1", progress_completed=2, winner=None)
    await backend.persist_result(workspace="w", result=result)

    saved = json.loads((backend._eo / "run.json").read_text())
    assert saved["summary"] == "the real run summary"
    assert (backend._eo / "OPTIMIZATION.md").read_text() == "the real run summary"


async def test_persist_result_names_the_winner_by_label(tmp_path: Path) -> None:
    """``winner_agent`` must hold 'agent-1', not the candidate id.

    Everything a run writes is filed under the label -- ``agents/agent-1``,
    ``results/agent-1-validation`` -- so this field exists to be used as a path
    component. Writing the id gives a reference that resolves to nothing, and the
    empty lookup downstream reads as "the winner has no results" rather than as a
    broken reference; a smoke-agent gate reported a passing control as a regression
    on exactly that.
    """
    from doubles import make_candidate
    from nemo_experimentalist_plugin.entities import ExperimentRun
    from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult

    backend = _local_backend(tmp_path)
    run = ExperimentRun(workspace="w", agent="a")
    run._id = "run-1"  # type: ignore[attr-defined]
    (backend._eo / "run.json").write_text(run.model_dump_json(indent=2))

    winner = make_candidate(label="agent-1", generation=1, ancestor="agent-0")
    result = ExperimentalistResult(summary="s", run_id="run-1", progress_completed=1, winner=winner)
    await backend.persist_result(workspace="w", result=result)

    saved = json.loads((backend._eo / "run.json").read_text())
    assert saved["winner_agent"] == "agent-1"
    assert saved["winner_agent"] != winner.id


async def test_persist_result_preserves_generated_optimization_report(tmp_path: Path) -> None:
    from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult

    backend = _local_backend(tmp_path)
    report_path = backend._eo / "OPTIMIZATION.md"
    report_path.write_text("# Full optimization report\n\nAdditional Metrics")

    result = ExperimentalistResult(summary="compact run summary", run_id="run-1", progress_completed=2, winner=None)
    await backend.persist_result(workspace="w", result=result)

    assert report_path.read_text() == "# Full optimization report\n\nAdditional Metrics"


# ---------------------------------------------------------------------------
# ATIF upload
# ---------------------------------------------------------------------------


def _atif_ref(tmp_path, session_id="sess-1"):
    from nemo_experimentalist_plugin.entities import ResourceRef

    path = tmp_path / f"trajectory-{session_id}.atif.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": session_id,
                "agent": {"name": "Codeact", "version": "0.0.0"},
                "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
            }
        ),
        encoding="utf-8",
    )
    return ResourceRef(uri=f"file://{path}", description="", metadata={"trace_format": "atif"})


class _Response:
    def __init__(self, body: object) -> None:
        self._body = body

    def data(self) -> object:
        return self._body


class _RecordingIntakeClient:
    """Exposes only the async typed Intake methods the trace uploads reach through."""

    def __init__(self) -> None:
        self.traces = SimpleNamespace(calls=[])
        self.atif = SimpleNamespace(calls=[])
        self.evaluator_results: list[dict[str, object]] = []
        self._otlp_result = SimpleNamespace(errors=[])

    async def create_otlp_traces(self, *, content: bytes, workspace: str) -> _Response:
        self.traces.calls.append({"workspace": workspace, "body": content})
        return _Response(self._otlp_result)

    async def create_atif(self, *, workspace: str, body: object) -> _Response:
        self.atif.calls.append({"workspace": workspace, "body": body})
        return _Response(None)

    async def create_evaluator_result(self, *, workspace: str, body: object) -> _Response:
        self.evaluator_results.append({"workspace": workspace, "body": body})
        return _Response(object())


def test_backend_stores_intake_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingIntakeClient()
    factory_calls: list[AsyncNemoClient] = []

    def from_client(platform_client: AsyncNemoClient) -> beim.AsyncIntakeClient:
        factory_calls.append(platform_client)
        return cast(beim.AsyncIntakeClient, client)

    monkeypatch.setattr(beim.AsyncIntakeClient, "from_client", from_client)
    platform_client = fake_client()
    backend = LocalExperimentalistBackend(client=platform_client, path=tmp_path / "backend")

    assert backend.intake_client is client
    assert factory_calls == [platform_client]


def _otlp_ref(tmp_path):
    from nemo_experimentalist_plugin.entities import ResourceRef

    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [{"traceId": "f" * 32, "spanId": "a" * 16}]}]}]})
        + "\n",
        encoding="utf-8",
    )
    return ResourceRef(uri=f"file://{path}", description="", metadata={"trace_format": "otlp"})


async def test_upload_trace_otlp_sends_protobuf_through_the_typed_sdk_method(tmp_path):
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import _upload_trace_otlp

    client = _RecordingIntakeClient()
    await _upload_trace_otlp(
        cast(Any, client),
        "ws-1",
        _otlp_ref(tmp_path),
        evaluation_name="exp-1",
        trial_id="trial-1",
        task_id="case-a",
    )

    assert len(client.traces.calls) == 1
    call = client.traces.calls[0]
    assert call["workspace"] == "ws-1"
    assert isinstance(call["body"], bytes)
    assert b"exp-1" in call["body"]


async def test_ingest_otlp_sends_every_payload_and_stays_quiet_when_intake_accepts(caplog):
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import _ingest_otlp

    client = _RecordingIntakeClient()

    with caplog.at_level(logging.WARNING):
        await _ingest_otlp(cast(Any, client), [b"one", b"two"], workspace="ws-1", trial_id="trial-1")

    assert [call["body"] for call in client.traces.calls] == [b"one", b"two"]
    assert all(call["workspace"] == "ws-1" for call in client.traces.calls)
    assert caplog.text == ""


async def test_upload_trace_otlp_raises_when_intake_rejects_spans(tmp_path):
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import _upload_trace_otlp

    client = _RecordingIntakeClient()
    client._otlp_result = SimpleNamespace(errors=["span 00a1: bad trace_id"])

    with pytest.raises(RuntimeError, match="bad trace_id"):
        await _upload_trace_otlp(
            cast(Any, client),
            "ws-1",
            _otlp_ref(tmp_path),
            evaluation_name="exp-1",
            trial_id="trial-1",
            task_id="case-a",
        )


async def test_persist_trial_leaves_the_local_trace_ref_when_ingest_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A partial ingest must not repoint the trial at a trace that never fully landed."""
    from nemo_experimentalist_plugin.entities import TrialResult

    client = _RecordingIntakeClient()
    client._otlp_result = SimpleNamespace(errors=["span 00a1: bad trace_id"])
    monkeypatch.setattr(beim.AsyncIntakeClient, "from_client", lambda _client: client)
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")
    ref = _otlp_ref(tmp_path)
    trial = TrialResult(id="trial-9", task_id="case-z", status="completed", trace=ref)

    with pytest.raises(RuntimeError):
        await backend._persist_trial(trial, workspace="ws-1", evaluation_name="exp-1", agent_attrs={})

    assert trial.trace is not None and trial.trace.uri == ref.uri


async def _completed(value):
    return value


class _SpanRow:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, **_kwargs):
        return self._payload


class _ReingestClient(_RecordingIntakeClient):
    """Adds the span listing `_persist_trial` reads before it re-ingests."""

    def __init__(self, rows: list[dict]) -> None:
        super().__init__()
        self.list_calls: list[dict] = []
        self._rows = rows

    async def list_spans(self, **kwargs):
        self.list_calls.append(kwargs)

        async def _iter():
            for row in self._rows:
                yield _SpanRow(row)

        return SimpleNamespace(items=_iter)


async def test_persist_trial_reingests_intake_traces_with_evaluation_attributes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    from nemo_experimentalist_plugin.entities import ResourceRef, TrialResult

    row = {
        "span_id": "00000000000000a1",
        "trace_id": "f" * 32,
        "name": "root",
        "started_at": "2026-08-14T00:00:00Z",
    }
    client = _ReingestClient([row])
    monkeypatch.setattr(beim.AsyncIntakeClient, "from_client", lambda _client: client)
    backend = LocalExperimentalistBackend(client=fake_client(), path=tmp_path / "backend")
    # The trace already exists in Intake but carries no evaluation context, which is the
    # condition that triggers the re-ingest.
    monkeypatch.setattr(
        LocalExperimentalistBackend,
        "_retrieve_trace_with_retry",
        lambda self, trace_id, **kwargs: _completed(SimpleNamespace(evaluation_context=None)),
    )
    trial = TrialResult(
        id="trial-9",
        task_id="case-z",
        status="completed",
        trace=ResourceRef(uri="intake://traces/abc123", description="", metadata={}),
    )

    await backend._persist_trial(trial, workspace="ws-1", evaluation_name="exp-1", agent_attrs={})

    assert client.list_calls[0]["workspace"] == "ws-1"
    assert client.list_calls[0]["query_params"]["filter"] == {"trace_id": "abc123"}
    assert len(client.traces.calls) == 1
    body = client.traces.calls[0]["body"]
    assert client.traces.calls[0]["workspace"] == "ws-1"
    assert b"exp-1" in body and b"case-z" in body and b"trial-9" in body


async def test_upload_trace_atif_sends_evaluation_context_and_agent_identity(tmp_path):
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import _upload_trace_atif

    client = _RecordingIntakeClient()
    await _upload_trace_atif(
        cast(Any, client),
        "ws-1",
        _atif_ref(tmp_path),
        evaluation_name="exp-1",
        task_id="case-a",
        extra_attrs={"gen_ai.request.model": "gpt-5-mini"},
    )

    assert len(client.atif.calls) == 1
    call = client.atif.calls[0]
    assert call["workspace"] == "ws-1"
    body = call["body"].root
    assert body["evaluation_context"] == {
        "evaluation_name": "exp-1",
        "test_case_name": "case-a",
    }
    assert body["agent"]["model_name"] == "gpt-5-mini"


async def test_upload_trace_atif_sends_a_trajectory_not_bytes(tmp_path):
    from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import _upload_trace_atif

    client = _RecordingIntakeClient()
    await _upload_trace_atif(cast(Any, client), "ws-1", _atif_ref(tmp_path), evaluation_name="exp-1", task_id="case-a")

    assert client.traces.calls == []
    assert client.atif.calls[0]["body"].root["schema_version"].startswith("ATIF-")


@pytest.mark.parametrize("bad_id", ["../run", "a/b", "..", ".", "nested/../../run"])
async def test_a_candidate_id_that_is_not_one_name_is_refused(tmp_path: Path, bad_id: str) -> None:
    """The id is interpolated into a path, so a separator escapes the candidates directory.

    `../run` lands exactly on `eval-and-optimize/run.json`, so writing that candidate would
    overwrite the run record and make the whole run unresumable. Ids are minted as uuids
    here, but a Candidate can arrive from a component this repo did not write -- which is
    what the registry is for -- so the check belongs at the path.
    """
    backend = _local_backend(tmp_path)

    with pytest.raises(ValueError, match="single path component|required"):
        backend._candidate_path(bad_id)


async def test_a_normal_candidate_id_still_resolves(tmp_path: Path) -> None:
    """The refusal must not fire on the ids the backend actually mints."""
    backend = _local_backend(tmp_path)

    path = backend._candidate_path("8374a3ea-f04d-4203-9f6e-21b56e64f1fc")

    assert path.parent.name == "candidates"
    assert path.name.endswith(".json")
