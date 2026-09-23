# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-API export (bundle capture): export.py + artifact.snapshot_export."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from evaluation import artifact, export
from evaluation.registry import Subject
from nemo_helix import AsyncNeMoHelix

# --------------------------------------------------------------------------- #
# fake SDK client
# --------------------------------------------------------------------------- #


class Doc:
    """Stand-in for an SDK model: dumps like the analyst does (json + exclude_none)."""

    def __init__(self, **payload):
        self.payload = payload

    def model_dump(self, mode="python", exclude_none=False):
        assert mode == "json", "export must dump SDK models with mode='json'"
        assert exclude_none, "export must dump SDK models with exclude_none=True"
        return {k: v for k, v in self.payload.items() if v is not None}

    def __getattr__(self, name):
        return self.payload[name]


def _paginator(items):
    async def gen():
        for item in items:
            yield item

    return gen()


class FakeClient:
    """Captures every list call's kwargs; serves canned docs per (collection, workspace)."""

    def __init__(self, docs: dict):
        self.docs = docs  # {("spans", ws): [Doc, ...], ...}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        outer = self

        class _Collection:
            def __init__(self, name):
                self.name = name

            def list(self, **kwargs):
                outer.calls.append((self.name, kwargs))
                items = outer.docs.get((self.name, kwargs["workspace"]), [])
                filters = kwargs.get("filter") or {}
                for key in ("evaluation_id", "trace_id", "session_id"):
                    if key in filters:
                        items = [item for item in items if item.payload.get(key) == filters[key]]
                return _paginator(items)

        class _Intake:
            spans = _Collection("spans")
            annotations = _Collection("annotations")
            evaluator_results = _Collection("evaluator_results")
            traces = _Collection("traces")

        class _Experiments:
            async def retrieve(self, name, **kwargs):
                outer.calls.append(("experiments", {"name": name, **kwargs}))
                return outer.docs[("experiments", kwargs["workspace"])][0]

        self.intake = _Intake()
        self.experiments = _Experiments()
        self.evaluations = _Collection("evaluations")

    async def close(self):
        self.closed = True

    def calls_for(self, collection: str) -> list[dict]:
        return [kwargs for name, kwargs in self.calls if name == collection]


def _install_fake_client(monkeypatch, docs) -> FakeClient:
    client = FakeClient(docs)
    monkeypatch.setattr(export, "make_client", lambda base_url: client)
    return client


# --------------------------------------------------------------------------- #
# export_workspaces
# --------------------------------------------------------------------------- #


def test_export_writes_jsonl_per_workspace_round_trip(tmp_path, monkeypatch):
    docs = {
        ("spans", "ws-a"): [
            Doc(span_id="s1", started_at="2026-07-01T10:00:00Z", raw_attributes={"k": "v"}, model=None),
            Doc(span_id="s2", started_at="2026-07-01T11:00:00Z"),
        ],
        ("annotations", "ws-a"): [Doc(id="a1", kind="feedback", value_text="positive")],
        ("evaluator_results", "ws-a"): [Doc(id="e1", name="reward", value=1.0)],
    }
    _install_fake_client(monkeypatch, docs)
    export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=None)

    spans = [json.loads(line) for line in (tmp_path / "export" / "ws-a" / "spans.jsonl").read_text().splitlines()]
    assert spans == [
        {"span_id": "s1", "started_at": "2026-07-01T10:00:00Z", "raw_attributes": {"k": "v"}},  # None dropped
        {"span_id": "s2", "started_at": "2026-07-01T11:00:00Z"},
    ]
    anns = [json.loads(line) for line in (tmp_path / "export" / "ws-a" / "annotations.jsonl").read_text().splitlines()]
    assert anns == [{"id": "a1", "kind": "feedback", "value_text": "positive"}]
    results_path = tmp_path / "export" / "ws-a" / "evaluator_results.jsonl"
    assert [json.loads(line) for line in results_path.read_text().splitlines()] == [
        {"id": "e1", "name": "reward", "value": 1.0}
    ]


def test_export_epoch_lower_bound_when_since_omitted(tmp_path, monkeypatch):
    """The read API injects a 30-day default lookback: every query MUST carry an explicit bound."""
    client = _install_fake_client(monkeypatch, {})
    export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=None)

    epoch = "1970-01-01T00:00:00+00:00"
    (span_call,) = client.calls_for("spans")
    assert span_call["filter"] == {"started_at": {"gte": epoch}}
    (ann_call,) = client.calls_for("annotations")
    assert ann_call["filter"] == {"created_at": {"gte": epoch}}
    (res_call,) = client.calls_for("evaluator_results")
    assert res_call["filter"] == {"created_at": {"gte": epoch}}


def test_export_since_becomes_lower_bound(tmp_path, monkeypatch):
    client = _install_fake_client(monkeypatch, {})
    since = datetime(2026, 7, 1, tzinfo=timezone.utc)
    export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=since)

    (span_call,) = client.calls_for("spans")
    assert span_call["filter"] == {"started_at": {"gte": "2026-07-01T00:00:00+00:00"}}
    (ann_call,) = client.calls_for("annotations")
    assert ann_call["filter"] == {"created_at": {"gte": "2026-07-01T00:00:00+00:00"}}


def test_export_detailed_mode_and_generous_pages(tmp_path, monkeypatch):
    client = _install_fake_client(monkeypatch, {})
    export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=None)

    (span_call,) = client.calls_for("spans")
    assert span_call["mode"] == "detailed"
    assert span_call["page_size"] >= 200
    for name in ("annotations", "evaluator_results"):
        (call,) = client.calls_for(name)
        assert call["page_size"] >= 200


def test_export_counts_and_time_bounds_across_workspaces(tmp_path, monkeypatch):
    docs = {
        ("spans", "ws-a"): [
            Doc(span_id="s1", started_at="2026-07-01T10:00:00Z"),
            Doc(span_id="s2", started_at="2026-07-03T10:00:00Z"),
        ],
        ("spans", "ws-b"): [Doc(span_id="s3", started_at="2026-06-30T09:00:00Z")],
        ("annotations", "ws-a"): [Doc(id="a1"), Doc(id="a2")],
        ("evaluator_results", "ws-b"): [Doc(id="e1")],
    }
    _install_fake_client(monkeypatch, docs)
    stats = export.export_workspaces("http://localhost:8080", ["ws-a", "ws-b"], tmp_path, since=None)

    assert stats["workspaces"] == {
        "ws-a": {"spans": 2, "annotations": 2, "evaluator_results": 0},
        "ws-b": {"spans": 1, "annotations": 0, "evaluator_results": 1},
    }
    assert stats["min_start_time"] == datetime(2026, 6, 30, 9, tzinfo=timezone.utc).isoformat()
    assert stats["max_start_time"] == datetime(2026, 7, 3, 10, tzinfo=timezone.utc).isoformat()


def test_export_empty_workspace_time_bounds_none(tmp_path, monkeypatch):
    _install_fake_client(monkeypatch, {})
    stats = export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=None)
    assert stats["min_start_time"] is None
    assert stats["max_start_time"] is None
    assert (tmp_path / "export" / "ws-a" / "spans.jsonl").read_text() == ""


def test_export_closes_client(tmp_path, monkeypatch):
    client = _install_fake_client(monkeypatch, {})
    export.export_workspaces("http://localhost:8080", ["ws-a"], tmp_path, since=None)
    assert client.closed


def test_export_closes_injected_client(tmp_path):
    client = FakeClient({})

    export.export_workspaces(
        "http://localhost:8080",
        ["ws-a"],
        tmp_path,
        since=None,
        client=cast(AsyncNeMoHelix, client),
    )

    assert client.closed


def test_export_experiment_scope_includes_complete_traces(tmp_path, monkeypatch):
    docs = {
        ("experiments", "ws-a"): [Doc(id="experiment-id", evaluation_count=2)],
        ("evaluations", "ws-a"): [Doc(name="eval-b"), Doc(name="eval-a")],
        ("traces", "ws-a"): [
            Doc(id="trace-a", session_id="session-a", evaluation_id="eval-a", span_count=2),
            Doc(id="trace-b", session_id="session-b", evaluation_id="eval-b", span_count=1),
        ],
        ("spans", "ws-a"): [
            Doc(span_id="root-a", trace_id="trace-a", started_at="2026-07-01T10:00:00Z"),
            Doc(span_id="child-a", trace_id="trace-a", started_at="2026-07-01T10:00:01Z"),
            Doc(span_id="root-b", trace_id="trace-b", started_at="2026-07-02T10:00:00Z"),
            Doc(span_id="unrelated", trace_id="other", started_at="2026-07-03T10:00:00Z"),
        ],
        ("annotations", "ws-a"): [
            Doc(annotation_id="a", session_id="session-a"),
            Doc(annotation_id="other", session_id="other"),
        ],
        ("evaluator_results", "ws-a"): [Doc(evaluator_result_id="e", session_id="session-b")],
    }
    client = _install_fake_client(monkeypatch, docs)

    stats = export.export_workspaces(
        "http://localhost:8080",
        ["ws-a"],
        tmp_path,
        since=None,
        experiment="prod-latest-completed",
    )

    assert stats["workspaces"]["ws-a"] == {"spans": 3, "annotations": 1, "evaluator_results": 1}
    selection = stats["selections"]["ws-a"]
    assert selection["evaluation_names"] == ["eval-a", "eval-b"]
    assert selection["trace_ids"] == ["trace-a", "trace-b"]
    assert selection["expected_spans"] == 3
    assert all("trace_id" in call["filter"] for call in client.calls_for("spans"))
    assert all("evaluation_id" not in call["filter"] for call in client.calls_for("spans"))


def test_export_experiment_scope_rejects_source_drift(tmp_path, monkeypatch):
    docs = {
        ("experiments", "ws-a"): [Doc(id="experiment-id", evaluation_count=1)],
        ("evaluations", "ws-a"): [Doc(name="eval-a")],
        ("traces", "ws-a"): [
            Doc(id="trace-a", session_id="session-a", evaluation_id="eval-a", span_count=2),
        ],
        ("spans", "ws-a"): [Doc(span_id="root-a", trace_id="trace-a", started_at="2026-07-01T10:00:00Z")],
    }
    _install_fake_client(monkeypatch, docs)

    with pytest.raises(RuntimeError, match="source changed during capture"):
        export.export_workspaces(
            "http://localhost:8080",
            ["ws-a"],
            tmp_path,
            since=None,
            experiment="prod-latest-completed",
        )


# --------------------------------------------------------------------------- #
# subject scoping
# --------------------------------------------------------------------------- #


def _subject(name, type_, **config):
    return Subject(name, type_, config)


def test_scoping_benchmark_gets_oracle_twin_by_default():
    subject = _subject("tau2-airline", "benchmark", workspace="tau2-airline")
    assert artifact.workspaces_for_subject(subject) == ["tau2-airline", "tau2-airline-oracle"]


def test_scoping_benchmark_include_rewards_false_is_realistic_only():
    subject = _subject("tau2-airline", "benchmark", workspace="tau2-airline", include_rewards=False)
    assert artifact.workspaces_for_subject(subject) == ["tau2-airline"]


def test_scoping_intake_is_single_workspace():
    subject = _subject("nvq", "intake", workspace="nvq")
    assert artifact.workspaces_for_subject(subject) == ["nvq"]


def test_scoping_unknown_type_rejected_with_clear_message():
    subject = _subject("unsupported-subject", "unsupported", workspace="tau2-airline")
    with pytest.raises(SystemExit) as exc:
        artifact.workspaces_for_subject(subject)
    message = str(exc.value)
    assert "unsupported-subject" in message
    assert "unsupported" in message


def test_scoping_missing_workspace_rejected():
    subject = _subject("broken", "benchmark")
    with pytest.raises(SystemExit) as exc:
        artifact.workspaces_for_subject(subject)
    assert "broken" in str(exc.value)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

_STATS = {
    "workspaces": {"tau2-airline": {"spans": 2, "annotations": 1, "evaluator_results": 0}},
    "min_start_time": "2026-07-01T00:00:00+00:00",
    "max_start_time": "2026-07-02T00:00:00+00:00",
}

CI_LINEAGE_KEYS = (
    "nemo_helix_sha",
    "tau2_bench_sha",
    "github_run_id",
    "num_tasks",
    "num_trials",
    "judge_llm",
    "reason",
)


def test_build_export_manifest_fields(tmp_path):
    rec = tmp_path / "tau2-airline.run.json"
    rec.write_text("{}")
    manifest = artifact.build_export_manifest(
        ["tau2-airline"],
        [rec],
        _STATS,
        source_url="http://localhost:8080",
        platform_info={"platform_version": "26.07", "revision": "deadbeef"},
        env={},
    )
    assert manifest["kind"] == "evaluation-export"
    assert manifest["source_url"] == "http://localhost:8080"
    assert manifest["subjects"] == ["tau2-airline"]
    assert manifest["workspaces"] == ["tau2-airline"]
    assert manifest["counts"] == _STATS["workspaces"]
    assert manifest["min_start_time"] == "2026-07-01T00:00:00+00:00"
    assert manifest["max_start_time"] == "2026-07-02T00:00:00+00:00"
    assert manifest["platform_version"] == "26.07"
    assert manifest["platform_revision"] == "deadbeef"
    assert manifest["records"] == ["tau2-airline.run.json"]
    assert not any(k in manifest for k in CI_LINEAGE_KEYS)


def test_build_export_manifest_platform_info_none(tmp_path):
    manifest = artifact.build_export_manifest(
        ["nvq"],
        [],
        _STATS,
        source_url="u",
        platform_info=None,
        env={},
    )
    assert manifest["platform_version"] is None
    assert manifest["platform_revision"] is None


def test_build_export_manifest_carries_ci_lineage(tmp_path):
    env = {
        "GITHUB_SHA": "abc",
        "TAU2_BENCH_REF": "t2sha",
        "GITHUB_RUN_ID": "42",
        "NUM_TASKS": "2",
        "NUM_TRIALS": "3",
        "TAU2_JUDGE_LLM": "judge",
        "REASON": "tau2 bump",
    }
    manifest = artifact.build_export_manifest(
        ["tau2-airline"],
        [],
        _STATS,
        source_url="u",
        platform_info=None,
        env=env,
    )
    assert manifest["nemo_helix_sha"] == "abc"
    assert manifest["tau2_bench_sha"] == "t2sha"
    assert manifest["github_run_id"] == "42"
    assert manifest["num_tasks"] == "2"
    assert manifest["num_trials"] == "3"
    assert manifest["judge_llm"] == "judge"
    assert manifest["reason"] == "tau2 bump"
    assert manifest["kind"] == "evaluation-export"  # base fields still present alongside lineage


# --------------------------------------------------------------------------- #
# fetch_platform_info (best-effort /cluster-info probe)
# --------------------------------------------------------------------------- #


def test_fetch_platform_info_reads_cluster_info(monkeypatch):
    seen = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"platform_version": "26.07", "revision": "deadbeef"}

    def fake_get(url, timeout):
        seen["url"] = url
        return FakeResp()

    monkeypatch.setattr(artifact.httpx, "get", fake_get)
    info = artifact.fetch_platform_info("http://localhost:8080/")
    assert seen["url"] == "http://localhost:8080/cluster-info"
    assert info == {"platform_version": "26.07", "revision": "deadbeef"}


def test_fetch_platform_info_none_when_unreachable(monkeypatch):
    def boom(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(artifact.httpx, "get", boom)
    assert artifact.fetch_platform_info("http://localhost:8080") is None


# --------------------------------------------------------------------------- #
# snapshot_export
# --------------------------------------------------------------------------- #


def _fake_export(seen):
    def fake(base_url, workspaces, out_dir, *, since, experiment=None, client=None):
        seen["base_url"] = base_url
        seen["workspaces"] = list(workspaces)
        seen["since"] = since
        seen["experiment"] = experiment
        seen["client"] = client
        for ws in workspaces:
            ws_dir = out_dir / "export" / ws
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "spans.jsonl").write_text('{"span_id": "s1"}\n', encoding="utf-8")
            (ws_dir / "annotations.jsonl").write_text("", encoding="utf-8")
            (ws_dir / "evaluator_results.jsonl").write_text("", encoding="utf-8")
        result = {
            "workspaces": {ws: {"spans": 1, "annotations": 0, "evaluator_results": 0} for ws in workspaces},
            "min_start_time": "2026-07-01T00:00:00+00:00",
            "max_start_time": "2026-07-02T00:00:00+00:00",
        }
        if experiment:
            result["selections"] = {workspaces[0]: {"kind": "experiment", "experiment": experiment}}
        return result

    return fake


def test_snapshot_export_bundle_layout_and_manifest(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export(seen))
    monkeypatch.setattr(
        artifact,
        "fetch_platform_info",
        lambda url: {"platform_version": "26.07", "revision": "deadbeef"},
    )
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    (tmp_dir / "tau2-airline.run.json").write_text('{"agent": "a"}', encoding="utf-8")
    (tmp_dir / "insights_tau2-airline.yaml").write_text("insights: []", encoding="utf-8")
    subject = _subject("tau2-airline", "benchmark", workspace="tau2-airline", base_url="http://remote:8080")
    out = tmp_path / "bundle.tar.zst"

    result = artifact.snapshot_export([subject], out, tmp_dir, since=None)

    assert result == out and out.exists()
    assert seen["base_url"] == "http://remote:8080"
    assert seen["workspaces"] == ["tau2-airline", "tau2-airline-oracle"]
    extract = tmp_path / "extract"
    extract.mkdir()
    subprocess.run(["tar", "--zstd", "-xf", str(out), "-C", str(extract)], check=True)
    root = extract / "state"
    assert (root / "export" / "tau2-airline" / "spans.jsonl").read_text() == '{"span_id": "s1"}\n'
    assert (root / "export" / "tau2-airline-oracle" / "annotations.jsonl").exists()
    assert (root / "tmp" / "tau2-airline.run.json").exists()
    # insights never travel in bundles: the subject's local YAML stays out
    assert not (root / "tmp" / "insights_tau2-airline.yaml").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["kind"] == "evaluation-export"
    assert manifest["source_url"] == "http://remote:8080"
    assert manifest["subjects"] == ["tau2-airline"]
    assert manifest["workspaces"] == ["tau2-airline", "tau2-airline-oracle"]
    assert manifest["counts"]["tau2-airline"] == {"spans": 1, "annotations": 0, "evaluator_results": 0}
    assert manifest["min_start_time"] == "2026-07-01T00:00:00+00:00"
    assert manifest["platform_version"] == "26.07"
    assert manifest["platform_revision"] == "deadbeef"
    assert manifest["records"] == ["tau2-airline.run.json"]


def test_snapshot_export_scopes_records_to_selected_subjects(tmp_path, monkeypatch):
    """Other subjects' run records and insight YAMLs in tmp/ must NOT leak into the bundle."""
    seen: dict = {}
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export(seen))
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda url: None)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    (tmp_dir / "tau2-airline.run.json").write_text('{"agent": "a"}', encoding="utf-8")
    (tmp_dir / "tau2-retail.run.json").write_text('{"agent": "r"}', encoding="utf-8")
    (tmp_dir / "insights_tau2-retail.yaml").write_text("insights: []", encoding="utf-8")
    subject = _subject("tau2-airline", "benchmark", workspace="tau2-airline", base_url="http://remote:8080")
    out = tmp_path / "bundle.tar.zst"

    artifact.snapshot_export([subject], out, tmp_dir, since=None)

    extract = tmp_path / "extract"
    extract.mkdir()
    subprocess.run(["tar", "--zstd", "-xf", str(out), "-C", str(extract)], check=True)
    names = sorted(p.name for p in (extract / "state" / "tmp").iterdir())
    assert names == ["tau2-airline.run.json"]
    manifest = json.loads((extract / "state" / "manifest.json").read_text())
    assert manifest["records"] == ["tau2-airline.run.json"]


def test_snapshot_export_passes_since_through(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export(seen))
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda url: None)
    subject = _subject("nvq", "intake", workspace="nvq", base_url="u")
    (tmp_path / "tmp").mkdir()
    since = datetime(2026, 7, 1, tzinfo=timezone.utc)
    artifact.snapshot_export([subject], tmp_path / "b.tar.zst", tmp_path / "tmp", since=since)
    assert seen["since"] == since


def test_snapshot_export_passes_experiment_scope(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export(seen))
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda url: None)
    subject = _subject(
        "kernel-factory",
        "intake",
        workspace="kf-prod-evals",
        base_url="u",
        experiment="prod-latest-completed",
    )
    (tmp_path / "tmp").mkdir()

    artifact.snapshot_export([subject], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)

    assert seen["experiment"] == "prod-latest-completed"
    extract = tmp_path / "extract"
    extract.mkdir()
    subprocess.run(["tar", "--zstd", "-xf", str(tmp_path / "b.tar.zst"), "-C", str(extract)], check=True)
    manifest = json.loads((extract / "state" / "manifest.json").read_text())
    assert manifest["selections"]["kf-prod-evals"]["experiment"] == "prod-latest-completed"


def test_snapshot_export_dedupes_workspaces_across_subjects(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export(seen))
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda url: None)
    a = _subject("a", "intake", workspace="shared", base_url="u")
    b = _subject("b", "intake", workspace="shared", base_url="u")
    (tmp_path / "tmp").mkdir()
    artifact.snapshot_export([a, b], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)
    assert seen["workspaces"] == ["shared"]


def test_snapshot_export_rejects_shared_experiment_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export({}))
    a = _subject("a", "intake", workspace="shared", base_url="u", experiment="run-a")
    b = _subject("b", "intake", workspace="shared", base_url="u")
    (tmp_path / "tmp").mkdir()
    with pytest.raises(SystemExit, match="shared"):
        artifact.snapshot_export([a, b], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)


def test_snapshot_export_conflicting_base_urls_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export({}))
    a = _subject("a", "intake", workspace="wa", base_url="http://one:8080")
    b = _subject("b", "intake", workspace="wb", base_url="http://two:8080")
    (tmp_path / "tmp").mkdir()
    with pytest.raises(SystemExit) as exc:
        artifact.snapshot_export([a, b], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)
    assert "base_url" in str(exc.value)


def test_snapshot_export_partial_missing_base_url_exits_naming_subject(tmp_path, monkeypatch):
    """Every selected subject must carry a base_url: a partial miss names the offender
    instead of silently letting the agreement check pass on the configured subset."""
    called: list = []
    monkeypatch.setattr(artifact.export, "export_workspaces", lambda *a, **k: called.append(1))
    a = _subject("a", "intake", workspace="wa", base_url="http://one:8080")
    b = _subject("b", "intake", workspace="wb")  # no base_url at all
    (tmp_path / "tmp").mkdir()
    with pytest.raises(SystemExit) as exc:
        artifact.snapshot_export([a, b], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)
    message = str(exc.value)
    assert "no base_url configured for subject(s) b" in message
    assert "--base" in message
    assert called == []  # exits before any export I/O


def test_snapshot_export_unknown_subject_type_exits_before_export(tmp_path, monkeypatch):
    called: list = []
    monkeypatch.setattr(artifact.export, "export_workspaces", lambda *a, **k: called.append(1))
    subject = _subject("unsupported-subject", "unsupported", workspace="tau2-airline", base_url="u")
    (tmp_path / "tmp").mkdir()
    with pytest.raises(SystemExit):
        artifact.snapshot_export([subject], tmp_path / "b.tar.zst", tmp_path / "tmp", since=None)
    assert called == []


def _extract_manifest(bundle: Path) -> dict:
    return json.loads(
        subprocess.run(
            ["tar", "--zstd", "-xOf", str(bundle), "state/manifest.json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def test_snapshot_export_lineage_env_lands_in_manifest(tmp_path, monkeypatch):
    """The CI lineage merge is wired through snapshot_export (os.environ), not just the builder."""
    monkeypatch.setattr(artifact.export, "export_workspaces", _fake_export({}))
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda url: None)
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("REASON", "export bundles")
    subject = _subject("nvq", "intake", workspace="nvq", base_url="u")
    (tmp_path / "tmp").mkdir()
    out = tmp_path / "b.tar.zst"
    artifact.snapshot_export([subject], out, tmp_path / "tmp", since=None)
    manifest = _extract_manifest(out)
    assert manifest["nemo_helix_sha"] == "abc"
    assert manifest["reason"] == "export bundles"
