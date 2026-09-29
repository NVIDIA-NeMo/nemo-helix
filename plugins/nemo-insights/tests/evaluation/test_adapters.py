# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.adapters import BenchmarkAdapter, IntakeAdapter, build_adapter
from evaluation.registry import Subject
from nemo_insights_plugin.analyst.observability import AnalystEvaluationContext

_CFG = {
    "domain": "airline",
    "base_url": "http://localhost:8080",
    "workspace": "tau2-airline",
    "agent_llm": "openai/m",
    "user_llm": "openai/m",
    "task_split_name": "test",
    "num_trials": 1,
    "seed": 300,
    "max_concurrency": 4,
}

_SIMS = [
    {
        "task_id": "0",
        "trial": 0,
        "messages": [{"role": "user", "content": "hi"}],
        "reward_info": {"reward": 1.0},
        "termination_reason": "done",
    },
    {
        "task_id": "1",
        "trial": 0,
        "messages": [{"role": "user", "content": "yo"}],
        "reward_info": {"reward": 0.0},
        "termination_reason": "done",
    },
]


class _FakeSDK:
    """Stand-in for the generated SDK that records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_FakeSDK":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True


def _intake_subject(**overrides) -> Subject:
    config = {"agent": "a", "workspace": "w", "base_url": "u", **overrides}
    return Subject(name="nvq", type="intake", config=config)


def _basic_intake_subject(**overrides) -> Subject:
    config = {
        "agent": "glamr",
        "workspace": "default",
        "base_url": "https://agenthub.aire.nvidia.com",
        "auth": "basic",
        "auth_user_env": "GLAMR_INTAKE_USER",
        "auth_password_env": "GLAMR_INTAKE_PASSWORD",
        **overrides,
    }
    return Subject(name="glamr", type="intake", config=config)


def test_intake_check_basic_auth_reports_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLAMR_INTAKE_USER", raising=False)
    monkeypatch.delenv("GLAMR_INTAKE_PASSWORD", raising=False)

    missing = IntakeAdapter(_basic_intake_subject()).check()

    assert any("GLAMR_INTAKE_USER" in item for item in missing)
    assert any("GLAMR_INTAKE_PASSWORD" in item for item in missing)


def test_intake_check_basic_auth_reports_missing_username_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "secret")

    missing = IntakeAdapter(_basic_intake_subject(auth_user_env=None)).check()

    assert any("auth_user_env" in item for item in missing)
    assert not any("auth_password_env" in item for item in missing)


def test_intake_check_basic_auth_reports_missing_password_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_USER", "intake")

    missing = IntakeAdapter(_basic_intake_subject(auth_password_env=None)).check()

    assert any("auth_password_env" in item for item in missing)
    assert not any("auth_user_env" in item for item in missing)


async def test_intake_analyze_basic_auth_injects_built_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_USER", "intake")
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "secret")
    sentinel = _FakeSDK()
    built: dict[str, object] = {}
    calls: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> object:
        built.update(kwargs)
        return sentinel

    async def fake_run_analyst(**kwargs: object) -> str:
        calls.update(kwargs)
        return "REPORT"

    monkeypatch.setattr("evaluation.adapters.build_basic_auth_intake_client", fake_builder)
    monkeypatch.setattr("evaluation.adapters.client_from_platform", lambda platform, client_cls: platform)
    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)

    report = await IntakeAdapter(_basic_intake_subject()).analyze(
        record=None,
        since=None,
        verbose=False,
        out_path=tmp_path / "insights.json",
    )

    assert report == "REPORT"
    assert calls["client"] is sentinel
    assert sentinel.closed
    assert calls["enable_observability"] is False
    assert built == {
        "base_url": "https://agenthub.aire.nvidia.com",
        "username": "intake",
        "password": "secret",
        "real_prefix": "/api/intake/",
    }


def test_build_adapter_returns_intake():
    assert isinstance(build_adapter(_intake_subject()), IntakeAdapter)


def test_build_adapter_unknown_type_exits():
    with pytest.raises(SystemExit):
        build_adapter(Subject(name="x", type="bogus", config={}))


async def test_intake_analyze_calls_run_analyst(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    built_client = _FakeSDK()

    async def fake_run_analyst(**kwargs):
        calls.update(kwargs)
        return "REPORT"

    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)
    monkeypatch.setattr("evaluation.adapters.make_client", lambda base_url: built_client)
    monkeypatch.setattr("evaluation.adapters.client_from_platform", lambda platform, client_cls: platform)
    out = tmp_path / "insights.json"
    report = await build_adapter(_intake_subject()).analyze(record=None, since=None, verbose=True, out_path=out)
    assert report == "REPORT"
    assert calls["agent"] == "a"
    assert calls["workspace"] == "w"
    assert calls["base_url"] == "u"
    assert calls["ethos"] is None
    assert calls["client"] is built_client
    assert built_client.closed
    assert calls["enable_observability"] is True


async def test_intake_analyze_closes_sdk_when_analyst_fails(monkeypatch, tmp_path: Path):
    built_client = _FakeSDK()

    async def failing_run_analyst(**kwargs):
        raise RuntimeError("analyst failed")

    monkeypatch.setattr("evaluation.adapters.run_analyst", failing_run_analyst)
    monkeypatch.setattr("evaluation.adapters.make_client", lambda base_url: built_client)
    monkeypatch.setattr("evaluation.adapters.client_from_platform", lambda platform, client_cls: platform)
    with pytest.raises(RuntimeError, match="analyst failed"):
        await build_adapter(_intake_subject()).analyze(
            record=None, since=None, verbose=False, out_path=tmp_path / "insights.json"
        )
    assert built_client.closed


async def test_intake_analyze_missing_keys_exits(tmp_path: Path):
    bad = Subject(name="nvq", type="intake", config={"agent": "a"})
    with pytest.raises(SystemExit):
        await IntakeAdapter(bad).analyze(record=None, since=None, verbose=False, out_path=tmp_path / "x.json")


async def test_intake_produce_message_says_analyze():
    """The `insights` alias is gone; the no-produce-step pointer must name `analyze --live`
    (bare analyze restores a pinned state — not what a produce-less intake subject wants)."""
    with pytest.raises(SystemExit) as exc:
        await IntakeAdapter(_intake_subject()).produce()
    message = str(exc.value)
    assert "evaluation analyze nvq --live" in message
    assert "evaluation insights" not in message


def test_build_adapter_dispatches_benchmark():
    adapter = build_adapter(Subject("tau2-airline", "benchmark", _CFG))
    assert isinstance(adapter, BenchmarkAdapter)


async def test_benchmark_preflight_lists_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    # stanza with no tau2_repo and no model -> tau2_repo + models + creds all flagged
    adapter = BenchmarkAdapter(Subject("tau2-airline", "benchmark", {"domain": "airline"}))
    with pytest.raises(SystemExit) as exc:
        await adapter.produce()
    msg = str(exc.value)
    assert "tau2_repo" in msg and "OPENAI_API_KEY" in msg


async def test_benchmark_preflight_rejects_placeholder_model(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr("evaluation.adapters.shutil.which", lambda _name: "/usr/bin/tau2")
    # valid tau2 paths (via overrides) so ONLY the placeholder model is flagged
    cfg = {
        **_CFG,
        "tau2_data_dir": str(tmp_path),
        "tau2_bin": "tau2",
        "agent_llm": "openai/<your-model>",
    }
    adapter = BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg))
    with pytest.raises(SystemExit) as exc:
        await adapter.produce()
    assert "agent_llm" in str(exc.value)


async def test_benchmark_preflight_flags_missing_binary(monkeypatch, tmp_path):
    # tau2 paths resolve (data dir exists) but the binary isn't found -> flagged.
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr("evaluation.adapters.shutil.which", lambda _name: None)
    cfg = {**_CFG, "tau2_data_dir": str(tmp_path), "tau2_bin": "tau2"}
    adapter = BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg))
    with pytest.raises(SystemExit) as exc:
        await adapter.produce()
    assert "tau2 binary" in str(exc.value)


def test_benchmark_check_reports_missing_argv_keys(monkeypatch, tmp_path):
    """check() covers every key build_argv hard-indexes, so `run` can't KeyError past doctor.

    seed=0 is a valid RNG seed and num_trials/max_concurrency are ints too — only an
    ABSENT key is missing, never a falsy-but-present value.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr("evaluation.adapters.shutil.which", lambda _name: "/usr/bin/tau2")
    base = {
        "domain": "airline",
        "base_url": "http://localhost:8080",
        "workspace": "w",
        "agent_llm": "openai/m",
        "user_llm": "openai/m",
        "tau2_data_dir": str(tmp_path),
        "tau2_bin": "tau2",
    }
    argv_keys = ("task_split_name", "num_trials", "seed", "max_concurrency")
    missing = BenchmarkAdapter(Subject("t", "benchmark", base)).check()
    assert {f"config key '{k}'" for k in argv_keys} <= set(missing)
    ready = {**base, "task_split_name": "train", "num_trials": 1, "seed": 0, "max_concurrency": 1}
    assert BenchmarkAdapter(Subject("t", "benchmark", ready)).check() == []


async def test_benchmark_preflight_flags_repo_when_only_bin_set(monkeypatch, tmp_path):
    # tau2_bin set but no tau2_repo/tau2_data_dir -> data dir unresolved -> tau2_repo flagged.
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr("evaluation.adapters.shutil.which", lambda _name: "/usr/bin/tau2")
    cfg = {**_CFG, "tau2_bin": "tau2"}  # no tau2_repo, no tau2_data_dir
    adapter = BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg))
    with pytest.raises(SystemExit) as exc:
        await adapter.produce()
    assert "tau2_repo" in str(exc.value)


def _kinds(spans: list[dict[str, Any]]) -> set[str]:
    return {s["kind"] for s in spans}


def _patch_otlp(
    monkeypatch,
    *,
    exported: list[tuple[str, str, list[dict[str, Any]]]],
    evals: list[tuple[str, str, str, float]],
    created: list[str] | None = None,
    experiments: list[tuple[str, str, str, str]] | None = None,
    poll: bool = True,
) -> None:
    """Patch the OTLP ingest seam: capture export_spans / post_evaluator_results."""
    monkeypatch.setattr("evaluation.adapters.shutil.which", lambda _name: "/usr/bin/tau2")
    monkeypatch.setattr("evaluation.adapters.read_policy", lambda data_dir, domain: "POLICY")
    monkeypatch.setattr(
        "evaluation.adapters.ensure_workspace",
        lambda base_url, workspace, *, client=None: created.append(workspace) if created is not None else None,
    )
    monkeypatch.setattr(
        "evaluation.adapters.ensure_experiment_group",
        lambda base_url, workspace, name, *, client=None: "grp-1",
    )
    monkeypatch.setattr(
        "evaluation.adapters.create_experiment",
        lambda base_url, workspace, *, name, experiment_group_id, dataset_name, dataset_version, metadata, client=None: (
            experiments.append((workspace, name, dataset_name, dataset_version)) if experiments is not None else None
        ),
    )
    monkeypatch.setattr(
        "evaluation.adapters.export_spans",
        lambda base_url, workspace, session_id, trace_id, spans, *, client=None: exported.append(
            (workspace, session_id, spans)
        ),
    )
    monkeypatch.setattr(
        "evaluation.adapters.post_evaluator_results",
        lambda base_url, workspace, *, span_id, session_id, score, client=None: evals.append(
            (workspace, span_id, session_id, score)
        ),
    )
    if poll:
        monkeypatch.setattr(
            "evaluation.adapters.poll_visible",
            lambda base_url, workspace, session_ids, *, client=None, **kw: set(session_ids),
        )


async def test_benchmark_produce_records_run_without_analyzing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr(
        "evaluation.adapters.run_tau2",
        lambda cfg, run_id, *, data_dir, tau2_bin: _SIMS,
    )
    created: list[str] = []
    exported: list[tuple[str, str, list[dict[str, Any]]]] = []
    evals: list[tuple[str, str, str, float]] = []
    experiments: list[tuple[str, str, str, str]] = []
    _patch_otlp(monkeypatch, exported=exported, evals=evals, created=created, experiments=experiments)
    analyst_called = {"v": False}

    async def fake_run_analyst(**kwargs):
        analyst_called["v"] = True
        return "SHOULD-NOT-RUN"

    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)
    monkeypatch.setattr("evaluation.adapters.make_client", lambda base_url: object())
    monkeypatch.setattr("evaluation.adapters.client_from_platform", lambda platform, client_cls: platform)

    cfg = {**_CFG, "tau2_data_dir": str(tmp_path), "tau2_bin": "tau2"}
    record = await BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg)).produce()
    experiment_id = record["experiment_id"]
    assert isinstance(experiment_id, str)

    assert analyst_called["v"] is False  # produce does NOT analyze
    assert record["realistic_workspace"] == "tau2-airline"  # stable, not per-run
    assert record["oracle_workspace"] == "tau2-airline-oracle"
    assert experiment_id.startswith("tau2-airline-")  # run id = experiment name
    assert record["experiment_id"] != record["realistic_workspace"]
    assert record["experiment_group"] == "tau2-airline"
    assert record["dataset_name"] == "tau2:airline"
    assert created == ["tau2-airline", "tau2-airline-oracle"]  # both stable workspaces ensured
    # Experiment entity created on the ORACLE workspace only.
    assert experiments == [
        ("tau2-airline", record["experiment_id"], "tau2:airline", record["dataset_version"]),
        ("tau2-airline-oracle", record["experiment_id"], "tau2:airline", record["dataset_version"]),
    ]
    assert len(exported) == 4  # 2 sims x 2 workspaces
    # Every exported span is tagged with the run id.
    for _ws, _sid, spans in exported:
        assert all(s["attributes"]["nemo.evaluation.name"] == record["experiment_id"] for s in spans)
    assert all("EVALUATOR" not in _kinds(spans) for ws, _, spans in exported if ws == "tau2-airline")
    assert all("EVALUATOR" in _kinds(spans) for ws, _, spans in exported if ws == "tau2-airline-oracle")
    assert len(evals) == 2
    assert {ws for ws, *_ in evals} == {"tau2-airline-oracle"}
    assert record["agent"] == "tau2-airline"
    assert record["base_url"] == "http://localhost:8080"
    assert record["domain"] == "airline"


async def test_benchmark_analyze_uses_record(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.adapters.read_policy", lambda data_dir, domain: "POLICY")
    ran_tau2 = {"v": False}
    monkeypatch.setattr(
        "evaluation.adapters.run_tau2",
        lambda *a, **k: ran_tau2.__setitem__("v", True) or [],
    )
    seen: dict[str, object] = {}

    async def fake_run_analyst(
        *,
        agent,
        ethos,
        workspace,
        base_url,
        client,
        insights_output,
        local_only,
        verbose,
        since,
        evaluation_id,
        analyst_evaluation,
    ):
        seen.update(
            agent=agent,
            ethos=ethos,
            workspace=workspace,
            base_url=base_url,
            evaluation_id=evaluation_id,
            local_only=local_only,
            analyst_evaluation=analyst_evaluation,
        )
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)
    built_client = _FakeSDK()
    monkeypatch.setattr("evaluation.adapters.make_client", lambda base_url: built_client)
    monkeypatch.setattr("evaluation.adapters.client_from_platform", lambda platform, client_cls: platform)

    cfg = {**_CFG, "tau2_data_dir": str(tmp_path), "tau2_bin": "tau2"}
    record: dict[str, object] = {
        "agent": "tau2-airline",
        "realistic_workspace": "tau2-airline",
        "oracle_workspace": "tau2-airline-oracle",
        "experiment_id": "tau2-airline-20260626-000000-abcd",
        "base_url": "http://localhost:8080",
        "domain": "airline",
    }
    analyst_evaluation = AnalystEvaluationContext(
        evaluation_name="nemo-analyst-run",
        test_case_name="tau2-airline",
    )
    adapter = BenchmarkAdapter(
        Subject("tau2-airline", "benchmark", cfg),
        analyst_evaluation=analyst_evaluation,
    )
    out = await adapter.analyze(
        record=record,
        since=None,
        verbose=True,
        out_path=tmp_path / "o.json",
    )
    assert out == "REPORT-OK"
    assert ran_tau2["v"] is False  # analyze never runs tau2
    assert seen["agent"] == "tau2-airline"
    assert seen["ethos"] == "POLICY"
    assert seen["workspace"] == "tau2-airline"  # the stable REALISTIC workspace, never the oracle one
    assert seen["evaluation_id"] == "tau2-airline-20260626-000000-abcd"  # run-scoped
    assert seen["base_url"] == "http://localhost:8080"
    assert seen["local_only"] is True  # benchmark runs capture to file and never write to the platform
    assert seen["analyst_evaluation"] == analyst_evaluation
    assert built_client.closed


async def test_benchmark_analyze_without_record_raises(tmp_path):
    adapter = BenchmarkAdapter(Subject("tau2-airline", "benchmark", _CFG))
    with pytest.raises(SystemExit):
        await adapter.analyze(record=None, since=None, verbose=False, out_path=tmp_path / "x.json")


async def test_benchmark_produce_splits_oracle_between_workspaces(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr(
        "evaluation.adapters.run_tau2",
        lambda cfg, run_id, *, data_dir, tau2_bin: [
            {
                "task_id": "8",
                "trial": 0,
                "messages": [{"role": "user", "content": "hi"}],
                "reward_info": {"reward": 1.0, "reward_breakdown": {"DB": 1.0}},
                "termination_reason": "done",
            }
        ],
    )
    monkeypatch.setattr(
        "evaluation.adapters.load_tasks",
        lambda data_dir, domain: {
            "8": {
                "id": "8",
                "user_scenario": {"instructions": "do X"},
                "evaluation_criteria": {"actions": ["book"]},
            }
        },
    )
    exported: list[tuple[str, str, list[dict[str, Any]]]] = []
    evals: list[tuple[str, str, str, float]] = []
    _patch_otlp(monkeypatch, exported=exported, evals=evals)

    async def fake_run_analyst(**kwargs):
        return "R"

    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)
    cfg = {**_CFG, "tau2_data_dir": str(tmp_path), "tau2_bin": "tau2"}
    record = await BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg)).produce()

    by_ws = {ws: spans for ws, _sid, spans in exported}
    realistic_workspace = record["realistic_workspace"]
    oracle_workspace = record["oracle_workspace"]
    assert isinstance(realistic_workspace, str)
    assert isinstance(oracle_workspace, str)
    realistic = by_ws[realistic_workspace]
    oracle = by_ws[oracle_workspace]

    def _root_task(spans):
        (root,) = [s for s in spans if s["kind"] == "AGENT"]
        return json.loads(root["attributes"]["tau2.task"])

    # Realistic: no EVALUATOR span, task SETUP retained, gold criteria withheld.
    assert "EVALUATOR" not in {s["kind"] for s in realistic}
    assert _root_task(realistic)["user_scenario"]["instructions"] == "do X"
    assert "evaluation_criteria" not in _root_task(realistic)

    # Oracle: EVALUATOR span + full answer key + a reward row targeting that span.
    (evaluator,) = [s for s in oracle if s["kind"] == "EVALUATOR"]
    assert _root_task(oracle)["evaluation_criteria"] == {"actions": ["book"]}
    assert evaluator["attributes"]["score"] == 1.0
    (eval_ws, eval_span_id, _sid, eval_score) = evals[0]
    assert eval_ws == record["oracle_workspace"]
    assert eval_span_id == evaluator["span_id"]  # reward row targets the EVALUATOR span
    assert eval_score == 1.0


async def test_benchmark_produce_realistic_only_when_rewards_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://gw")
    monkeypatch.setattr(
        "evaluation.adapters.run_tau2",
        lambda cfg, run_id, *, data_dir, tau2_bin: _SIMS,
    )
    monkeypatch.setattr("evaluation.adapters.load_tasks", lambda data_dir, domain: {})
    created: list[str] = []
    exported: list[tuple[str, str, list[dict[str, Any]]]] = []
    evals: list[tuple[str, str, str, float]] = []
    experiments: list[tuple[str, str, str, str]] = []
    _patch_otlp(monkeypatch, exported=exported, evals=evals, created=created, experiments=experiments)

    async def fake_run_analyst(**kwargs):
        return "R"

    monkeypatch.setattr("evaluation.adapters.run_analyst", fake_run_analyst)
    cfg = {**_CFG, "tau2_data_dir": str(tmp_path), "tau2_bin": "tau2", "include_rewards": False}
    record = await BenchmarkAdapter(Subject("tau2-airline", "benchmark", cfg)).produce()

    assert record["oracle_workspace"] is None  # no oracle workspace
    assert created == [record["realistic_workspace"]]  # only the realistic one created
    assert len(exported) == 2  # 2 sims x 1 workspace
    assert all(ws == record["realistic_workspace"] for ws, _, _ in exported)
    assert evals == []  # no oracle → no reward rows
    assert experiments == [("tau2-airline", record["experiment_id"], "tau2:airline", record["dataset_version"])]
