# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from evaluation import cli
from nemo_insights_plugin.analyst.observability import AnalystEvaluationContext

_REAL_LOAD_REGISTRY = cli.load_registry
_REAL_REGISTER_ANALYSIS_EVALUATION = cli._register_analysis_evaluation


@pytest.fixture(autouse=True)
def isolate_checked_in_insights(monkeypatch, tmp_path):
    from evaluation.registry import Subject

    def load_registry_with_test_intake(path):
        subjects = dict(_REAL_LOAD_REGISTRY(path))
        subjects["nvq"] = Subject(
            "nvq",
            "intake",
            {
                "agent": "content-dedup",
                "workspace": "nvq",
                "base_url": "https://nemo-helix-freeplay.dev.aire.nvidia.com",
                "state": "state-v7",
            },
        )
        return subjects

    helpers = {
        "check_in": getattr(cli, "_check_in_insights", None),
        "write_manifest": getattr(cli, "_write_insights_manifest", None),
    }
    monkeypatch.setattr(cli, "load_registry", load_registry_with_test_intake)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", tmp_path / "checked-in-insights", raising=False)
    monkeypatch.setattr(
        cli,
        "_register_analysis_evaluation",
        lambda **_kwargs: AnalystEvaluationContext(
            evaluation_name="nemo-analyst-test",
            test_case_name="test-subject",
        ),
    )
    if helpers["check_in"] is not None:
        monkeypatch.setattr(
            cli,
            "_check_in_insights",
            lambda subject_name, _source, *, directory=None: (directory or cli.INSIGHTS_DIR) / f"{subject_name}.yaml",
        )
    if helpers["write_manifest"] is not None:
        monkeypatch.setattr(
            cli,
            "_write_insights_manifest",
            lambda *_args, **_kwargs: cli.INSIGHTS_DIR / "manifest.yaml",
        )
    return helpers


# --------------------------------------------------------------------------- #
# bundle fixtures: real tar.zst files, kind-switched exactly like production
# --------------------------------------------------------------------------- #


def _make_export_bundle(
    path: Path,
    *,
    workspaces=("tau2-airline", "tau2-airline-oracle"),
    min_start_time="2026-06-01T12:00:00+00:00",
    records: dict[str, str] | None = None,
) -> Path:
    """A minimal but structurally faithful `kind: evaluation-export` bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        state = Path(tmp) / "state"
        (state / "tmp").mkdir(parents=True)
        counts = {}
        for ws in workspaces:
            ws_dir = state / "export" / ws
            ws_dir.mkdir(parents=True)
            (ws_dir / "spans.jsonl").write_text('{"span_id": "s1"}\n', encoding="utf-8")
            (ws_dir / "annotations.jsonl").write_text("", encoding="utf-8")
            (ws_dir / "evaluator_results.jsonl").write_text("", encoding="utf-8")
            counts[ws] = {"spans": 1, "annotations": 0, "evaluator_results": 0}
        for name, text in (records or {}).items():
            (state / "tmp" / name).write_text(text, encoding="utf-8")
        manifest = {
            "kind": "evaluation-export",
            "subjects": sorted({ws.removesuffix("-oracle") for ws in workspaces}),
            "workspaces": sorted(workspaces),
            "counts": counts,
            "min_start_time": min_start_time,
            "max_start_time": min_start_time,
            "source_url": "http://origin:8080",
        }
        (state / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["tar", "--zstd", "-cf", str(path), "-C", tmp, "state"], check=True)
    return path


def _make_legacy_bundle(path: Path) -> Path:
    """A legacy tar bundle: state/manifest.json without the evaluation-export kind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        state = Path(tmp) / "state"
        (state / "clickhouse").mkdir(parents=True)
        (state / "manifest.json").write_text('{"created_at": "2026-01-01"}', encoding="utf-8")
        subprocess.run(["tar", "--zstd", "-cf", str(path), "-C", tmp, "state"], check=True)
    return path


def _make_tar_with_manifest_text(path: Path, manifest_text: str | None) -> Path:
    """A readable tar.zst whose state/manifest.json is *manifest_text* (or absent when None)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        state = Path(tmp) / "state"
        state.mkdir(parents=True)
        (state / "other.txt").write_text("x", encoding="utf-8")
        if manifest_text is not None:
            (state / "manifest.json").write_text(manifest_text, encoding="utf-8")
        subprocess.run(["tar", "--zstd", "-cf", str(path), "-C", tmp, "state"], check=True)
    return path


_RUN_RECORD = {
    "agent": "tau2-airline",
    "realistic_workspace": "tau2-airline",
    "oracle_workspace": "tau2-airline-oracle",
    "experiment_id": "run-1",
    "base_url": "http://origin:8080",
    "domain": "airline",
}


@pytest.fixture
def stub_reingest(monkeypatch):
    """Stub the reingest touchpoints (no catalog import, no network); record every call."""
    calls = {"ingest": [], "platform_root": [], "catalog_root": []}

    def fake_resolve(explicit=None, **kw):
        calls["platform_root"].append(explicit)
        return Path(explicit) if explicit else Path("/stub/nemo-helix")

    def fake_load(root):
        calls["catalog_root"].append(Path(root))
        return "stub-catalog"

    def fake_ingest(base_url, export_dir, manifest, *, workspace_map, catalog, **kw):
        calls["ingest"].append(
            {
                "base_url": base_url,
                "export_dir": Path(export_dir),
                "manifest": manifest,
                "workspace_map": dict(workspace_map),
                "catalog": catalog,
                "require_empty": kw.get("require_empty", False),
            }
        )
        return {
            ws: {
                "workspace": target,
                "spans": {"ingested": manifest["counts"][ws]["spans"], "skipped": 0},
                "annotations": {"ingested": 0, "skipped": 0},
                "evaluator_results": {"ingested": 0, "skipped": 0},
            }
            for ws, target in workspace_map.items()
        }

    monkeypatch.setattr("evaluation.reingest.resolve_platform_root", fake_resolve)
    monkeypatch.setattr("evaluation.reingest.load_catalog", fake_load)
    monkeypatch.setattr("evaluation.reingest.ingest_bundle", fake_ingest)
    return calls


def test_list_prints_configured_subjects(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluation", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "nvq" in out
    assert "tau2-airline" in out
    assert "tau2-retail" in out


def test_analyze_unknown_subject_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nope"])
    with pytest.raises(SystemExit):
        cli.main()


def test_analyze_does_not_require_plugin_inference_key(monkeypatch, tmp_path, capsys):
    # Analyst credentials belong to the configured Platform Model provider, not
    # a plugin-owned environment variable.
    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "nvq", "--live", "--no-baseline-update"],
    )

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()
    assert "REPORT-OK" in capsys.readouterr().out


def test_analyze_live_happy_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)  # keep the suite out of the real evaluation/tmp
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "-v"])

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()
    out = capsys.readouterr().out
    assert "REPORT-OK" in out
    assert "Insights written" in out


def test_analyze_registers_and_tags_analyst_run_by_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(cli, "_register_analysis_evaluation", _REAL_REGISTER_ANALYSIS_EVALUATION)
    monkeypatch.setattr(cli, "mint_agent_id", lambda _base: "nemo-analyst-run-1")
    calls: dict[str, object] = {}

    def fake_ensure_experiment_group(base_url, workspace, name):
        calls["experiment"] = (base_url, workspace, name)
        return "experiment-1"

    def fake_create_experiment(base_url, workspace, **kwargs):
        calls["evaluation"] = (base_url, workspace, kwargs)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        calls["observability"] = self.analyst_evaluation
        return "REPORT-OK"

    monkeypatch.setattr(cli, "ensure_experiment_group", fake_ensure_experiment_group)
    monkeypatch.setattr(cli, "create_experiment", fake_create_experiment)
    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "analyze",
            "nvq",
            "--live",
            "--base",
            "http://localhost:8080",
            "--set",
            "agent=smoke-agent",
            "--set",
            "workspace=default",
            "--no-baseline-update",
        ],
    )

    cli.main()

    assert calls["experiment"] == (
        "http://localhost:8080",
        "default",
        "nemo-analyst",
    )
    assert calls["evaluation"] == (
        "http://localhost:8080",
        "default",
        {
            "name": "nemo-analyst-run-1",
            "experiment_group_id": "experiment-1",
            "dataset_name": "nemo-analyst:smoke-agent",
            "dataset_version": "live",
            "metadata": {
                "subject": "nvq",
                "target_agent": "smoke-agent",
                "test_case_name": "nvq",
                "state": "live",
            },
        },
    )
    context = calls["observability"]
    assert isinstance(context, AnalystEvaluationContext)
    assert context.evaluation_name == "nemo-analyst-run-1"
    assert context.test_case_name == "nvq"
    assert "Experiment 'nemo-analyst', Evaluation 'nemo-analyst-run-1'" in capsys.readouterr().out


def test_register_analysis_evaluation_leaves_custom_intake_auth_unchanged(monkeypatch):
    from evaluation.registry import Subject

    subject = Subject(
        "custom-intake",
        "intake",
        {
            "agent": "custom-agent",
            "workspace": "default",
            "base_url": "https://intake.example",
            "auth": "basic",
        },
    )
    monkeypatch.setattr(cli, "ensure_experiment_group", lambda *_args, **_kwargs: pytest.fail("must not register"))

    assert (
        _REAL_REGISTER_ANALYSIS_EVALUATION(
            subject=subject,
            subject_name=subject.name,
            record=None,
            state_label="live",
        )
        is None
    )


def test_analyze_checks_in_insights_with_spdx_and_provenance_by_default(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    if isolate_checked_in_insights["check_in"] is not None:
        monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    if isolate_checked_in_insights["write_manifest"] is not None:
        monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])
    monkeypatch.setattr(cli, "_analyst_sha256", lambda: "analyst-digest", raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live"])

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights: []\n", encoding="utf-8")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()

    contents = (
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "insights: []\n"
    )
    assert (checked_in / "nvq.yaml").read_text(encoding="utf-8") == contents
    manifest = yaml.safe_load((checked_in / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest == {
        "snapshots": {
            "nvq": {
                "analyst_sha256": "analyst-digest",
                "insights_sha256": hashlib.sha256(contents.encode()).hexdigest(),
                "state": "live",
            }
        }
    }


def test_analyze_checks_in_insights_and_merges_existing_manifest(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    prior = {
        "analyst_sha256": "old-analyst",
        "insights_sha256": "old-insights",
        "state": "state-v1",
    }
    (checked_in / "manifest.yaml").write_text(
        yaml.safe_dump({"snapshots": {"other": prior}}),
        encoding="utf-8",
    )
    (checked_in / "other.yaml").write_text("prior insights\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    if isolate_checked_in_insights["check_in"] is not None:
        monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    if isolate_checked_in_insights["write_manifest"] is not None:
        monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])
    monkeypatch.setattr(cli, "_analyst_sha256", lambda: "new-analyst", raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live"])

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights: []\n", encoding="utf-8")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()

    snapshots = yaml.safe_load((checked_in / "manifest.yaml").read_text(encoding="utf-8"))["snapshots"]
    assert snapshots["other"] == prior
    assert snapshots["nvq"]["state"] == "live"
    assert (checked_in / "other.yaml").read_text(encoding="utf-8") == "prior insights\n"


def test_check_in_single_insights_manifest_failure_leaves_directory_unchanged(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "nvq.yaml").write_text("old insights\n", encoding="utf-8")
    (checked_in / "other.yaml").write_text("other insights\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    source = tmp_path / "new-insights.yaml"
    source.write_text("insights: []\n", encoding="utf-8")
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])

    def fail_manifest(*_args, **_kwargs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(cli, "_write_insights_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="manifest failed"):
        cli._check_in_single_insights("nvq", source, "live")

    assert _checked_in_bytes(checked_in) == before


def test_analyze_checks_in_insights_requires_source_output(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path / "tmp")
    monkeypatch.setattr(cli, "INSIGHTS_DIR", tmp_path / "insights")
    if isolate_checked_in_insights["check_in"] is not None:
        monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    if isolate_checked_in_insights["write_manifest"] is not None:
        monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live"])

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)

    with pytest.raises(SystemExit, match="Analyst did not write"):
        cli.main()


def test_analyze_no_baseline_update_leaves_checked_in_files_untouched(monkeypatch, tmp_path):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    checked_path = checked_in / "nvq.yaml"
    manifest_path = checked_in / "manifest.yaml"
    checked_path.write_text("old insights\n", encoding="utf-8")
    manifest_path.write_text("old manifest\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--no-baseline-update"])

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights: []\n", encoding="utf-8")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()

    assert (runtime / "insights_nvq.yaml").read_text(encoding="utf-8") == "insights: []\n"
    assert checked_path.read_text(encoding="utf-8") == "old insights\n"
    assert manifest_path.read_text(encoding="utf-8") == "old manifest\n"


def _write_analyst_lock(
    path: Path,
    *,
    unrelated_version: str = "1",
    transitive_version: str = "1",
    litellm_version: str = "1",
) -> None:
    path.write_text(
        f"""
version = 1

[[package]]
name = "unrelated"
version = "{unrelated_version}"

[[package]]
name = "nemo-insights-plugin"
version = "1"
dependencies = [
    {{ name = "nooa" }},
]

[[package]]
name = "nooa"
version = "2"
dependencies = [{{ name = "transitive" }}, {{ name = "litellm" }}]

[[package]]
name = "transitive"
version = "{transitive_version}"

[[package]]
name = "litellm"
version = "{litellm_version}"
""",
        encoding="utf-8",
    )


def test_analyst_hash_tracks_complete_plugin_source(tmp_path):
    plugin_root = tmp_path / "nemo_insights_plugin"
    analyst_root = plugin_root / "analyst"
    analyst_root.mkdir(parents=True)
    (analyst_root / "agent.py").write_text("AGENT = 1\n", encoding="utf-8")
    entities = plugin_root / "entities.py"
    schema = plugin_root / "schema.py"
    entities.write_text("ENTITY = 1\n", encoding="utf-8")
    schema.write_text("SCHEMA = 1\n", encoding="utf-8")
    lockfile = tmp_path / "uv.lock"
    _write_analyst_lock(lockfile)

    initial = cli._analyst_sha256(plugin_root, lockfile)
    entities.write_text("ENTITY = 2\n", encoding="utf-8")
    assert cli._analyst_sha256(plugin_root, lockfile) != initial

    entities.write_text("ENTITY = 1\n", encoding="utf-8")
    schema.write_text("SCHEMA = 2\n", encoding="utf-8")
    assert cli._analyst_sha256(plugin_root, lockfile) != initial


def test_analyst_hash_tracks_resolved_dependency_closure(tmp_path):
    plugin_root = tmp_path / "nemo_insights_plugin"
    plugin_root.mkdir()
    (plugin_root / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    lockfile = tmp_path / "uv.lock"
    _write_analyst_lock(lockfile)
    initial = cli._analyst_sha256(plugin_root, lockfile)

    _write_analyst_lock(lockfile, unrelated_version="changed")
    assert cli._analyst_sha256(plugin_root, lockfile) == initial

    _write_analyst_lock(lockfile, unrelated_version="changed", transitive_version="changed")
    transitive_changed = cli._analyst_sha256(plugin_root, lockfile)
    assert transitive_changed != initial

    _write_analyst_lock(
        lockfile,
        unrelated_version="changed",
        transitive_version="changed",
        litellm_version="changed",
    )
    assert cli._analyst_sha256(plugin_root, lockfile) != transitive_changed


def test_analyst_hash_defaults_to_platform_paths():
    assert cli.ANALYST_SOURCE_ROOT == cli.HERE.parent / "src" / "nemo_insights_plugin"
    assert cli.ANALYST_LOCKFILE == cli.HERE.parents[2] / "uv.lock"


def test_analyst_hash_resolves_platform_lock():
    assert re.fullmatch(r"[0-9a-f]{64}", cli._analyst_sha256())


def _analyze_all_subjects():
    from evaluation.registry import Subject

    return {
        "zeta": Subject("zeta", "intake", {"workspace": "z", "state": "state-zeta"}),
        "alpha": Subject("alpha", "benchmark", {"workspace": "a", "state": "state-alpha"}),
        "producer": Subject("producer", "harbor", {"workspace": "p"}),
    }


def _checked_in_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }


def test_analyze_all_runs_every_pinned_analyzable_subject_in_sorted_order(
    monkeypatch,
    tmp_path,
):
    runtime = tmp_path / "tmp"
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())

    def fake_run(command, *, check):
        calls.append(command)
        name = command[command.index("analyze") + 1]
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"insights_{name}.yaml").write_text(f"insights: [{name}]\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "analyze",
            "all",
            "--no-baseline-update",
            "--base",
            "http://platform",
            "--platform-root",
            "/platform/root",
            "--summary-md",
            "/tmp/summary.md",
            "--verbose",
        ],
    )

    cli.main()

    assert calls == [
        [
            sys.executable,
            "-m",
            "evaluation",
            "analyze",
            "alpha",
            "--no-baseline-update",
            "--base",
            "http://platform",
            "--platform-root",
            "/platform/root",
            "--summary-md",
            "/tmp/summary.md",
            "--verbose",
        ],
        [
            sys.executable,
            "-m",
            "evaluation",
            "analyze",
            "zeta",
            "--no-baseline-update",
            "--base",
            "http://platform",
            "--platform-root",
            "/platform/root",
            "--summary-md",
            "/tmp/summary.md",
            "--verbose",
        ],
    ]


def test_analyze_all_validates_every_pin_before_subprocess_execution(monkeypatch):
    subjects = _analyze_all_subjects()
    subjects["zeta"] = dataclasses.replace(subjects["zeta"], config={"workspace": "z"})
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "load_registry", lambda _path: subjects)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, *, check: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    with pytest.raises(SystemExit, match="missing: zeta"):
        cli.main()

    assert calls == []


def test_analyze_all_child_failure_leaves_checked_in_directory_unchanged(
    monkeypatch,
    tmp_path,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "stale.yaml").write_text("old stale\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        if name == "zeta":
            raise subprocess.CalledProcessError(7, command)
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"insights_{name}.yaml").write_text("new alpha\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    with pytest.raises(SystemExit, match="zeta failed with exit code 7"):
        cli.main()

    assert _checked_in_bytes(checked_in) == before


def test_analyze_all_missing_child_output_leaves_checked_in_directory_unchanged(
    monkeypatch,
    tmp_path,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        if name == "alpha":
            runtime.mkdir(parents=True, exist_ok=True)
            (runtime / "insights_alpha.yaml").write_text("new alpha\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    with pytest.raises(SystemExit, match="without Insights output for: zeta"):
        cli.main()

    assert _checked_in_bytes(checked_in) == before


def test_analyze_all_stale_output_cannot_mask_missing_child_output(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    runtime = tmp_path / "tmp"
    runtime.mkdir()
    stale = runtime / "insights_zeta.yaml"
    stale.write_text("stale zeta\n", encoding="utf-8")
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())
    monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        if name == "alpha":
            (runtime / "insights_alpha.yaml").write_text("new alpha\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    with pytest.raises(SystemExit, match="without Insights output for: zeta"):
        cli.main()

    assert _checked_in_bytes(checked_in) == before
    (backup,) = runtime.glob("backup-*")
    assert (backup / stale.name).read_text(encoding="utf-8") == "stale zeta\n"


def test_analyze_all_top_level_no_baseline_update_skips_promotion(monkeypatch, tmp_path):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"insights_{name}.yaml").write_text(f"new {name}\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all", "--no-baseline-update"])

    cli.main()

    assert _checked_in_bytes(checked_in) == before


def test_analyze_all_successfully_swaps_complete_set_and_removes_stale_yaml(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "stale.yaml").write_text("old stale\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())
    monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])
    monkeypatch.setattr(cli, "_analyst_sha256", lambda: "analyst-digest")

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"insights_{name}.yaml").write_text(f"insights: [{name}]\n", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    cli.main()

    assert {path.name for path in checked_in.glob("*.yaml")} == {
        "alpha.yaml",
        "zeta.yaml",
        "manifest.yaml",
    }
    snapshots = yaml.safe_load((checked_in / "manifest.yaml").read_text(encoding="utf-8"))["snapshots"]
    assert snapshots == {
        "alpha": {
            "analyst_sha256": "analyst-digest",
            "insights_sha256": hashlib.sha256((checked_in / "alpha.yaml").read_bytes()).hexdigest(),
            "state": "state-alpha",
        },
        "zeta": {
            "analyst_sha256": "analyst-digest",
            "insights_sha256": hashlib.sha256((checked_in / "zeta.yaml").read_bytes()).hexdigest(),
            "state": "state-zeta",
        },
    }


def test_analyze_all_swap_failure_rolls_back_old_directory(
    monkeypatch,
    tmp_path,
    isolate_checked_in_insights,
):
    runtime = tmp_path / "tmp"
    checked_in = tmp_path / "insights"
    checked_in.mkdir()
    (checked_in / "alpha.yaml").write_text("old alpha\n", encoding="utf-8")
    (checked_in / "manifest.yaml").write_text("old manifest\n", encoding="utf-8")
    before = _checked_in_bytes(checked_in)
    monkeypatch.setattr(cli, "TMP", runtime)
    monkeypatch.setattr(cli, "INSIGHTS_DIR", checked_in)
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())
    monkeypatch.setattr(cli, "_check_in_insights", isolate_checked_in_insights["check_in"])
    monkeypatch.setattr(cli, "_write_insights_manifest", isolate_checked_in_insights["write_manifest"])
    monkeypatch.setattr(cli, "_analyst_sha256", lambda: "analyst-digest")

    def fake_run(command, *, check):
        name = command[command.index("analyze") + 1]
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / f"insights_{name}.yaml").write_text(f"insights: [{name}]\n", encoding="utf-8")

    real_replace = cli.os.replace

    def fail_staging_swap(source, destination):
        source_path = Path(source)
        if source_path.is_dir() and source_path.name.startswith(".insights.staging-"):
            raise OSError("simulated swap failure")
        real_replace(source, destination)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.os, "replace", fail_staging_swap)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all"])

    with pytest.raises(OSError, match="simulated swap failure"):
        cli.main()

    assert _checked_in_bytes(checked_in) == before


@pytest.mark.parametrize(
    "flags",
    [
        ["--live"],
        ["--state", "state-v1"],
        ["--since", "1d"],
        ["--set", "seed=1"],
        ["--update-insights"],
    ],
)
def test_analyze_all_rejects_incompatible_flags_before_execution(monkeypatch, flags):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "load_registry", lambda _path: _analyze_all_subjects())
    monkeypatch.setattr(cli.subprocess, "run", lambda command, *, check: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "all", *flags])

    with pytest.raises(SystemExit, match="cannot be combined"):
        cli.main()

    assert calls == []


def test_missing_registry_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REGISTRY_PATH", tmp_path / "nope.toml")
    monkeypatch.setattr(sys, "argv", ["evaluation", "list"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "registry" in str(exc.value).lower()


def test_empty_since_is_epoch_no_lower_bound(monkeypatch, tmp_path, capsys):
    # `--live --since ''` means "no lower bound" and must NOT fall back to the stanza
    # default. The analyst must receive the EPOCH, not None — a None bound lets the
    # read API reapply its implicit 30d lookback (export.py's `since or EPOCH`
    # discipline, mirrored).
    from evaluation.registry import Subject

    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    subject = Subject(
        "demo",
        "intake",
        {"agent": "a", "workspace": "w", "base_url": "u", "since": "7d"},
    )
    monkeypatch.setattr(cli, "load_registry", lambda _path: {"demo": subject})
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "demo", "--live", "--since", ""])
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["since"] = since
        return "ok"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    cli.main()
    assert seen["since"] == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert "since: 1970-01-01T00:00:00+00:00 (--since (epoch — no lower bound))" in capsys.readouterr().out


def test_load_dotenv_sets_missing_and_preserves_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# a comment\nFOO=from_file\nBAR=baz\nexport QUX="q v"\n\nNOEQ_LINE\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("FOO", "from_shell")  # already set -> file must NOT override
    monkeypatch.delenv("BAR", raising=False)
    monkeypatch.delenv("QUX", raising=False)
    cli._load_dotenv(env_file)
    assert os.environ["FOO"] == "from_shell"  # setdefault: real env wins
    assert os.environ["BAR"] == "baz"
    assert os.environ["QUX"] == "q v"  # quotes stripped, export prefix dropped


def test_load_dotenv_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_SENTINEL", raising=False)
    cli._load_dotenv(tmp_path / "nope.env")  # must not raise
    assert os.environ.get("NOPE_SENTINEL") is None  # and sets nothing


def test_aws_cli_version(monkeypatch):
    result = subprocess.CompletedProcess([], 0, stdout="aws-cli/2.36.24 Python/3.13.11", stderr="")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: result)
    assert cli._aws_cli_version("/usr/bin/aws") == (2, 36, 24)


@pytest.mark.parametrize(
    "failure",
    [OSError(), subprocess.TimeoutExpired("/usr/bin/aws", 5), subprocess.CompletedProcess([], 1)],
)
def test_aws_cli_version_probe_failure(monkeypatch, failure):
    def fail(*args, **kwargs):
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(cli.subprocess, "run", fail)
    assert cli._aws_cli_version("/usr/bin/aws") is None


def test_doctor_ready(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.check", lambda self: [])
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/aws")
    monkeypatch.setattr(cli, "_aws_cli_version", lambda _executable: (2, 33, 0))
    monkeypatch.setenv(cli.release.ACCESS_KEY_ENV, "team-test")
    monkeypatch.setenv(cli.release.SECRET_KEY_ENV, "secret")
    monkeypatch.setattr(sys, "argv", ["evaluation", "doctor", "nvq"])
    cli.main()
    out = capsys.readouterr().out
    assert "✓" in out and "ready" in out
    # doctor checks the adapter's (live-analysis) prerequisites, so the ready
    # hint must name the live flow, not the pinned restore
    assert "analyze nvq --live" in out


def test_doctor_flags_missing_aws(monkeypatch, capsys):
    """Pinned/--state analyze shells out to AWS CLI; doctor must flag its absence."""
    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.check", lambda self: [])
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.setenv(cli.release.ACCESS_KEY_ENV, "team-test")
    monkeypatch.setenv(cli.release.SECRET_KEY_ENV, "secret")
    monkeypatch.setattr(sys, "argv", ["evaluation", "doctor", "nvq"])
    cli.main()
    out = capsys.readouterr().out
    assert "✗" in out
    assert "AWS CLI 2.33.0+ (needed for pinned/--state analyze; install with `brew install awscli`)" in out


@pytest.mark.parametrize(
    ("version", "message"),
    [(None, "could not determine AWS CLI version"), ((2, 32, 34), "AWS CLI 2.33.0+")],
)
def test_doctor_flags_unsupported_aws(monkeypatch, capsys, version, message):
    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.check", lambda self: [])
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/aws")
    monkeypatch.setattr(cli, "_aws_cli_version", lambda _executable: version)
    monkeypatch.setenv(cli.release.ACCESS_KEY_ENV, "team-test")
    monkeypatch.setenv(cli.release.SECRET_KEY_ENV, "secret")
    cli._doctor(cli.load_registry(cli.REGISTRY_PATH), "nvq")
    assert message in capsys.readouterr().out


def test_doctor_lists_unmet(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv(cli.release.ACCESS_KEY_ENV, raising=False)
    monkeypatch.delenv(cli.release.SECRET_KEY_ENV, raising=False)
    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.check", lambda self: ["config key 'agent'"])
    monkeypatch.setattr(sys, "argv", ["evaluation", "doctor", "nvq"])
    cli.main()
    out = capsys.readouterr().out
    assert "✗" in out and "config key 'agent'" in out
    assert "CSS S3 credentials in evaluation/.env" in out
    assert cli.release.ACCESS_KEY_ENV not in out
    assert cli.release.SECRET_KEY_ENV not in out
    assert "INFERENCE_API_KEY" not in out


def test_run_records_and_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "TMP", tmp_path)

    async def fake_produce(self):
        return {
            "agent": "tau2-airline-xyz",
            "realistic_workspace": "tau2-airline-ts-realistic",
            "oracle_workspace": "tau2-airline-ts-realistic-oracle",
            "base_url": "u",
            "domain": "airline",
        }

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.produce", fake_produce)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "tau2-airline"])
    cli.main()
    out = capsys.readouterr().out
    assert "recorded" in out and "tau2-airline-xyz" in out
    # the hint must name the live flow: bare analyze restores a pinned state,
    # which would ignore the run that was just recorded
    assert "analyze tau2-airline --live" in out
    from evaluation.runstore import load_run

    recorded_run = load_run(tmp_path / "tau2-airline.run.json")
    assert recorded_run is not None
    assert recorded_run["agent"] == "tau2-airline-xyz"


def test_analyze_live_passes_record_to_analyze(monkeypatch, tmp_path):
    from evaluation.runstore import save_run

    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setenv("INFERENCE_API_KEY", "sk")
    save_run(
        tmp_path / "tau2-airline.run.json",
        {
            "agent": "tau2-airline-xyz",
            "realistic_workspace": "tau2-airline-ts-realistic",
            "oracle_workspace": "tau2-airline-ts-realistic-oracle",
            "base_url": "u",
            "domain": "airline",
        },
    )
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["record"] = record
        return "REPORT"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline", "--live"])
    cli.main()
    assert seen["record"]["agent"] == "tau2-airline-xyz"
    assert seen["record"]["realistic_workspace"] == "tau2-airline-ts-realistic"  # live mode: no remap


def test_run_intake_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "nvq"])
    with pytest.raises(SystemExit):
        cli.main()


def test_run_base_replaces_local(monkeypatch, tmp_path):
    """`run --base URL` points the subject at URL (the old --local, generalized)."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    seen: dict = {}

    async def fake_produce(self):
        seen["base_url"] = self.subject.config["base_url"]
        return {
            "agent": "a",
            "realistic_workspace": "a",
            "oracle_workspace": None,
            "base_url": self.subject.config["base_url"],
            "domain": "airline",
        }

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.produce", fake_produce)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "tau2-airline", "--base", "http://localhost:8080"])
    cli.main()
    assert seen["base_url"] == "http://localhost:8080"


def test_with_base_drops_remote_auth_by_default() -> None:
    from evaluation.registry import Subject

    glamr = Subject(
        "glamr",
        "intake",
        {
            "agent": "glamr",
            "workspace": "default",
            "base_url": "https://remote",
            "auth": "basic",
            "intake_path_prefix": "/api/intake",
            "auth_user_env": "GLAMR_INTAKE_USER",
            "auth_password_env": "GLAMR_INTAKE_PASSWORD",
        },
    )

    assert cli._with_base(glamr, "http://localhost:8080").config == {
        "agent": "glamr",
        "workspace": "default",
        "base_url": "http://localhost:8080",
    }


def test_with_base_preserves_remote_auth_when_requested() -> None:
    from evaluation.registry import Subject

    glamr = Subject(
        "glamr",
        "intake",
        {
            "agent": "glamr",
            "workspace": "default",
            "base_url": "https://remote",
            "auth": "basic",
            "intake_path_prefix": "/api/intake",
            "auth_user_env": "GLAMR_INTAKE_USER",
            "auth_password_env": "GLAMR_INTAKE_PASSWORD",
        },
    )

    assert cli._with_base(glamr, "https://remote", drop_auth=False).config["auth"] == "basic"


def test_analyze_live_base_overrides_stanza(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setenv("INFERENCE_API_KEY", "sk")
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["base_url"] = self.subject.config["base_url"]
        return "ok"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--base", "http://localhost:8080"])
    cli.main()
    assert seen["base_url"] == "http://localhost:8080"


def test_analyze_glamr_live_base_preserves_remote_auth(monkeypatch, tmp_path):
    from evaluation.adapters import EvaluationAdapter, IntakeAdapter
    from evaluation.registry import Subject

    glamr = Subject(
        "glamr",
        "intake",
        {
            "agent": "glamr",
            "workspace": "default",
            "base_url": "https://original.example",
            "auth": "basic",
            "intake_path_prefix": "/glamr/intake",
            "auth_user_env": "GLAMR_INTAKE_USER",
            "auth_password_env": "GLAMR_INTAKE_PASSWORD",
        },
    )
    monkeypatch.setattr(cli, "load_registry", lambda _path: {"glamr": glamr})
    monkeypatch.setattr(cli, "_register_analysis_evaluation", _REAL_REGISTER_ANALYSIS_EVALUATION)
    monkeypatch.setattr(cli, "_load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setenv("GLAMR_INTAKE_USER", "intake-user")
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "secret")
    built: dict[str, object] = {}
    real_build_adapter = cli.build_adapter

    def capture_build_adapter(
        subject: Subject,
        *,
        analyst_evaluation: AnalystEvaluationContext | None = None,
    ) -> EvaluationAdapter:
        built.update(subject.config)
        built["analyst_evaluation"] = analyst_evaluation
        return real_build_adapter(subject, analyst_evaluation=analyst_evaluation)

    async def fake_analyze(
        self: IntakeAdapter,
        *,
        record: dict[str, object] | None,
        since: datetime | None,
        verbose: bool,
        out_path: Path,
    ) -> str:
        return "REPORT-OK"

    monkeypatch.setattr(cli, "build_adapter", capture_build_adapter)
    monkeypatch.setattr(IntakeAdapter, "analyze", fake_analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "glamr", "--live", "--base", "https://override.example"],
    )

    cli.main()

    assert built["analyst_evaluation"] is None

    assert built["base_url"] == "https://override.example"
    assert {key: built[key] for key in cli._REMOTE_AUTH_KEYS} == {
        "auth": "basic",
        "intake_path_prefix": "/glamr/intake",
        "auth_user_env": "GLAMR_INTAKE_USER",
        "auth_password_env": "GLAMR_INTAKE_PASSWORD",
    }


def test_analyze_live_base_retargets_record(monkeypatch, tmp_path):
    """--live --base must also retarget the benchmark run record."""
    from evaluation.runstore import save_run

    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setenv("INFERENCE_API_KEY", "sk")
    save_run(tmp_path / "tau2-airline.run.json", dict(_RUN_RECORD))  # recorded at http://origin:8080
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["record"] = record
        return "REPORT"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "tau2-airline", "--live", "--base", "http://localhost:8080"],
    )
    cli.main()
    assert seen["record"]["base_url"] == "http://localhost:8080"  # not the recorded origin URL
    assert seen["record"]["realistic_workspace"] == "tau2-airline"  # workspaces still un-remapped (live)


def test_run_set_overrides_reach_adapter(monkeypatch, tmp_path):
    captured = {}

    async def fake_produce(self):
        captured.update(self.subject.config)
        return {"agent": "a", "realistic_workspace": "w", "oracle_workspace": None}

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.produce", fake_produce)
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "run", "tau2-airline", "--set", "num_tasks=2", "--set", "user_llm=other/model"],
    )
    cli.main()
    assert captured["num_tasks"] == 2  # int-coerced, not "2"
    assert captured["user_llm"] == "other/model"


def test_set_rejects_malformed_pair(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "tau2-airline", "--set", "num_tasks"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--set" in str(exc.value)


def test_set_rejects_non_int_for_int_key(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "tau2-airline", "--set", "num_tasks=lots"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "integer" in str(exc.value)


def test_analyze_live_set_overrides_reach_adapter(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    captured = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        captured.update(self.subject.config)
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--set", "agent=other-agent"])
    cli.main()
    assert captured["agent"] == "other-agent"


def _bool_subject():
    """A benchmark stanza carrying include_rewards=true (the toml keeps it as a commented default)."""
    from evaluation.registry import Subject

    return Subject(
        "demo",
        "benchmark",
        {
            "domain": "airline",
            "base_url": "u",
            "workspace": "w",
            "agent_llm": "m",
            "user_llm": "m",
            "include_rewards": True,
        },
    )


def _capture_produce(monkeypatch, captured):
    async def fake_produce(self):
        captured.update(self.subject.config)
        return {"agent": "a", "realistic_workspace": "w", "oracle_workspace": None, "base_url": "u"}

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.produce", fake_produce)


def test_set_bool_false_is_false(monkeypatch, tmp_path):
    """--set include_rewards=false must land as bool False (a "false" STRING is truthy —
    the adapter would mint the oracle twin and snapshot scoping would export it)."""
    from evaluation import artifact
    from evaluation.registry import Subject

    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": _bool_subject()})
    monkeypatch.setattr(cli, "TMP", tmp_path)
    captured: dict = {}
    _capture_produce(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "demo", "--set", "include_rewards=false"])
    cli.main()
    assert captured["include_rewards"] is False
    # snapshot scoping drops the oracle twin for the overridden subject
    assert artifact.workspaces_for_subject(Subject("demo", "benchmark", captured)) == ["w"]
    # case-insensitive: TRUE/False parse too
    assert cli._apply_overrides(_bool_subject(), ["include_rewards=FALSE"]).config["include_rewards"] is False
    assert cli._apply_overrides(_bool_subject(), ["include_rewards=True"]).config["include_rewards"] is True


def test_set_bool_garbage_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": _bool_subject()})
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "demo", "--set", "include_rewards=maybe"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "include_rewards" in message  # the exit names the key
    assert "true" in message.lower() and "false" in message.lower()
    assert "maybe" in message


def test_benchmark_stanzas_carry_include_rewards():
    """The tau2 benchmark stanzas must carry include_rewards=true so that --set include_rewards=false
    engages typed coercion (a missing key stays as a truthy "false" STRING)."""
    from evaluation.registry import load_registry

    registry = load_registry(cli.HERE / "evaluations.toml")
    assert registry["tau2-airline"].config["include_rewards"] is True
    assert registry["tau2-retail"].config["include_rewards"] is True


def test_set_int_coerced(monkeypatch, tmp_path):
    """Stanza-typed coercion: `seed` (the tau2 RNG, an int in the stanza) still parses to int."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    captured: dict = {}
    _capture_produce(monkeypatch, captured)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "run", "tau2-airline", "--set", "seed=42", "--set", "num_tasks=2"],
    )
    cli.main()
    assert captured["seed"] == 42 and isinstance(captured["seed"], int)
    assert captured["num_tasks"] == 2 and isinstance(captured["num_tasks"], int)


def test_set_float_coerced_by_stanza_type():
    from evaluation.registry import Subject

    subject = Subject("demo", "benchmark", {"user_temperature": 0.0})
    assert cli._apply_overrides(subject, ["user_temperature=0.7"]).config["user_temperature"] == 0.7
    with pytest.raises(SystemExit) as exc:
        cli._apply_overrides(subject, ["user_temperature=warm"])
    assert "user_temperature" in str(exc.value)


def test_set_unknown_key_stays_string(monkeypatch, tmp_path):
    """Keys absent from the stanza are allowed and stay verbatim strings (no guessing)."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    captured: dict = {}
    _capture_produce(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "tau2-airline", "--set", "brand_new_key=7"])
    cli.main()
    assert captured["brand_new_key"] == "7"


def test_set_bool_literal_coerces_even_when_key_absent(monkeypatch, tmp_path):
    """--set include_rewards=false on a stanza WITHOUT the key must land as bool False —
    a 'false' STRING is truthy, so the adapter would silently mint the oracle twin."""
    from evaluation.registry import Subject

    subject = Subject(
        "demo",
        "benchmark",
        {"domain": "airline", "base_url": "u", "workspace": "w", "agent_llm": "m", "user_llm": "m"},
    )
    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": subject})
    monkeypatch.setattr(cli, "TMP", tmp_path)
    captured: dict = {}
    _capture_produce(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", ["evaluation", "run", "demo", "--set", "include_rewards=false"])
    cli.main()
    assert captured["include_rewards"] is False
    # bare true/false literals coerce case-insensitively; other unknown-key values stay strings
    assert cli._apply_overrides(subject, ["new_flag=TRUE"]).config["new_flag"] is True
    assert cli._apply_overrides(subject, ["new_flag=False"]).config["new_flag"] is False
    assert cli._apply_overrides(subject, ["new_flag=0"]).config["new_flag"] == "0"


def test_doctor_flags_missing_benchmark_argv_keys(monkeypatch, capsys):
    """A stanza missing a key `tau2 run` hard-indexes gets a doctor line, not a KeyError at run time."""
    from evaluation.registry import Subject

    monkeypatch.setattr(cli, "_load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("INFERENCE_API_KEY", "sk")
    subject = Subject(
        "demo",
        "benchmark",
        {"domain": "airline", "base_url": "u", "workspace": "w", "agent_llm": "m", "user_llm": "m"},
    )
    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": subject})
    monkeypatch.setattr(sys, "argv", ["evaluation", "doctor", "demo"])
    cli.main()
    out = capsys.readouterr().out
    for key in ("task_split_name", "num_trials", "seed", "max_concurrency"):
        assert f"config key '{key}'" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "nvq", "--pinned"],
        ["analyze", "nvq", "--latest"],
        ["analyze", "nvq", "--ref", "state-v4"],
        ["analyze", "nvq", "--from", "b.tar.zst"],
        ["analyze", "nvq", "--local"],
        ["insights", "nvq"],  # the backward-compatible alias is gone
        ["restore", "--from", "b.tar.zst"],
        ["restore", "--latest"],
        ["restore", "--ref", "state-v4"],
        ["restore", "--pinned"],
        ["run", "tau2-airline", "--local"],
        ["snapshot", "nvq", "--local"],
    ],
)
def test_removed_flags_error(monkeypatch, argv):
    """Every retired source/target flag (and the insights alias) is an argparse usage error."""
    monkeypatch.setattr(sys, "argv", ["evaluation", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2  # argparse usage error, not a evaluation message


def test_analyze_defaults_to_pinned_stanza_entry(monkeypatch, tmp_path, stub_reingest):
    """Bare `analyze <subject>` restores the state pinned in its registry stanza."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        records={"tau2-airline.run.json": json.dumps(_RUN_RECORD)},
    )
    seen: dict = {}
    monkeypatch.setattr(
        "evaluation.release.resolve_state",
        lambda state, *, subject: seen.update(state=state, subject=subject) or "state-v6",
    )
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline"])
    cli.main()
    assert seen["state"] is None, "no flag -> the stanza decides"
    assert seen["subject"].name == "tau2-airline"
    assert seen["subject"].config["state"] == "state-v6"
    assert stub_reingest["ingest"][0]["base_url"] == "http://localhost:8080"  # default restore target: local platform


def test_analyze_live_skips_restore_and_targets_stanza(monkeypatch, tmp_path, stub_reingest):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(
        "evaluation.release.download_ref",
        lambda *a, **k: pytest.fail("--live must not download a state"),
    )
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["base_url"] = self.subject.config["base_url"]
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live"])
    cli.main()
    assert seen["base_url"] == "https://nemo-helix-freeplay.dev.aire.nvidia.com"  # the stanza, not localhost
    assert stub_reingest["ingest"] == []  # no restore in live mode


def test_analyze_live_since_precedence_and_print(monkeypatch, tmp_path, capsys):
    """Live-mode since: --since beats the stanza beats the 30d default; one line names the origin."""
    from evaluation.registry import Subject

    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["since"] = since
        return "ok"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    stanza = Subject("demo", "intake", {"agent": "a", "workspace": "w", "base_url": "u", "since": "2026-06-15"})
    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": stanza})

    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "demo", "--live", "--since", "2026-07-01"])
    cli.main()
    assert seen["since"] == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert "since: 2026-07-01T00:00:00+00:00 (--since)" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "demo", "--live"])
    cli.main()
    assert seen["since"] == datetime(2026, 6, 15, tzinfo=timezone.utc)
    assert "since: 2026-06-15T00:00:00+00:00 (stanza)" in capsys.readouterr().out

    bare = Subject("demo", "intake", {"agent": "a", "workspace": "w", "base_url": "u"})
    monkeypatch.setattr(cli, "load_registry", lambda _p: {"demo": bare})
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "demo", "--live"])
    cli.main()
    expected = datetime.now(timezone.utc) - timedelta(days=30)
    assert abs((seen["since"] - expected).total_seconds()) < 60  # computed client-side, ~now-30d
    out = capsys.readouterr().out
    assert "(default 30d)" in out
    assert out.count("since: ") == 1  # exactly one origin line


def test_analyze_live_and_state_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--state", "state-v6"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_analyze_state_accepts_ref_and_file(monkeypatch, tmp_path, capsys, stub_reingest):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)

    # ref form: resolved verbatim (no lock), downloaded; fixture suffix = the ref
    dl = _make_export_bundle(tmp_path / "dl" / "state-v4.tar.zst", workspaces=("nvq",))
    downloads: list = []
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: downloads.append(ref) or dl)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v4"])
    cli.main()
    assert downloads == ["state-v4"]
    assert stub_reingest["ingest"][0]["workspace_map"] == {"nvq": "nvq-state-v4"}
    assert "resolved state ref: state-v4" in capsys.readouterr().out

    # file form: an existing path wins over the ref pattern; fixture suffix = content digest
    local = _make_export_bundle(tmp_path / "local.tar.zst", workspaces=("nvq",))
    digest = hashlib.sha256(local.read_bytes()).hexdigest()[:8]
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(local)])
    cli.main()
    assert stub_reingest["ingest"][1]["workspace_map"] == {"nvq": f"nvq-{digest}"}
    assert downloads == ["state-v4"]  # the file form never downloads


def test_analyze_state_directory_named_like_ref_resolves_ref(monkeypatch, tmp_path, stub_reingest):
    """A DIRECTORY named state-v6 in cwd must not shadow the published ref (is_file, not exists)."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path / "tmp")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state-v6").mkdir()
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v6.tar.zst", workspaces=("nvq",))
    downloads: list = []
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: downloads.append(ref) or bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v6"])
    cli.main()
    assert downloads == ["state-v6"]  # the dir fell through to ref resolution


def test_analyze_state_missing_path_like_value_exits_no_such_file(monkeypatch, tmp_path):
    """A path-shaped --state that names nothing on disk must exit 'no such bundle file',
    not resolve_state's malformed-ref message."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path / "tmp")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "evaluation.release.download_ref",
        lambda *a, **k: pytest.fail("a missing file must not resolve"),
    )
    for value in (str(tmp_path / "cand.tar.zst"), "./cand.tar.zst", "cand.tar.zst"):
        monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", value])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert str(exc.value) == f"no such bundle file: {value}"


def test_analyze_file_state_provenance_names_local_file(monkeypatch, tmp_path, stub_reingest):
    """File-source provenance is disambiguated from refs: label = '<name> (local file <digest8>)'."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    local = _make_export_bundle(tmp_path / "cand.tar.zst", workspaces=("nvq",))
    digest = hashlib.sha256(local.read_bytes()).hexdigest()[:8]

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights: []\n")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "nvq", "--state", str(local), "--summary-md", str(summary)],
    )
    cli.main()
    assert f"### analyze @ cand.tar.zst (local file {digest}) (nvq)" in summary.read_text()


def test_analyze_file_shadowing_ref_name_prints_note(monkeypatch, tmp_path, capsys, stub_reingest):
    """A local FILE named exactly like a published ref wins over ref resolution — say so."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path / "tmp")
    monkeypatch.chdir(tmp_path)
    _make_export_bundle(tmp_path / "state-v6", workspaces=("nvq",))
    monkeypatch.setattr(
        "evaluation.release.download_ref",
        lambda *a, **k: pytest.fail("the local file wins — no download"),
    )

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v6"])
    cli.main()
    out = capsys.readouterr().out
    assert "note: 'state-v6' is a local file shadowing a published ref name — analyzing the file" in out


def test_analyze_base_overrides_fixture_target(monkeypatch, tmp_path, stub_reingest):
    """Pinned/state mode restores onto --base (and the analyst reads it) instead of localhost."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v4.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["base_url"] = self.subject.config["base_url"]
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "nvq", "--state", "state-v4", "--base", "http://ci-host:8080"],
    )
    cli.main()
    assert stub_reingest["ingest"][0]["base_url"] == "http://ci-host:8080"
    assert seen["base_url"] == "http://ci-host:8080"


def test_analyze_summary_md_records_restore_provenance(monkeypatch, tmp_path, capsys, stub_reingest):
    """A pinned/state analyze must record what it restored, before the insights section."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v4.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights: []\n")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "nvq", "--state", "state-v4", "--summary-md", str(summary)],
    )
    cli.main()
    text = summary.read_text()
    assert "analyze @ state-v4" in text
    # provenance line lands before the insights rendering, not after
    assert text.index("analyze @ state-v4") < text.index("## Insights")


def test_analyze_same_ref_twice_succeeds(monkeypatch, tmp_path, stub_reingest):
    """Re-running analyze with the same --state ref must not crash on the leftover download.

    The real download_ref runs against a fake S3 call. The second invocation
    reuses the immutable local cache, while restore remains idempotent.
    """
    from evaluation import release

    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)

    def fake_aws(_store, *args):
        _make_export_bundle(Path(args[-1]), workspaces=("nvq",))
        return ""

    monkeypatch.setattr(release, "_aws", fake_aws)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v4"])
    cli.main()
    cli.main()  # second run with the same ref must not raise
    assert len(stub_reingest["ingest"]) == 2  # restore ran both times (idempotency lives in reingest)


def test_analyze_second_restore_reports_skip(monkeypatch, tmp_path, capsys, stub_reingest):
    """When reingest's guard skips an already-restored workspace, the status line says so."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v4.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    def skip_ingest(base_url, export_dir, manifest, *, workspace_map, catalog, **kw):
        return {
            ws: {
                "workspace": target,
                "spans": {"ingested": 0, "skipped": manifest["counts"][ws]["spans"]},
                "annotations": {"ingested": 0, "skipped": 0},
                "evaluator_results": {"ingested": 0, "skipped": 0},
            }
            for ws, target in workspace_map.items()
        }

    monkeypatch.setattr("evaluation.reingest.ingest_bundle", skip_ingest)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v4"])
    cli.main()
    out = capsys.readouterr().out
    assert "0 spans ingested, 1 skipped" in out
    assert "REPORT-OK" in out  # the analysis still runs against the already-restored fixture


# --------------------------------------------------------------------------- #
# restore: export bundles re-ingest (additive); legacy tars exit with a pointer
# --------------------------------------------------------------------------- #


def test_restore_into_uses_exact_target(monkeypatch, tmp_path, stub_reingest) -> None:
    bundle = _make_export_bundle(tmp_path / "state.tar.zst", workspaces=("source-workspace",))
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle), "--into", "stable-workspace"])

    cli.main()

    call = stub_reingest["ingest"][0]
    assert call["workspace_map"] == {"source-workspace": "stable-workspace"}
    assert call["require_empty"] is True


def test_restore_into_rejects_invalid_target_before_catalog_or_ingest(
    monkeypatch,
    tmp_path,
    stub_reingest,
) -> None:
    bundle = _make_export_bundle(tmp_path / "state.tar.zst", workspaces=("source-workspace",))
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle), "--into", "INVALID!"])

    with pytest.raises(SystemExit, match="platform naming rule"):
        cli.main()

    assert stub_reingest["platform_root"] == []
    assert stub_reingest["catalog_root"] == []
    assert stub_reingest["ingest"] == []


def test_restore_into_help_states_fresh_target_contract(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "fresh, empty target" in output
    assert "not idempotent" in output


def test_restore_export_bundle_reingests_with_digest_suffix(monkeypatch, tmp_path, capsys, stub_reingest):
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "b.tar.zst")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()[:8]
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    cli.main()
    ingest = stub_reingest["ingest"][0]
    assert ingest["base_url"] == "http://localhost:8080"  # default target
    assert ingest["workspace_map"] == {
        "tau2-airline": f"tau2-airline-{digest}",
        "tau2-airline-oracle": f"tau2-airline-oracle-{digest}",
    }
    assert ingest["catalog"] == "stub-catalog"
    out = capsys.readouterr().out
    assert "2 spans ingested, 0 skipped" in out
    assert f"tau2-airline-{digest}" in out


def test_restore_state_ref_downloads(monkeypatch, tmp_path, stub_reingest):
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v6.tar.zst", workspaces=("nvq",))
    downloads: list = []
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: downloads.append(ref) or bundle)
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", "--state", "state-v6", "--base", "http://ci-host:8080"])
    cli.main()
    assert downloads == ["state-v6"]
    ingest = stub_reingest["ingest"][0]
    assert ingest["workspace_map"] == {"nvq": "nvq-state-v6"}
    assert ingest["base_url"] == "http://ci-host:8080"


def test_restore_state_only_takes_refs_not_files(monkeypatch, tmp_path, stub_reingest):
    """restore --state is refs-only; local bundle files go through the positional FILE."""
    bundle = _make_export_bundle(tmp_path / "b.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", "--state", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "state-v<N>" in str(exc.value)
    assert stub_reingest["ingest"] == []


def test_restore_platform_root_flag_reaches_catalog(monkeypatch, tmp_path, stub_reingest):
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "b.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle), "--platform-root", "/opt/nemo-helix"])
    cli.main()
    assert stub_reingest["platform_root"] == ["/opt/nemo-helix"]
    assert stub_reingest["catalog_root"] == [Path("/opt/nemo-helix")]


def test_restore_seeds_run_records_only(monkeypatch, tmp_path, stub_reingest):
    """Only the bundle's run records land in TMP: insights no longer travel in bundles, so a
    stray insights YAML inside an old bundle is ignored and local ones stay untouched."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    (tmp_path / "insights_nvq.yaml").write_text("local", encoding="utf-8")  # the restored subject's own prior
    (tmp_path / "insights_stale.yaml").write_text("stale", encoding="utf-8")  # unrelated local subject
    bundle = _make_export_bundle(
        tmp_path / "b.tar.zst",
        workspaces=("nvq",),
        records={"nvq.run.json": json.dumps(_RUN_RECORD), "insights_nvq.yaml": "insights: []"},
    )
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    cli.main()
    assert (tmp_path / "nvq.run.json").exists()  # record copied out of the bundle
    assert (tmp_path / "insights_nvq.yaml").read_text() == "local"  # the bundle's stray YAML is ignored
    assert (tmp_path / "insights_stale.yaml").read_text() == "stale"  # unrelated local file survives
    assert list(tmp_path.glob("backup-*")) == []  # nothing clobbered, nothing moved


def test_restore_legacy_bundle_exits_with_migration_pointer(monkeypatch, tmp_path, stub_reingest):
    """The destructive tar path is gone; legacy bundles get a clear rejection, not a swap."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_legacy_bundle(tmp_path / "legacy.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "legacy tar bundle (state-v1..v5)" in message
    assert "pre-migration checkout" in message
    assert "README" in message
    assert stub_reingest["ingest"] == []  # never re-ingested


def test_restore_missing_bundle_file_exits(monkeypatch, tmp_path, stub_reingest):
    """A missing file must be a hard exit, not a misleading legacy-bundle rejection."""
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(tmp_path / "nope.tar.zst")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "no such bundle" in str(exc.value)


def test_restore_corrupt_bundle_exits_with_read_error(monkeypatch, tmp_path, stub_reingest):
    """A garbage file must surface tar's own error, not the misleading legacy-bundle message."""
    bad = tmp_path / "bad.tar.zst"
    bad.write_bytes(b"garbage, definitely not a zstd tar")
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bad)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "could not read bundle bad.tar.zst" in message
    assert "corrupt or truncated" in message
    assert "tar" in message  # carries tar's stderr (the actual reason)
    assert "legacy" not in message
    assert stub_reingest["ingest"] == []


def test_restore_truncated_bundle_not_misdiagnosed_as_legacy(monkeypatch, tmp_path, stub_reingest):
    good = _make_export_bundle(tmp_path / "good.tar.zst", workspaces=("nvq",))
    trunc = tmp_path / "trunc.tar.zst"
    trunc.write_bytes(good.read_bytes()[:40])
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(trunc)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "could not read bundle trunc.tar.zst" in message and "corrupt or truncated" in message
    assert "legacy" not in message


def test_restore_tar_without_manifest_is_not_an_evaluation_bundle(monkeypatch, tmp_path, stub_reingest):
    bundle = _make_tar_with_manifest_text(tmp_path / "plain.tar.zst", None)
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "not an evaluation bundle" in message
    assert "legacy" not in message and "corrupt" not in message


def test_restore_unparseable_manifest_is_not_an_evaluation_bundle(monkeypatch, tmp_path, stub_reingest):
    bundle = _make_tar_with_manifest_text(tmp_path / "badjson.tar.zst", "{not json")
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "not an evaluation bundle" in message
    assert "legacy" not in message


def test_restore_non_object_manifest_is_not_an_evaluation_bundle(monkeypatch, tmp_path, stub_reingest):
    bundle = _make_tar_with_manifest_text(tmp_path / "array.tar.zst", "[]")
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "not an evaluation bundle" in str(exc.value)


def test_restore_requires_exactly_one_source(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "exactly one" in str(exc.value)
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", str(tmp_path / "b.tar.zst"), "--state", "state-v6"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "exactly one" in str(exc.value)


def test_restore_empty_file_arg_exits(monkeypatch):
    """`restore ""` satisfies the arity check but names nothing — it must exit clearly,
    not fall through to ref resolution's subject-less lock lookup."""
    monkeypatch.setattr(sys, "argv", ["evaluation", "restore", ""])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert str(exc.value) == "restore: FILE must be a non-empty path"


def _fake_snapshot_export(seen):
    def fake(subjects, out, tmp_dir, *, since):
        seen.update(
            subjects=[s.name for s in subjects],
            base_urls=[s.config.get("base_url") for s in subjects],
            out=out,
            tmp_dir=tmp_dir,
            since=since,
        )
        return out

    return fake


def test_snapshot_command_exports_subject(monkeypatch, tmp_path, capsys):
    seen: dict = {}
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr("evaluation.artifact.snapshot_export", _fake_snapshot_export(seen))
    monkeypatch.setattr(sys, "argv", ["evaluation", "snapshot", "tau2-airline", "-o", str(tmp_path / "x.tar.zst")])
    cli.main()
    assert "x.tar.zst" in capsys.readouterr().out
    assert seen["subjects"] == ["tau2-airline"]
    assert seen["since"] is None


def test_snapshot_since_flag(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr("evaluation.artifact.snapshot_export", _fake_snapshot_export(seen))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "snapshot", "nvq", "--since", "2026-07-01", "-o", str(tmp_path / "x.tar.zst")],
    )
    cli.main()
    assert seen["since"] == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_snapshot_base_overrides_source_url(monkeypatch, tmp_path):
    """--base replaces every listed subject's stanza URL, so disagreeing stanzas snapshot together."""
    seen: dict = {}
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr("evaluation.artifact.snapshot_export", _fake_snapshot_export(seen))
    # nvq (freeplay) and tau2-retail (localhost) disagree on base_url in evaluations.toml
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "snapshot",
            "nvq",
            "tau2-retail",
            "--base",
            "http://ci-host:8080",
            "-o",
            str(tmp_path / "x.tar.zst"),
        ],
    )
    cli.main()
    assert seen["base_urls"] == ["http://ci-host:8080", "http://ci-host:8080"]


def test_snapshot_glamr_base_preserves_remote_auth(monkeypatch, tmp_path):
    from evaluation.registry import Subject

    glamr = Subject(
        "glamr",
        "intake",
        {
            "agent": "glamr",
            "workspace": "default",
            "base_url": "https://original.example",
            "auth": "basic",
            "intake_path_prefix": "/glamr/intake",
            "auth_user_env": "GLAMR_INTAKE_USER",
            "auth_password_env": "GLAMR_INTAKE_PASSWORD",
        },
    )
    monkeypatch.setattr(cli, "load_registry", lambda _path: {"glamr": glamr})
    monkeypatch.setattr(cli, "TMP", tmp_path)
    captured: dict[str, object] = {}

    def fake_snapshot_export(
        subjects: list[Subject],
        out: Path,
        tmp_dir: Path,
        *,
        since: datetime | None,
    ) -> Path:
        (subject,) = subjects
        captured.update(subject.config)
        return out

    monkeypatch.setattr(cli.artifact, "snapshot_export", fake_snapshot_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "snapshot",
            "glamr",
            "--base",
            "https://override.example",
            "-o",
            str(tmp_path / "glamr.tar.zst"),
        ],
    )

    cli.main()

    assert captured["base_url"] == "https://override.example"
    assert {key: captured[key] for key in cli._REMOTE_AUTH_KEYS} == {
        "auth": "basic",
        "intake_path_prefix": "/glamr/intake",
        "auth_user_env": "GLAMR_INTAKE_USER",
        "auth_password_env": "GLAMR_INTAKE_PASSWORD",
    }


def test_snapshot_subjects_json_for_ci(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr("evaluation.artifact.snapshot_export", _fake_snapshot_export(seen))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "snapshot", "--subjects-json", '["tau2-airline", "nvq"]', "-o", str(tmp_path / "x.tar.zst")],
    )
    cli.main()
    assert seen["subjects"] == ["tau2-airline", "nvq"]


def test_snapshot_garbage_since_exits_before_any_export(monkeypatch):
    """Injection-shaped --since garbage must die at parse time, before any network I/O."""
    called: list = []
    monkeypatch.setattr("evaluation.artifact.snapshot_export", lambda *a, **k: called.append("export"))
    monkeypatch.setattr("evaluation.export.make_client", lambda *a, **k: called.append("client"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "snapshot", "tau2-airline", "--since", "'); DROP TABLE spans;--"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--since" in str(exc.value)
    assert called == []


def test_snapshot_without_subjects_exits(monkeypatch):
    monkeypatch.setattr("evaluation.artifact.snapshot_export", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "argv", ["evaluation", "snapshot"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "subject" in str(exc.value)


def test_snapshot_unknown_subject_exits(monkeypatch):
    monkeypatch.setattr("evaluation.artifact.snapshot_export", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "argv", ["evaluation", "snapshot", "nope"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "nope" in str(exc.value)


def test_snapshot_bad_subjects_json_exits(monkeypatch):
    monkeypatch.setattr("evaluation.artifact.snapshot_export", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "argv", ["evaluation", "snapshot", "--subjects-json", "not-json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--subjects-json" in str(exc.value)


def test_analyze_live_summary_md_appends_markdown_with_provenance(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        out_path.write_text("insights:\n- {id: i1, title: T1, status: proposed, description: D, trace_refs: [r1]}\n")
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    summary = tmp_path / "summary.md"
    summary.write_text("PREVIOUS\n")
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--summary-md", str(summary)])
    cli.main()
    text = summary.read_text()
    assert text.startswith("PREVIOUS\n")  # appended, not overwritten
    assert "### analyze @ live (nvq)" in text  # live provenance, ahead of the insights
    assert text.index("analyze @ live") < text.index("## Insights")
    assert "## Insights — nvq (1)" in text
    assert "**T1**" in text


# --------------------------------------------------------------------------- #
# analyze with a state: restore-then-live-analyze against fixture workspaces
# --------------------------------------------------------------------------- #


def test_analyze_state_remaps_workspaces_and_floors_since(monkeypatch, tmp_path, stub_reingest):
    """The analyst must read the fixture workspaces with an explicit manifest-derived since."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        workspaces=("tau2-airline", "tau2-airline-oracle"),
        min_start_time="2026-06-01T12:00:00+00:00",
        records={"tau2-airline.run.json": json.dumps(_RUN_RECORD)},
    )
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["record"] = record
        seen["since"] = since
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline", "--state", "state-v6"])
    cli.main()
    ingest = stub_reingest["ingest"][0]
    assert ingest["base_url"] == "http://localhost:8080"  # default target: the local platform
    assert ingest["workspace_map"] == {
        "tau2-airline": "tau2-airline-state-v6",
        "tau2-airline-oracle": "tau2-airline-oracle-state-v6",
    }
    # the run record the analyst consumes is remapped onto the fixtures + target platform
    assert seen["record"]["realistic_workspace"] == "tau2-airline-state-v6"
    assert seen["record"]["oracle_workspace"] == "tau2-airline-oracle-state-v6"
    assert seen["record"]["base_url"] == "http://localhost:8080"
    assert seen["record"]["experiment_id"] == "run-1"  # evaluation_id scoping untouched
    # explicit since = manifest min_start_time floored by one day (defeats the 30-day read default)
    assert seen["since"] == datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)


def test_analyze_intake_subject_workspace_remapped(monkeypatch, tmp_path, stub_reingest):
    """Intake adapters read the workspace from subject config — the remap must land there."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v6.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["workspace"] = self.subject.config["workspace"]
        seen["base_url"] = self.subject.config["base_url"]
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", "state-v6"])
    cli.main()
    assert seen["workspace"] == "nvq-state-v6"
    assert seen["base_url"] == "http://localhost:8080"


def test_analyze_legacy_bundle_exits_with_migration_pointer(monkeypatch, tmp_path, stub_reingest):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_legacy_bundle(tmp_path / "state-v4.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "legacy tar bundle (state-v1..v5)" in message
    assert "pre-migration checkout" in message
    assert "README" in message
    assert stub_reingest["ingest"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "nvq", "--seed", "none"],
        ["analyze", "nvq", "--seed", "keep"],
        ["analyze", "nvq", "--state", "state-v6", "--seed", "keep"],
        ["restore", "b.tar.zst", "--seed", "none"],
        ["restore", "--state", "state-v6", "--seed", "keep"],
        ["analyze", "nvq", "--keep-insights"],  # old flag name
    ],
)
def test_seed_flag_removed(monkeypatch, argv):
    """--seed died with in-bundle insights; --update-insights is the only prior-seeding surface."""
    monkeypatch.setattr(sys, "argv", ["evaluation", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2  # argparse usage error


def test_analyze_fresh_default_moves_local_insights_to_backup(monkeypatch, tmp_path, capsys, stub_reingest):
    """Bare analyze is fresh: a prior local insights YAML moves aside (never deleted) before the analyst."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    (tmp_path / "insights_tau2-airline.yaml").write_text("prior", encoding="utf-8")
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        records={"tau2-airline.run.json": json.dumps(_RUN_RECORD)},
    )
    monkeypatch.setattr("evaluation.release.resolve_state", lambda state, *, subject: "state-v6")
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        assert not (tmp_path / "insights_tau2-airline.yaml").exists(), "prior must move BEFORE the analyst runs"
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline"])
    cli.main()
    assert not (tmp_path / "insights_tau2-airline.yaml").exists()
    (backup,) = tmp_path.glob("backup-*")
    assert (backup / "insights_tau2-airline.yaml").read_text() == "prior"  # moved, not destroyed
    out = capsys.readouterr().out
    assert f"fresh insights: moved prior insights_tau2-airline.yaml to {backup.name}/" in out
    assert "use --update-insights to seed the analyst with priors" in out


def test_analyze_update_insights_leaves_file(monkeypatch, tmp_path, capsys, stub_reingest):
    """--update-insights leaves the local prior in the analyst's path, in every analyze mode."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    (tmp_path / "insights_nvq.yaml").write_text("prior", encoding="utf-8")
    bundle = _make_export_bundle(tmp_path / "local.tar.zst", workspaces=("nvq",))

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(bundle), "--update-insights"])
    cli.main()
    assert (tmp_path / "insights_nvq.yaml").read_text() == "prior"  # not-fresh flow preserved
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--update-insights"])
    cli.main()
    assert (tmp_path / "insights_nvq.yaml").read_text() == "prior"  # valid with --live too
    assert list(tmp_path.glob("backup-*")) == []
    assert "fresh insights" not in capsys.readouterr().out


def test_analyze_live_fresh_also_applies(monkeypatch, tmp_path, capsys):
    """--live is no less fresh than pinned/state mode: the local prior moves aside there too."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    (tmp_path / "insights_nvq.yaml").write_text("prior", encoding="utf-8")

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live"])
    cli.main()
    assert not (tmp_path / "insights_nvq.yaml").exists()
    (backup,) = tmp_path.glob("backup-*")
    assert (backup / "insights_nvq.yaml").read_text() == "prior"
    # microsecond-stamped: two invocations in the same second must not share a dir
    assert re.fullmatch(r"backup-\d{8}-\d{6}-\d{6}", backup.name)
    assert "fresh insights: moved prior insights_nvq.yaml" in capsys.readouterr().out


def test_update_insights_warns_when_priors_live_under_other_workspace(monkeypatch, tmp_path, capsys):
    """--update-insights priors are scoped by 'workspace' inside the YAML: when none of the
    records match the workspace this analysis reads, warn loudly — and never rewrite."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    prior = yaml.safe_dump({"insights": [{"id": "i1", "workspace": "nvq-state-v6"}]})
    (tmp_path / "insights_nvq.yaml").write_text(prior, encoding="utf-8")

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    # live intake analysis reads the stanza workspace 'nvq'; the prior carries only the fixture ws
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--update-insights"])
    cli.main()
    out = capsys.readouterr().out
    assert (
        "warning: --update-insights: insights_nvq.yaml carries workspace(s) nvq-state-v6 "
        "but this analysis runs under 'nvq' — priors will be invisible and updates will be skipped"
    ) in out
    assert (tmp_path / "insights_nvq.yaml").read_text(encoding="utf-8") == prior  # never rewritten


def test_update_insights_no_warning_when_workspace_matches(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    prior = yaml.safe_dump({"insights": [{"id": "i1", "workspace": "nvq"}]})
    (tmp_path / "insights_nvq.yaml").write_text(prior, encoding="utf-8")

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--live", "--update-insights"])
    cli.main()
    assert "warning: --update-insights" not in capsys.readouterr().out


def test_update_insights_state_mode_warns_against_fixture_workspace(monkeypatch, tmp_path, capsys, stub_reingest):
    """The most likely real trip-up: priors kept from a LIVE run, then a pinned/state
    analyze — the analysis runs under the remapped fixture workspace, so the check must
    compare against it (post-remap), not the stanza/record original."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    prior = yaml.safe_dump({"insights": [{"id": "i1", "workspace": "tau2-airline"}]})  # live-run prior
    (tmp_path / "insights_tau2-airline.yaml").write_text(prior, encoding="utf-8")
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        records={"tau2-airline.run.json": json.dumps(_RUN_RECORD)},
    )
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "analyze", "tau2-airline", "--state", "state-v6", "--update-insights"],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "warning: --update-insights: insights_tau2-airline.yaml carries workspace(s) tau2-airline" in out
    assert "runs under 'tau2-airline-state-v6'" in out


def _ticking_clock(monkeypatch, *modules):
    """Replace each module's `datetime` with one whose now() advances 2 minutes per call.

    Deterministically models a slow restore: any code that stamps backup dirs
    per-call (instead of once per invocation) produces divergent stamps.
    """
    import datetime as real

    calls = {"n": 0}

    class TickingDateTime(real.datetime):
        @classmethod
        def now(cls, tz=None):
            calls["n"] += 1
            return real.datetime(2026, 7, 6, 12, 0, 0, tzinfo=tz) + real.timedelta(minutes=2 * calls["n"])

    fake = types.SimpleNamespace(datetime=TickingDateTime, timezone=real.timezone, timedelta=real.timedelta)
    for module in modules:
        monkeypatch.setattr(module, "datetime", fake)


def test_analyze_single_backup_dir_per_invocation(monkeypatch, tmp_path, stub_reingest):
    """One analyze = at most one tmp/backup-<stamp>/: the restore's clobber-backup and the
    fresh-insights move share one destination even when the restore takes wall-clock time."""
    from evaluation import artifact
    from evaluation.runstore import load_run, save_run

    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    _ticking_clock(monkeypatch, cli, artifact)
    save_run(tmp_path / "tau2-airline.run.json", {**_RUN_RECORD, "experiment_id": "stale"})  # will be clobbered
    (tmp_path / "insights_tau2-airline.yaml").write_text("prior", encoding="utf-8")  # will move (fresh default)
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        records={"tau2-airline.run.json": json.dumps(_RUN_RECORD)},
    )
    monkeypatch.setattr("evaluation.release.resolve_state", lambda state, *, subject: "state-v6")
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline"])
    cli.main()
    (backup,) = tmp_path.glob("backup-*")  # exactly one dir carries both parked files
    assert json.loads((backup / "tau2-airline.run.json").read_text(encoding="utf-8"))["experiment_id"] == "stale"
    assert (backup / "insights_tau2-airline.yaml").read_text(encoding="utf-8") == "prior"
    restored_run = load_run(tmp_path / "tau2-airline.run.json")
    assert restored_run is not None
    assert restored_run["experiment_id"] == "run-1"  # bundle record seeded


def test_analyze_bundle_without_run_record_exits(monkeypatch, tmp_path, stub_reingest):
    """A stale local run record must not silently stand in for a bundle that carries none —
    its experiment_id would filter every restored trace to zero (empty 'analysis')."""
    from evaluation.runstore import save_run

    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    save_run(tmp_path / "tau2-airline.run.json", {**_RUN_RECORD, "experiment_id": "stale-run"})
    bundle = _make_export_bundle(tmp_path / "dl" / "state-v6.tar.zst")  # no tmp/ records at all
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    called: list = []

    async def fake_analyze(self, *, record, since, verbose, out_path):
        called.append(record)
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline", "--state", "state-v6"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "no run record" in message and "tau2-airline" in message and "state-v6" in message
    assert called == []  # the analyst never ran against the stale record


def test_analyze_intake_bundle_without_run_record_still_works(monkeypatch, tmp_path, stub_reingest):
    """Intake subjects don't need a run record — a record-less bundle must keep analyzing."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "local.tar.zst", workspaces=("nvq",))

    async def fake_analyze(self, *, record, since, verbose, out_path):
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.IntakeAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(bundle)])
    cli.main()  # must not raise


def test_analyze_subject_not_in_bundle_exits(monkeypatch, tmp_path, stub_reingest):
    """Analyzing tau2-retail from an airline-only bundle must die before restoring anything."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bundle = _make_export_bundle(tmp_path / "airline.tar.zst")  # tau2-airline (+oracle) only
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-retail", "--state", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "workspace 'tau2-retail'" in message  # the one workspace analysis reads
    assert "tau2-airline" in message  # what the bundle actually carries
    assert stub_reingest["ingest"] == []  # membership check fires before any restore


def test_analyze_benchmark_oracle_twin_not_required(monkeypatch, tmp_path, stub_reingest):
    """Membership requires only the workspace analysis reads: a realistic-only bundle
    (no -oracle twin) still analyzes a benchmark subject."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    record = {**_RUN_RECORD, "oracle_workspace": None}
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        workspaces=("tau2-airline",),  # no oracle twin in the bundle
        records={"tau2-airline.run.json": json.dumps(record)},
    )
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    seen: dict = {}

    async def fake_analyze(self, *, record, since, verbose, out_path):
        seen["record"] = record
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline", "--state", "state-v6"])
    cli.main()  # must not raise on the absent oracle twin
    assert seen["record"]["realistic_workspace"] == "tau2-airline-state-v6"


def test_analyze_record_workspace_not_in_restored_set_exits(monkeypatch, tmp_path, stub_reingest):
    """A bundle record pointing at a workspace the bundle doesn't restore is a minting
    bug — hard exit, or the analyst would read an empty (or wrong) fixture."""
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    record = {**_RUN_RECORD, "realistic_workspace": "tau2-airline-b", "oracle_workspace": None}
    bundle = _make_export_bundle(
        tmp_path / "dl" / "state-v6.tar.zst",
        workspaces=("tau2-airline", "tau2-airline-oracle"),
        records={"tau2-airline.run.json": json.dumps(record)},
    )
    monkeypatch.setattr("evaluation.release.download_ref", lambda ref, dest: bundle)
    called: list = []

    async def fake_analyze(self, *, record, since, verbose, out_path):
        called.append(record)
        return "REPORT-OK"

    monkeypatch.setattr("evaluation.adapters.BenchmarkAdapter.analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "tau2-airline", "--state", "state-v6"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "bundle record for 'tau2-airline'" in message
    assert "'tau2-airline-b'" in message
    assert "tau2-airline-state-v6" in message  # names the restored workspaces
    assert "mismatched record" in message
    assert called == []  # the analyst never ran against the mismatched record


def test_analyze_since_with_pinned_exits(monkeypatch, tmp_path, stub_reingest):
    """--since is live-only; pinned/state analysis derives its lower bound from the manifest."""
    monkeypatch.setattr(cli, "TMP", tmp_path)
    monkeypatch.setattr(
        "evaluation.release.download_ref",
        lambda *a, **k: pytest.fail("must exit before any resolve/download"),
    )
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--since", "7d"])  # bare = pinned mode
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--since only applies to --live" in str(exc.value)

    bundle = _make_export_bundle(tmp_path / "b.tar.zst", workspaces=("nvq",))
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(bundle), "--since", "7d"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--since only applies to --live" in str(exc.value)
    assert stub_reingest["ingest"] == []  # rejected before any restore


def test_analyze_corrupt_bundle_exits_with_read_error(monkeypatch, tmp_path, stub_reingest):
    monkeypatch.setenv("INFERENCE_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "TMP", tmp_path)
    bad = tmp_path / "bad.tar.zst"
    bad.write_bytes(b"garbage, definitely not a zstd tar")
    monkeypatch.setattr(sys, "argv", ["evaluation", "analyze", "nvq", "--state", str(bad)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "could not read bundle bad.tar.zst" in message and "corrupt or truncated" in message
    assert "legacy" not in message
    assert stub_reingest["ingest"] == []


# --------------------------------------------------------------------------- #
# roundtrip: the mint-time fidelity guard as a CLI hook
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_roundtrip(monkeypatch):
    """Record roundtrip_diff calls; each test sets the mismatch list it returns."""
    calls: list[dict] = []
    result: dict = {"mismatches": []}

    def fake_diff(base_url, export_dir, manifest, *, scratch_prefix, catalog, **kw):
        calls.append(
            {
                "base_url": base_url,
                "export_dir": Path(export_dir),
                "manifest": manifest,
                "scratch_prefix": scratch_prefix,
                "catalog": catalog,
            }
        )
        return list(result["mismatches"])

    monkeypatch.setattr("evaluation.reingest.roundtrip_diff", fake_diff)
    return {"calls": calls, "result": result}


def test_roundtrip_pass_prints_one_liner(monkeypatch, tmp_path, capsys, stub_reingest, stub_roundtrip):
    bundle = _make_export_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "roundtrip", str(bundle)])
    cli.main()
    call = stub_roundtrip["calls"][0]
    assert call["base_url"] == "http://localhost:8080"  # default target
    assert call["scratch_prefix"] == "scratch-rt-"
    assert call["catalog"] == "stub-catalog"
    assert call["export_dir"].name == "export"  # the extracted bundle's export/ dir
    assert call["manifest"]["kind"] == "evaluation-export"
    out = capsys.readouterr().out
    assert "✓" in out and "candidate.tar.zst" in out


def test_roundtrip_mismatches_print_and_exit_nonzero(monkeypatch, tmp_path, capsys, stub_reingest, stub_roundtrip):
    stub_roundtrip["result"]["mismatches"] = ["ws/spans s1.name: 'a' != 'b'", "ws/spans: 2 exported vs 1 restored"]
    bundle = _make_export_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "roundtrip", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0
    assert "2 mismatch" in str(exc.value)
    assert "ws/spans s1.name" in capsys.readouterr().out


def test_roundtrip_base_and_platform_root_flags(monkeypatch, tmp_path, stub_reingest, stub_roundtrip):
    bundle = _make_export_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "roundtrip",
            str(bundle),
            "--base",
            "http://ci-host:8080",
            "--platform-root",
            "/opt/nemo-helix",
        ],
    )
    cli.main()
    assert stub_roundtrip["calls"][0]["base_url"] == "http://ci-host:8080"
    assert stub_reingest["platform_root"] == ["/opt/nemo-helix"]
    assert stub_reingest["catalog_root"] == [Path("/opt/nemo-helix")]


def test_roundtrip_legacy_bundle_exits(monkeypatch, tmp_path, stub_reingest, stub_roundtrip):
    bundle = _make_legacy_bundle(tmp_path / "legacy.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "roundtrip", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "evaluation export bundle" in str(exc.value)
    assert stub_roundtrip["calls"] == []  # guard never ran


def test_roundtrip_corrupt_bundle_exits_with_read_error(monkeypatch, tmp_path, stub_roundtrip):
    bad = tmp_path / "bad.tar.zst"
    bad.write_bytes(b"garbage, definitely not a zstd tar")
    monkeypatch.setattr(sys, "argv", ["evaluation", "roundtrip", str(bad)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    message = str(exc.value)
    assert "could not read bundle bad.tar.zst" in message and "corrupt or truncated" in message
    assert stub_roundtrip["calls"] == []


def test_roundtrip_missing_bundle_exits(monkeypatch, tmp_path, stub_roundtrip):
    monkeypatch.setattr(sys, "argv", ["evaluation", "roundtrip", str(tmp_path / "nope.tar.zst")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "no such bundle" in str(exc.value)
    assert stub_roundtrip["calls"] == []


def test_publish_platform_root_threaded(monkeypatch, tmp_path, stub_reingest, stub_roundtrip):
    """publish --base runs the round-trip guard with --platform-root, not a hardcoded None."""
    published: list = []
    monkeypatch.setattr("evaluation.publish.publish", lambda path, *, reason: published.append(path))
    bundle = _make_export_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "publish",
            str(bundle),
            "--base",
            "http://ci-host:8080",
            "--platform-root",
            "/opt/nemo-helix",
        ],
    )
    cli.main()
    assert stub_reingest["platform_root"] == ["/opt/nemo-helix"]
    assert stub_reingest["catalog_root"] == [Path("/opt/nemo-helix")]
    assert stub_roundtrip["calls"][0]["base_url"] == "http://ci-host:8080"
    assert published == [bundle]
