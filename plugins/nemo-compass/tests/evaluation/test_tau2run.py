# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
from pathlib import Path

import pytest
from evaluation.tau2run import (
    build_argv,
    load_simulations,
    policy_version,
    read_policy,
    resolve_paths,
    run_tau2,
)

CFG = {
    "domain": "airline",
    "agent_llm": "openai/m",
    "user_llm": "openai/m",
    "task_split_name": "test",
    "num_trials": 1,
    "seed": 300,
    "max_concurrency": 4,
}


def test_build_argv_core_flags():
    argv = build_argv(CFG, "run1")
    assert argv == [
        "tau2",
        "run",
        "--domain",
        "airline",
        "--agent-llm",
        "openai/m",
        "--user-llm",
        "openai/m",
        "--task-split-name",
        "test",
        "--num-trials",
        "1",
        "--seed",
        "300",
        "--max-concurrency",
        "4",
        "--save-to",
        "run1",
    ]


def test_build_argv_optional_flags_and_bin():
    argv = build_argv({**CFG, "num_tasks": 5, "timeout": 600}, "run1", tau2_bin="/x/tau2")
    assert argv[0] == "/x/tau2"
    assert argv[-4:] == ["--num-tasks", "5", "--timeout", "600"]


def test_run_tau2_sets_data_dir_and_loads(tmp_path):
    run_dir = tmp_path / "simulations" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps({"simulations": [{"task_id": "0"}]}))
    recorded = {}

    def fake_runner(argv, *, env, check):
        recorded["argv"] = argv
        recorded["env"] = env
        return subprocess.CompletedProcess(argv, 0)

    sims = run_tau2(CFG, "run1", data_dir=tmp_path, tau2_bin="tau2", runner=fake_runner)
    assert sims == [{"task_id": "0"}]
    assert recorded["env"]["TAU2_DATA_DIR"] == str(tmp_path)
    assert recorded["argv"][:2] == ["tau2", "run"]


def test_run_tau2_raises_on_nonzero(tmp_path):
    def fake_runner(argv, *, env, check):
        return subprocess.CompletedProcess(argv, 1)

    with pytest.raises(RuntimeError, match="exit 1"):
        run_tau2(CFG, "run1", data_dir=tmp_path, tau2_bin="tau2", runner=fake_runner)


def _write_results(tmp_path: Path, payload) -> Path:
    run_dir = tmp_path / "simulations" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(payload))
    return tmp_path


def test_load_simulations_simulations_wrapper(tmp_path):
    assert load_simulations(_write_results(tmp_path, {"simulations": [{"a": 1}]}), "r") == [{"a": 1}]


def test_load_simulations_results_wrapper(tmp_path):
    assert load_simulations(_write_results(tmp_path, {"results": [{"b": 2}]}), "r") == [{"b": 2}]


def test_load_simulations_bare_list(tmp_path):
    assert load_simulations(_write_results(tmp_path, [{"c": 3}]), "r") == [{"c": 3}]


def test_load_simulations_empty_simulations_not_overridden(tmp_path):
    payload = {"simulations": [], "results": [{"x": 1}]}
    assert load_simulations(_write_results(tmp_path, payload), "r") == []


def test_load_simulations_finds_tau2_nested(tmp_path):
    # vanilla tau2-bench writes some layouts under data/tau2/...
    run_dir = tmp_path / "tau2" / "simulations" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps({"simulations": [{"n": 1}]}))
    assert load_simulations(tmp_path, "r") == [{"n": 1}]


def test_load_simulations_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no tau2 results"):
        load_simulations(tmp_path, "nope")


def test_read_policy_flat_and_nested_and_absent(tmp_path):
    # flat layout: <data>/domains/<domain>/policy.md
    flat = tmp_path / "domains" / "airline"
    flat.mkdir(parents=True)
    (flat / "policy.md").write_text("FLAT POLICY")
    # vanilla tau2-bench layout: <data>/tau2/domains/<domain>/policy.md
    nested = tmp_path / "tau2" / "domains" / "retail"
    nested.mkdir(parents=True)
    (nested / "policy.md").write_text("NESTED POLICY")
    assert read_policy(tmp_path, "airline") == "FLAT POLICY"
    assert read_policy(tmp_path, "retail") == "NESTED POLICY"
    assert read_policy(tmp_path, "telecom") is None


def test_read_policy_main_policy_fallback_flat_and_nested(tmp_path):
    """Telecom uses main_policy.md; policy.md wins when both exist."""
    flat = tmp_path / "domains" / "telecom"
    flat.mkdir(parents=True)
    (flat / "main_policy.md").write_text("TELECOM FLAT MAIN")
    nested = tmp_path / "tau2" / "domains" / "telecom-nested"
    nested.mkdir(parents=True)
    (nested / "main_policy.md").write_text("TELECOM NESTED MAIN")
    both = tmp_path / "domains" / "telecom-both"
    both.mkdir(parents=True)
    (both / "policy.md").write_text("PRIMARY")
    (both / "main_policy.md").write_text("FALLBACK")
    assert read_policy(tmp_path, "telecom") == "TELECOM FLAT MAIN"
    assert read_policy(tmp_path, "telecom-nested") == "TELECOM NESTED MAIN"
    assert read_policy(tmp_path, "telecom-both") == "PRIMARY"


def test_read_policy_nested_layout_precedes_flat_filename_priority(tmp_path):
    nested = tmp_path / "tau2" / "domains" / "telecom"
    nested.mkdir(parents=True)
    (nested / "main_policy.md").write_text("NESTED MAIN")
    flat = tmp_path / "domains" / "telecom"
    flat.mkdir(parents=True)
    (flat / "policy.md").write_text("FLAT POLICY")

    assert read_policy(tmp_path, "telecom") == "NESTED MAIN"


def test_resolve_paths_from_repo_absolute():
    tau2_bin, data_dir = resolve_paths({"tau2_repo": "/r"}, repo_root=Path("/root"))
    assert tau2_bin == str(Path("/r") / ".venv" / "bin" / "tau2")
    assert data_dir == Path("/r/data")


def test_resolve_paths_relative_anchored_to_repo_root():
    root = Path("/home/x/proj")
    tau2_bin, data_dir = resolve_paths({"tau2_repo": "../tau2-bench"}, repo_root=root)
    assert data_dir == root / "../tau2-bench" / "data"
    assert tau2_bin == str(root / "../tau2-bench" / ".venv" / "bin" / "tau2")


def test_resolve_paths_explicit_overrides_win():
    tau2_bin, data_dir = resolve_paths(
        {"tau2_repo": "/r", "tau2_bin": "/x/tau2", "tau2_data_dir": "/d"},
        repo_root=Path("/root"),
    )
    assert tau2_bin == "/x/tau2"
    assert data_dir == Path("/d")


def test_resolve_paths_unset_is_none():
    tau2_bin, data_dir = resolve_paths({}, repo_root=Path("/root"))
    assert tau2_bin is None
    assert data_dir is None


def test_policy_version():
    v = policy_version("POLICY TEXT")
    assert len(v) == 12 and all(c in "0123456789abcdef" for c in v)
    assert policy_version(None) == "unknown"
    assert policy_version("") == "unknown"


def test_load_tasks_reads_nested_and_missing(tmp_path):
    from evaluation.tau2run import load_tasks

    dom = tmp_path / "tau2" / "domains" / "airline"
    dom.mkdir(parents=True)
    (dom / "tasks.json").write_text(json.dumps([{"id": "8", "description": {"purpose": "p"}}, {"id": 9}]))
    tasks = load_tasks(tmp_path, "airline")
    assert set(tasks) == {"8", "9"}
    assert tasks["8"]["description"]["purpose"] == "p"
    assert load_tasks(tmp_path, "retail") == {}  # absent -> empty
