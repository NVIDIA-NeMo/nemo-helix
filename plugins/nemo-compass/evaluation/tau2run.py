# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the tau2 benchmark as a subprocess and load its simulation output."""

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path


def resolve_paths(cfg: Mapping[str, object], *, repo_root: Path) -> tuple[str | None, Path | None]:
    """Resolve the tau2 binary and data dir from the stanza config.

    ``tau2_repo`` (absolute, or relative to the nemo-compass plugin repo root — the
    sibling-clone convention, e.g. ``../tau2-bench``) yields both the CLI
    (``<repo>/.venv/bin/tau2``, where ``uv sync`` installs it) and the data dir
    (``<repo>/data``). The explicit ``tau2_bin``/``tau2_data_dir`` keys override
    the derived values for non-standard installs (``tau2_data_dir`` is anchored
    to the repo root when relative; ``tau2_bin`` is taken verbatim so a bare
    name still resolves on ``PATH``). Either return value is ``None`` when not
    configured.
    """

    def _anchor(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else repo_root / p

    repo_path = _anchor(str(cfg["tau2_repo"])) if cfg.get("tau2_repo") else None

    bin_override = cfg.get("tau2_bin")
    if bin_override:
        tau2_bin: str | None = str(bin_override)
    elif repo_path:
        tau2_bin = str(repo_path / ".venv" / "bin" / "tau2")
    else:
        tau2_bin = None

    data_override = cfg.get("tau2_data_dir")
    if data_override:
        data_dir: Path | None = _anchor(str(data_override))
    elif repo_path:
        data_dir = repo_path / "data"
    else:
        data_dir = None

    return tau2_bin, data_dir


def build_argv(cfg: Mapping[str, object], run_id: str, *, tau2_bin: str = "tau2") -> list[str]:
    """Assemble the ``tau2 run`` command line from a benchmark stanza."""
    argv = [
        tau2_bin,
        "run",
        "--domain",
        str(cfg["domain"]),
        "--agent-llm",
        str(cfg["agent_llm"]),
        "--user-llm",
        str(cfg["user_llm"]),
        "--task-split-name",
        str(cfg["task_split_name"]),
        "--num-trials",
        str(cfg["num_trials"]),
        "--seed",
        str(cfg["seed"]),
        "--max-concurrency",
        str(cfg["max_concurrency"]),
        "--save-to",
        run_id,
    ]
    if cfg.get("num_tasks") is not None:
        argv += ["--num-tasks", str(cfg["num_tasks"])]
    if cfg.get("timeout") is not None:
        argv += ["--timeout", str(cfg["timeout"])]
    return argv


def load_simulations(data_dir: Path, run_id: str) -> list[dict]:
    """Load the SimulationRun list from a tau2 run's ``results.json``.

    Probes the known output layouts (a tau2 checkout may nest under ``tau2/``)
    and uses the first that exists, so the runner is robust to where the clone
    actually writes results.
    """
    candidates = [
        data_dir / "simulations" / run_id / "results.json",
        data_dir / "tau2" / "simulations" / run_id / "results.json",
        data_dir / "tau2" / "results" / run_id / "results.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        looked = ", ".join(str(p) for p in candidates)
        raise RuntimeError(f"no tau2 results for run '{run_id}'; looked in: {looked}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "simulations" in data:
            return data["simulations"]
        if "results" in data:
            return data["results"]
        return []
    if isinstance(data, list):
        return data
    raise RuntimeError(f"unexpected tau2 results shape in {path}: {type(data).__name__}")


def run_tau2(
    cfg: Mapping[str, object],
    run_id: str,
    *,
    data_dir: Path,
    tau2_bin: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict]:
    """Run tau2 as a subprocess (with ``TAU2_DATA_DIR`` set) and return its sims."""
    argv = build_argv(cfg, run_id, tau2_bin=tau2_bin)
    env = {**os.environ, "TAU2_DATA_DIR": str(data_dir)}
    result = runner(argv, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tau2 run failed (exit {result.returncode}): {' '.join(argv)}")
    return load_simulations(data_dir, run_id)


def load_tasks(data_dir: Path, domain: str) -> dict[str, dict]:
    """Map ``task_id -> task definition`` from tau2's ``tasks.json`` (``{}`` if absent)."""
    for base in (data_dir / "tau2" / "domains", data_dir / "domains"):
        path = base / domain / "tasks.json"
        if path.exists():
            tasks = json.loads(path.read_text(encoding="utf-8"))
            return {str(t.get("id")): t for t in tasks if isinstance(t, dict)}
    return {}


def read_policy(data_dir: Path, domain: str) -> str | None:
    """Return the domain policy markdown (Compass's Ethos), or None.

    A tau2 checkout nests domains under ``tau2/domains/<domain>/``; some data
    dirs are flat (``domains/<domain>/``). For each layout, tries ``policy.md``
    then ``main_policy.md`` (Telecom uses the latter). Returns the first file
    found, else ``None`` (Compass then runs without Ethos).
    """
    for base in (data_dir / "tau2" / "domains", data_dir / "domains"):
        domain_dir = base / domain
        for filename in ("policy.md", "main_policy.md"):
            path = domain_dir / filename
            if path.exists():
                return path.read_text(encoding="utf-8")
    return None


def policy_version(policy_text: str | None) -> str:
    """A short, stable version string for ATIF agent.version from the policy."""
    if not policy_text:
        return "unknown"
    return hashlib.sha256(policy_text.encode("utf-8")).hexdigest()[:12]
