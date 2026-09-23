# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_guardrails_plugin.benchmarks.paths import (
    build_run_paths,
    default_nemoguardrails_repo_root,
    discover_nhx_repo_root,
)


def _make_fake_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "plugins").mkdir()
    return root


class TestDiscoverNhxRepoRoot:
    def test_finds_root_from_nested_path(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path / "repo")
        nested = repo / "plugins" / "foo" / "src" / "deep"
        nested.mkdir(parents=True)

        assert discover_nhx_repo_root(nested) == repo

    def test_returns_repo_when_pointed_directly_at_it(self, tmp_path: Path) -> None:
        repo = _make_fake_repo(tmp_path / "repo")
        assert discover_nhx_repo_root(repo) == repo

    def test_raises_when_no_repo_in_ancestry(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="repo root"):
            discover_nhx_repo_root(tmp_path)


class TestDefaultNgRepoRoot:
    def test_sibling_of_nhx_root(self, tmp_path: Path) -> None:
        nhx = _make_fake_repo(tmp_path / "nemo-helix")
        ng = tmp_path / "NeMo-Guardrails"

        assert default_nemoguardrails_repo_root(nhx) == ng.resolve()


class TestBuildRunPaths:
    def test_layout_matches_documented_structure(self, tmp_path: Path) -> None:
        nhx = _make_fake_repo(tmp_path / "nemo-helix")
        ng = tmp_path / "NeMo-Guardrails"
        ng.mkdir()

        paths = build_run_paths(nhx_repo_root=nhx, nemoguardrails_repo_root=ng, run_id="20260527_120000")

        assert paths.run_dir == nhx / "plugins/nemo-guardrails/benchmarks/artifacts/runs/20260527_120000"
        assert paths.log_dir == paths.run_dir / "logs"
        assert paths.generated_dir == paths.run_dir / "generated"
        assert paths.aiperf_output_dir == paths.run_dir / "aiperf_results"
        assert paths.nhx_data_dir == nhx / "plugins/nemo-guardrails/benchmarks/artifacts/nhx-data"
        assert (
            paths.config_template
            == nhx / "plugins/nemo-guardrails/benchmarks/configs/nhx_igw_guardrails_sweep_concurrency.yaml"
        )
        assert paths.runtime_config == paths.generated_dir / "nhx_igw_guardrails_sweep_concurrency.yaml"

    def test_ensure_directories_creates_required_dirs(self, tmp_path: Path) -> None:
        nhx = _make_fake_repo(tmp_path / "nemo-helix")
        ng = tmp_path / "NeMo-Guardrails"
        ng.mkdir()

        paths = build_run_paths(nhx_repo_root=nhx, nemoguardrails_repo_root=ng, run_id="x")
        paths.ensure_directories()

        assert paths.log_dir.is_dir()
        assert paths.generated_dir.is_dir()
        assert paths.aiperf_output_dir.is_dir()
        assert paths.nhx_data_dir.is_dir()

    def test_run_id_uses_timestamp_when_not_given(self, tmp_path: Path) -> None:
        nhx = _make_fake_repo(tmp_path / "nemo-helix")
        ng = tmp_path / "NeMo-Guardrails"
        ng.mkdir()

        paths = build_run_paths(nhx_repo_root=nhx, nemoguardrails_repo_root=ng)
        # Timestamp format: YYYYmmdd_HHMMSS = 15 chars including underscore.
        assert len(paths.run_id) == 15
        assert paths.run_id[8] == "_"

    def test_aiperf_venv_dir_is_outside_run_dir(self, tmp_path: Path) -> None:
        """The cached aiperf venv must be shared across runs, not under run_dir."""
        nhx = _make_fake_repo(tmp_path / "nemo-helix")
        ng = tmp_path / "NeMo-Guardrails"
        ng.mkdir()

        paths = build_run_paths(nhx_repo_root=nhx, nemoguardrails_repo_root=ng, run_id="x")
        assert paths.run_dir not in paths.aiperf_venv_dir.parents
        assert paths.aiperf_venv_dir.parent.name == "venvs"
