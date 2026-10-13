# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.jobs._cli_options` — the pure CLI helpers.

These are the dotted-key / options-file / merge primitives the ``submit``
verb assembles the wire options bag from. Pure logic — no I/O beyond
``Path.read_text`` in the file loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nemo_helix_plugin.jobs._cli_options import (
    load_options_file,
    load_spec_file,
    merge_options,
    parse_dotted_kv_list,
)

# ---------------------------------------------------------------------------
# parse_dotted_kv_list
# ---------------------------------------------------------------------------


class TestParseDottedKv:
    def test_single_top_level(self) -> None:
        assert parse_dotted_kv_list(["partition=gpu-long"]) == {"partition": "gpu-long"}

    def test_nested_two_levels(self) -> None:
        assert parse_dotted_kv_list(["slurm.partition=gpu-long"]) == {"slurm": {"partition": "gpu-long"}}

    def test_multiple_keys_same_backend(self) -> None:
        assert parse_dotted_kv_list(
            [
                "slurm.partition=gpu-long",
                "slurm.nodes=4",
                "slurm.time_limit=02:00:00",
            ]
        ) == {
            "slurm": {
                "partition": "gpu-long",
                "nodes": "4",
                "time_limit": "02:00:00",
            }
        }

    def test_multiple_backends(self) -> None:
        assert parse_dotted_kv_list(
            [
                "slurm.partition=gpu-long",
                "docker.shm_size=2Gi",
            ]
        ) == {"slurm": {"partition": "gpu-long"}, "docker": {"shm_size": "2Gi"}}

    def test_three_level_nesting(self) -> None:
        assert parse_dotted_kv_list(["slurm.resources.shm_size=2Gi"]) == {"slurm": {"resources": {"shm_size": "2Gi"}}}

    def test_values_arrive_as_strings_no_coercion(self) -> None:
        """CLI deliberately doesn't coerce -o values — server handles types."""
        result = parse_dotted_kv_list(["slurm.nodes=4", "slurm.gpu_enabled=true"])
        assert result == {"slurm": {"nodes": "4", "gpu_enabled": "true"}}
        assert result["slurm"]["nodes"] == "4"  # not 4

    def test_empty_list_returns_empty_dict(self) -> None:
        assert parse_dotted_kv_list([]) == {}

    def test_value_may_contain_equals(self) -> None:
        # Only the first '=' separates key from value.
        assert parse_dotted_kv_list(["slurm.env=FOO=bar"]) == {"slurm": {"env": "FOO=bar"}}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            parse_dotted_kv_list(["slurm.partition"])

    def test_empty_segment_raises(self) -> None:
        with pytest.raises(ValueError, match="empty segment"):
            parse_dotted_kv_list(["slurm..nodes=4"])

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="empty segment"):
            parse_dotted_kv_list(["=value"])


# ---------------------------------------------------------------------------
# load_options_file / load_spec_file
# ---------------------------------------------------------------------------


class TestLoadFiles:
    def test_loads_json_by_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "opts.json"
        f.write_text(json.dumps({"slurm": {"nodes": 4}}))
        assert load_options_file(f) == {"slurm": {"nodes": 4}}

    def test_loads_yaml_by_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "opts.yaml"
        f.write_text("slurm:\n  nodes: 4\n  partition: gpu-long\n")
        assert load_options_file(f) == {"slurm": {"nodes": 4, "partition": "gpu-long"}}

    def test_loads_yml_by_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "opts.yml"
        f.write_text("docker:\n  shm_size: 2Gi\n")
        assert load_options_file(f) == {"docker": {"shm_size": "2Gi"}}

    def test_unknown_extension_tries_json_then_yaml(self, tmp_path: Path) -> None:
        # File contents are YAML without .yaml extension.
        f = tmp_path / "opts.txt"
        f.write_text("slurm:\n  nodes: 4\n")
        assert load_options_file(f) == {"slurm": {"nodes": 4}}

    def test_top_level_scalar_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "opts.json"
        f.write_text("42")
        with pytest.raises(ValueError, match="top-level mapping"):
            load_options_file(f)

    def test_spec_file_loader_same_contract(self, tmp_path: Path) -> None:
        f = tmp_path / "spec.yaml"
        f.write_text("num_records: 100\nmodel: gpt-oss-120b\n")
        assert load_spec_file(f) == {"num_records": 100, "model": "gpt-oss-120b"}

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_options_file(tmp_path / "does-not-exist.json")


# ---------------------------------------------------------------------------
# merge_options
# ---------------------------------------------------------------------------


class TestMergeOptions:
    def test_empty_inputs_produce_empty(self) -> None:
        assert merge_options(None, None) == {}
        assert merge_options({}, None) == {}
        assert merge_options(None, {}) == {}

    def test_overlay_wins_at_leaf(self) -> None:
        base = {"slurm": {"nodes": 4, "partition": "batch"}}
        overlay = {"slurm": {"partition": "gpu-long"}}
        assert merge_options(base, overlay) == {"slurm": {"nodes": 4, "partition": "gpu-long"}}

    def test_overlay_adds_new_keys(self) -> None:
        base = {"slurm": {"nodes": 4}}
        overlay = {"slurm": {"partition": "gpu-long"}, "docker": {"shm_size": "2Gi"}}
        assert merge_options(base, overlay) == {
            "slurm": {"nodes": 4, "partition": "gpu-long"},
            "docker": {"shm_size": "2Gi"},
        }

    def test_overlay_scalar_replaces_base_dict(self) -> None:
        # Backend-level replacement is fine — the submitter overrode the
        # whole block, even if the base had a nested dict.
        base = {"slurm": {"nodes": 4}}
        overlay = {"slurm": "disabled"}
        assert merge_options(base, overlay) == {"slurm": "disabled"}

    def test_lists_are_replaced_not_concatenated(self) -> None:
        base = {"slurm": {"env": ["A=1"]}}
        overlay = {"slurm": {"env": ["B=2"]}}
        assert merge_options(base, overlay) == {"slurm": {"env": ["B=2"]}}

    def test_deep_copy_isolation_of_inputs(self) -> None:
        base = {"slurm": {"nodes": 4}}
        overlay = {"slurm": {"partition": "gpu-long"}}
        merged = merge_options(base, overlay)
        merged["slurm"]["nodes"] = 999
        assert base == {"slurm": {"nodes": 4}}

    def test_three_levels_deep(self) -> None:
        base = {"slurm": {"resources": {"shm_size": "1Gi", "cpu": "4"}}}
        overlay = {"slurm": {"resources": {"shm_size": "2Gi"}}}
        assert merge_options(base, overlay) == {"slurm": {"resources": {"shm_size": "2Gi", "cpu": "4"}}}
