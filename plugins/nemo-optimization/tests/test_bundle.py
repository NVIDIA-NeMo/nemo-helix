# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight for an optimize bundle staged with ``nemo agents optimize prepare-fileset``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from nemo_optimization.bundle import BundlePreflightError, preflight_bundle

FABRIC_AGENT: dict[str, Any] = {
    "schema_version": "fabric.agent/v1alpha1",
    "metadata": {"name": "hermes-optimize-demo"},
    "harness": {"adapter_id": "nvidia.fabric.hermes"},
    "models": {"default": {"provider": "nvidia", "model": "nvidia/meta/llama-3.1-8b-instruct"}},
}
OPTIMIZER: dict[str, Any] = {
    "numeric": {"enabled": True, "n_trials": 2},
    "search_space": {"temperature": {"type": "fabric", "path": "models.default.temperature", "values": [0.0, 0.2]}},
}


def make_bundle(root: Path, config: dict[str, Any], *, files: dict[str, str] | None = None) -> Path:
    (root / "optimize.yml").write_text(yaml.safe_dump(config))
    for relative, contents in (files or {}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    return root


def full_config(**eval_overrides: Any) -> dict[str, Any]:
    return {
        **FABRIC_AGENT,
        "optimizer": OPTIMIZER,
        "eval": {
            "general": {"dataset": {"file_path": "dataset.json"}},
            "fabric": {"base_dir": "."},
            **eval_overrides,
        },
    }


DATASET = json.dumps([{"question": "q", "answer": "a"}])


def test_accepts_a_self_contained_bundle(tmp_path: Path) -> None:
    make_bundle(tmp_path, full_config(), files={"dataset.json": DATASET})
    config = preflight_bundle(tmp_path, "optimize.yml")
    assert config["optimizer"]["numeric"]["enabled"] is True


def test_accepts_a_config_in_a_subdirectory(tmp_path: Path) -> None:
    config = full_config()
    config["eval"]["general"]["dataset"] = {"file_path": "data/rows.json"}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "optimize.yml").write_text(yaml.safe_dump(config))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rows.json").write_text(DATASET)

    assert preflight_bundle(tmp_path, "configs/optimize.yml")["eval"]["fabric"]["base_dir"] == "."


def test_rejects_absolute_asset_paths(tmp_path: Path) -> None:
    config = full_config()
    config["eval"]["general"]["dataset"] = {"file_path": "/Users/me/agents/dataset.json"}
    make_bundle(tmp_path, config)

    with pytest.raises(BundlePreflightError, match="eval.general.dataset is an absolute path"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_rejects_assets_missing_from_the_bundle(tmp_path: Path) -> None:
    make_bundle(tmp_path, full_config())

    with pytest.raises(BundlePreflightError, match="eval.general.dataset points at 'dataset.json'"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_reports_every_problem_at_once(tmp_path: Path) -> None:
    config = full_config()
    config["eval"]["general"]["dataset"] = {"file_path": "/abs/dataset.json"}
    config["eval"]["fabric"]["base_dir"] = "missing-dir"
    make_bundle(tmp_path, config)

    with pytest.raises(BundlePreflightError) as excinfo:
        preflight_bundle(tmp_path, "optimize.yml")
    message = str(excinfo.value)
    assert "2 problem(s)" in message
    assert "eval.general.dataset" in message
    assert "eval.fabric.base_dir" in message


def test_rejects_a_bundle_without_an_agent_under_test(tmp_path: Path) -> None:
    config = full_config()
    del config["schema_version"]
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="no Agent under Test"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_accepts_an_overlay_only_config_with_a_platform_agent(tmp_path: Path) -> None:
    config = full_config()
    del config["schema_version"]
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    assert preflight_bundle(tmp_path, "optimize.yml", agent="hermes-chatonly")["optimizer"] == OPTIMIZER


def test_rejects_legacy_nat_workflow_yaml(tmp_path: Path) -> None:
    make_bundle(tmp_path, {"workflow": {"_type": "react_agent"}, "optimizer": OPTIMIZER})

    with pytest.raises(BundlePreflightError, match="legacy NAT workflow YAML"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_rejects_a_config_with_no_optimizer_enabled(tmp_path: Path) -> None:
    config = full_config()
    config["optimizer"] = {"numeric": {"enabled": False}}
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="no optimizer is enabled"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_rejects_numeric_optimization_with_an_empty_search_space(tmp_path: Path) -> None:
    config = full_config()
    config["optimizer"] = {"numeric": {"enabled": True}}
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="search_space is empty"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_rejects_prompt_optimization_without_model_reference(tmp_path: Path) -> None:
    config = full_config()
    config["instructions"] = {"system": {"content": "Base prompt."}}
    config["optimizer"] = {
        "prompt": {"enabled": True},
        "search_space": {
            "system_prompt": {
                "type": "fabric",
                "path": "instructions.system.content",
                "is_prompt": True,
                "purpose": "Answer accurately.",
            }
        },
    }
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="optimizer.prompt.model"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_rejects_prompt_optimization_when_inline_prompt_path_is_not_string(tmp_path: Path) -> None:
    config = full_config()
    config["models"]["prompt_optimizer"] = {"provider": "openai", "model": "gpt-5-mini"}
    config["instructions"] = {"system": {"content": {"text": "Base prompt."}}}
    config["optimizer"] = {
        "prompt": {"enabled": True, "model": "prompt_optimizer"},
        "search_space": {
            "system_prompt": {
                "type": "fabric",
                "path": "instructions.system.content",
                "is_prompt": True,
                "purpose": "Answer accurately.",
            }
        },
    }
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="must resolve to a string"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_checks_stdio_mcp_server_scripts_ship_in_the_bundle(tmp_path: Path) -> None:
    config = full_config()
    config["mcp"] = {
        "servers": {
            "analyzer": {
                "transport": "stdio",
                "url": "python3",
                "args": ["mcps/analyzer.py", "--quiet", "--endpoint", "https://api.example.test/v1/chat", "42"],
            },
            "remote": {"transport": "http", "url": "https://example.test/mcp"},
        }
    }
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError) as excinfo:
        preflight_bundle(tmp_path, "optimize.yml")
    message = str(excinfo.value)
    assert "mcp.servers.analyzer.args[0]" in message
    for not_a_path in ("--quiet", "https://api.example.test", "args[3]", "args[4]"):
        assert not_a_path not in message  # flags, URLs and plain values are not bundle files

    (tmp_path / "mcps").mkdir()
    (tmp_path / "mcps" / "analyzer.py").write_text("print(1)\n", encoding="utf-8")
    assert preflight_bundle(tmp_path, "optimize.yml")["mcp"]["servers"]["analyzer"]["args"][0] == "mcps/analyzer.py"


def test_checks_a_stdio_mcp_server_launched_directly_from_a_bundle_url(tmp_path: Path) -> None:
    """``url`` may name the bundled script itself; the runtime resolves it, so preflight checks it too."""
    config = full_config()
    config["mcp"] = {
        "servers": {
            "bundled": {"transport": "stdio", "url": "mcps/server.py"},
            # Commands on PATH and absolute host executables are not bundle files and are left alone.
            "on_path": {"transport": "stdio", "url": "email-phishing-analyzer-mcp"},
            "host": {"transport": "stdio", "url": "/opt/analyzer/bin/analyzer-mcp"},
        }
    }
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError) as excinfo:
        preflight_bundle(tmp_path, "optimize.yml")
    message = str(excinfo.value)
    assert "mcp.servers.bundled.url" in message
    assert "on_path" not in message and "host" not in message

    (tmp_path / "mcps").mkdir()
    (tmp_path / "mcps" / "server.py").write_text("print(1)\n", encoding="utf-8")
    assert preflight_bundle(tmp_path, "optimize.yml")["mcp"]["servers"]["bundled"]["url"] == "mcps/server.py"


def test_ignores_a_dataset_staged_from_its_own_fileset(tmp_path: Path) -> None:
    config = full_config()
    config["eval"]["general"]["dataset"] = {"file_path": "default/evals#rows.json"}
    make_bundle(tmp_path, config)

    assert preflight_bundle(tmp_path, "optimize.yml")["eval"]["general"]["dataset"]["file_path"] == (
        "default/evals#rows.json"
    )


def test_does_not_require_output_directories_to_exist(tmp_path: Path) -> None:
    config = {
        **full_config(),
        "runtime": {"artifacts": "./artifacts"},
        "environment": {"provider": "local", "workspace": "./.tmp/workspace", "artifacts": "./artifacts"},
    }
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    assert preflight_bundle(tmp_path, "optimize.yml")["runtime"]["artifacts"] == "./artifacts"


def test_rejects_absolute_output_directories(tmp_path: Path) -> None:
    config = {**full_config(), "environment": {"provider": "local", "workspace": "/var/tmp/workspace"}}
    make_bundle(tmp_path, config, files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="environment.workspace is an absolute path"):
        preflight_bundle(tmp_path, "optimize.yml")


@pytest.mark.parametrize("config_path", ["../optimize.yml", "/abs/optimize.yml", "D:optimize.yml"])
def test_rejects_a_config_path_outside_the_source(tmp_path: Path, config_path: str) -> None:
    make_bundle(tmp_path, full_config(), files={"dataset.json": DATASET})

    with pytest.raises(BundlePreflightError, match="must be a path relative to --source"):
        preflight_bundle(tmp_path, config_path)


def test_reports_a_missing_config(tmp_path: Path) -> None:
    with pytest.raises(BundlePreflightError, match="was not found under"):
        preflight_bundle(tmp_path, "optimize.yml")


def test_reports_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "optimize.yml").write_text("optimizer: [unclosed\n")

    with pytest.raises(BundlePreflightError, match="is not valid YAML"):
        preflight_bundle(tmp_path, "optimize.yml")
