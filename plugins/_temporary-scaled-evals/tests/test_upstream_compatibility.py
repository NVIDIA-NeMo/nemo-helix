# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cover portable upstream compatibility behavior."""

from __future__ import annotations

import json
import runpy
import tomllib
from pathlib import Path

import pytest

try:
    from scaled_evals.cli.main import _default_switchyard_dockerfile_path
    from scaled_evals.dispatch.switchyard import SwitchyardProfileConfig, _routing_profiles_text
    from scaled_evals.harbor_runners import resolve_harbor_runner, supported_harbor_versions
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_switchyard_defaults_to_native_rust_server_and_parses_toml(tmp_path: Path) -> None:
    dockerfile = tmp_path / "benchmark" / "switchyard-rust-server.Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM rust:1.96.1-bookworm\n")

    assert (
        _default_switchyard_dockerfile_path(tmp_path, context_path=".", requested=None)
        == "benchmark/switchyard-rust-server.Dockerfile"
    )

    rendered = _routing_profiles_text(
        SwitchyardProfileConfig(
            routing_config_toml="""
# preserved
[llm_clients.primary]
format = "openai_chat_completions"
extra_headers = { x-inference-priority = "interactive", X-Existing = "retained" }

[llm_clients.fallback]
format = "anthropic_messages"

[tool.ruff]
[tool.ruff.lint.a]
[tool.ruff.lint]
[[tool.poetry.source]]
[tool.ruff.lint.b]
"""
        )
    )
    config = tomllib.loads(rendered)
    assert "# preserved" in rendered
    assert config["llm_clients"]["primary"]["extra_headers"] == {
        "X-Existing": "retained",
        "X-Inference-Priority": "batch",
    }
    assert config["llm_clients"]["fallback"]["extra_headers"] == {"X-Inference-Priority": "batch"}
    assert config["tool"]["ruff"]["lint"]["a"] == {}
    assert config["tool"]["ruff"]["lint"]["b"] == {}

    with pytest.raises(ValueError, match="routing config formats are mutually exclusive"):
        SwitchyardProfileConfig(routing_config_toml="schema_version = 1", routing_profiles={"routes": {}})


def test_harbor_catalog_and_compose_image_advertise_the_same_runners() -> None:
    catalog = json.loads((PLUGIN_ROOT / "src/scaled_evals/data/harbor_runner_qualifications.json").read_text())
    assert catalog["default_version"] == "0.13.2"
    assert catalog["aliases"]["default"] == "0.13.2"
    assert "0.20.0" in supported_harbor_versions()
    assert resolve_harbor_runner("0.20.0").harbor_dir == "/opt/harbor/0.20.0"
    assert catalog["adapter"] == {
        "version": "nemo-helix-plugin-overlay-v1",
        "files": [
            "harbor-patches/patch_langgraph_writable_venv.py",
            "harbor-patches/patch_pi_extra_env.py",
            "harbor-patches/patch_sandbox_k8s_root.py",
            "harbor-patches/sandbox_k8s_harbor.py",
        ],
    }

    dockerfile = (PLUGIN_ROOT / "deploy/compose/Dockerfile").read_text()
    assert "selectable-versions.txt" in dockerfile
    assert "for version in $(cat /opt/harbor/selectable-versions.txt)" in dockerfile
    assert 'test -x "${runner}/.venv/bin/harbor"' in dockerfile


def test_generic_harbor_020_patches_cover_langgraph_and_pi(tmp_path: Path) -> None:
    langgraph_patch = runpy.run_path(str(PLUGIN_ROOT / "harbor-patches/patch_langgraph_writable_venv.py"))["patch"]
    langgraph = tmp_path / "langgraph.py"
    langgraph.write_text(
        '_REMOTE_VENV_DIR = PurePosixPath("/opt/harbor-langgraph-venv")\n'
        'command = f"uv venv {venv_dir} --python {python_version} --clear; "\n'
    )
    langgraph_patch(langgraph)
    assert "/tmp/harbor-langgraph-venv" in langgraph.read_text()
    assert "uv venv {venv_dir} --python {python_version} --clear" in langgraph.read_text()
    assert "/opt/harbor/pip.pyz" in (PLUGIN_ROOT / "deploy/compose/Dockerfile").read_text()

    legacy = tmp_path / "legacy_langgraph.py"
    legacy.write_text(
        '_REMOTE_VENV_DIR = PurePosixPath("/opt/harbor-langgraph-venv")\n'
        'command = f"python3 -m venv {venv_dir}; " "python -m pip install uv; "\n'
    )
    langgraph_patch(legacy)
    assert "/tmp/harbor-langgraph-venv" in legacy.read_text()
    assert "python /tmp/pip.pyz install uv" in legacy.read_text()
    assert "bootstrap.pypa.io" not in legacy.read_text()

    unknown = tmp_path / "unknown_langgraph.py"
    unknown.write_text("class LangGraph:\n    pass\n")
    with pytest.raises(RuntimeError, match="source anchor"):
        langgraph_patch(unknown)

    pi_patch = runpy.run_path(str(PLUGIN_ROOT / "harbor-patches/patch_pi_extra_env.py"))["patch"]
    pi = tmp_path / "pi.py"
    pi.write_text(
        "import json\nimport os\nimport shlex\n"
        "val = os.environ.get(key)\n"
        "    @with_prompt_template\n"
        "    async def run(\n"
        '                f". ~/.nvm/nvm.sh; "\n'
        '                f"pi --print --mode json --session-dir /logs/agent/pi/sessions "\n'
        '                f"{escaped_instruction} "\n'
        '                f"2>&1 </dev/null |"\n'
        "        skills_command = self._build_register_skills_command()\n"
    )
    pi_patch(pi)
    patched = pi.read_text()
    assert 'f"printf %s {escaped_instruction} | "' in patched
    assert "val = self._get_env(key)" in patched
    assert "_write_inference_headers_config" in patched
    assert "</dev/null" not in patched

    pi_patch(pi)
    assert pi.read_text() == patched
