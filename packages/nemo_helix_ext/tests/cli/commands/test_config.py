# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the config CLI commands."""

import re
from pathlib import Path

import pytest
import yaml
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.config.config import Config
from typer.testing import CliRunner

from ..utils import assert_exit_code

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch):
    """Create a temp config file and set NHX_CONFIG_FILE env var."""
    config_data = {
        "current_context": "local",
        "clusters": [
            {"name": "local", "base_url": "http://localhost:8080"},
            {"name": "production", "base_url": "https://api.example.com"},
        ],
        "users": [
            {"name": "local", "api_key": "secret-key-123"},
            {"name": "production", "api_key": "secret-key-456"},
        ],
        "contexts": [
            {"name": "local", "cluster": "local", "user": "local", "workspace": "default"},
            {"name": "production", "cluster": "production", "user": "production", "workspace": "prod-ns"},
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_path))
    yield config_path
    monkeypatch.delenv("NHX_CONFIG_FILE", raising=False)


def test_current_context(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config current-context")
    assert_exit_code(result, 0)
    assert "local" in result.stdout


def test_current_context_respects_environment_override(config_file: Path, monkeypatch):
    monkeypatch.setenv("NHX_CURRENT_CONTEXT", "production")

    result = runner.invoke(app, "config current-context")

    assert_exit_code(result, 0)
    assert result.output.strip() == "production"


def test_current_context_cli_flag_beats_env_var(config_file: Path, monkeypatch):
    monkeypatch.setenv("NHX_CURRENT_CONTEXT", "local")

    result = runner.invoke(app, "--context production config current-context")

    assert_exit_code(result, 0)
    assert result.output.strip() == "production"


def test_current_context_no_config_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NHX_CONFIG_FILE", str(tmp_path / "nonexistent.yaml"))
    result = runner.invoke(app, "config current-context")
    assert_exit_code(result, 1)


def test_view(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config view --output-format json")
    assert_exit_code(result, 0)
    assert "current_context" in result.stdout
    assert "clusters" in result.stdout
    assert "contexts" in result.stdout
    assert '"name": "local"' in result.stdout
    assert '"name": "production"' not in result.stdout

    combined_output = f"{result.output}{getattr(result, 'stderr', '')}"
    assert "Showing config for context: local" in combined_output
    assert "--all-contexts" in combined_output


def test_view_all_contexts(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config view --all-contexts --output-format json")
    assert_exit_code(result, 0)
    assert '"name": "local"' in result.stdout
    assert '"name": "production"' in result.stdout


def test_view_context_hint_omits_all_contexts_when_single_context(tmp_path: Path, monkeypatch):
    config_data = {
        "current_context": "local",
        "clusters": [{"name": "local", "base_url": "http://localhost:8080"}],
        "users": [{"name": "local", "api_key": "secret-key-123"}],
        "contexts": [{"name": "local", "cluster": "local", "user": "local", "workspace": "default"}],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_path))
    result = runner.invoke(app, "config view --output-format json")
    monkeypatch.delenv("NHX_CONFIG_FILE", raising=False)

    assert_exit_code(result, 0)
    combined_output = f"{result.output}{getattr(result, 'stderr', '')}"
    assert "Showing config for context: local" in combined_output
    assert "--all-contexts" not in combined_output


def test_view_redacts_secrets(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config view --output-format json")
    assert_exit_code(result, 0)
    assert "***REDACTED***" in result.stdout
    assert "secret-key-123" not in result.stdout


def test_use_context(config_file: Path):
    result = runner.invoke(app, "config use-context production")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Switched to context 'production'"

    with open(config_file) as f:
        data = yaml.safe_load(f)
    assert data["current_context"] == "production"


def test_use_nonexistent_context(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config use-context nonexistent")
    assert_exit_code(result, 1)
    assert "not found" in result.output


def test_delete_context_removes_only_context_by_default(config_file: Path):
    result = runner.invoke(app, "config delete-context production")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Deleted context 'production'"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert {context["name"] for context in data["contexts"]} == {"local"}
    assert {cluster["name"] for cluster in data["clusters"]} == {"local", "production"}
    assert {user["name"] for user in data["users"]} == {"local", "production"}
    assert data["current_context"] == "local"


def test_delete_current_context_clears_current_context(config_file: Path):
    result = runner.invoke(app, "config delete-context local")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Deleted context 'local'\nCurrent context cleared"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert {context["name"] for context in data["contexts"]} == {"production"}
    assert data.get("current_context") is None


def test_delete_context_prune_orphans_removes_unreferenced_records(config_file: Path):
    result = runner.invoke(app, "config delete-context production --prune-orphans")
    assert_exit_code(result, 0)
    assert "Deleted context 'production'" in result.output
    assert "Pruned clusters: production" in result.output
    assert "Pruned users: production" in result.output

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert {context["name"] for context in data["contexts"]} == {"local"}
    assert {cluster["name"] for cluster in data["clusters"]} == {"local"}
    assert {user["name"] for user in data["users"]} == {"local"}


def test_delete_context_rejects_unknown_context(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config delete-context missing")
    assert_exit_code(result, 1)
    assert "Context 'missing' not found" in result.output
    assert "Available contexts:" in result.output
    assert "local" in result.output
    assert "production" in result.output


def test_set_workspace(config_file: Path):
    result = runner.invoke(app, "config set --workspace new-workspace")
    assert_exit_code(result, 0)

    context = Config.load(config_path=config_file).resolve()
    assert context.workspace == "new-workspace"


def test_set_workspace_in_specific_context(config_file: Path):
    result = runner.invoke(app, "config set --context production --workspace prod-workspace")
    assert_exit_code(result, 0)

    context = Config.load(config_path=config_file, overrides={"current_context": "production"}).resolve()
    assert context.workspace == "prod-workspace"


def test_set_access_token(config_file: Path):
    result = runner.invoke(app, "config set --access-token test-access-token-123")
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    user = next(u for u in data["users"] if u["name"] == "local")
    assert user["type"] == "oauth"
    assert user["token"] == "test-access-token-123"
    assert "refresh_token" not in user or user["refresh_token"] is None


def test_set_rejects_api_key_and_access_token_together(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config set --api-key test-key --access-token test-token")
    assert_exit_code(result, 1)
    assert "Cannot specify both --api-key and --access-token" in result.output


def test_set_activate_requires_context(config_file: Path):
    _ = config_file
    result = runner.invoke(app, "config set --workspace foo --activate")
    assert_exit_code(result, 1)
    assert "--activate requires --context" in result.output


def test_set_can_create_and_activate_named_context(config_file: Path):
    result = runner.invoke(app, "config set --context staging --base-url https://staging.example.com --activate")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Context 'staging' created and set as current"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "staging"

    context = next(c for c in data["contexts"] if c["name"] == "staging")
    cluster = next(c for c in data["clusters"] if c["name"] == context["cluster"])
    assert cluster["base_url"] == "https://staging.example.com/"


def test_set_activate_existing_context_reports_switch(config_file: Path):
    result = runner.invoke(app, "config set --context production --activate")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Switched to context 'production'"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "production"


def test_set_update_and_activate_existing_context_reports_both_actions(config_file: Path):
    result = runner.invoke(app, "config set --context production --workspace updated-ws --activate")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Configuration updated for context 'production' and set as current"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "production"
    production = next(context for context in data["contexts"] if context["name"] == "production")
    assert production["workspace"] == "updated-ws"


def test_set_creates_isolated_cluster_per_context(config_file: Path):
    """Creating two contexts via config set should produce distinct clusters."""
    runner.invoke(app, "config set --context alpha --base-url https://alpha.example.com")
    runner.invoke(app, "config set --context beta --base-url https://beta.example.com")

    with open(config_file) as f:
        data = yaml.safe_load(f)

    alpha_ctx = next(c for c in data["contexts"] if c["name"] == "alpha")
    beta_ctx = next(c for c in data["contexts"] if c["name"] == "beta")

    assert alpha_ctx["cluster"] != beta_ctx["cluster"], "Contexts must reference different clusters"

    alpha_cluster = next(c for c in data["clusters"] if c["name"] == alpha_ctx["cluster"])
    beta_cluster = next(c for c in data["clusters"] if c["name"] == beta_ctx["cluster"])

    assert alpha_cluster["base_url"] == "https://alpha.example.com/"
    assert beta_cluster["base_url"] == "https://beta.example.com/"


def test_set_base_url_does_not_affect_other_contexts(config_file: Path):
    """Changing base-url on one context must not clobber another context's URL."""
    runner.invoke(app, "config set --context ctx-a --base-url https://a.example.com")
    runner.invoke(app, "config set --context ctx-b --base-url https://b.example.com")

    # Update ctx-a's URL
    runner.invoke(app, "config set --context ctx-a --base-url https://a-updated.example.com")

    with open(config_file) as f:
        data = yaml.safe_load(f)

    ctx_b = next(c for c in data["contexts"] if c["name"] == "ctx-b")
    b_cluster = next(c for c in data["clusters"] if c["name"] == ctx_b["cluster"])
    assert b_cluster["base_url"] == "https://b.example.com/", "ctx-b's URL must be unchanged"

    ctx_a = next(c for c in data["contexts"] if c["name"] == "ctx-a")
    a_cluster = next(c for c in data["clusters"] if c["name"] == ctx_a["cluster"])
    assert a_cluster["base_url"] == "https://a-updated.example.com/"


def test_set_certificate_authority_updates_selected_context_cluster(config_file: Path):
    result = runner.invoke(app, "config set --context production --certificate-authority /tmp/nemo-ca.pem")
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    local_cluster = next(c for c in data["clusters"] if c["name"] == "local")
    production_cluster = next(c for c in data["clusters"] if c["name"] == "production")
    assert "certificate_authority" not in local_cluster
    assert production_cluster["certificate_authority"] == str(Path("/tmp/nemo-ca.pem").resolve())


def test_set_certificate_authority_persists_relative_path_as_absolute(
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, "config set --context production --certificate-authority ./ca.crt")
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    production_cluster = next(c for c in data["clusters"] if c["name"] == "production")
    assert production_cluster["certificate_authority"] == str((tmp_path / "ca.crt").resolve())


def test_set_certificate_authority_expands_home_relative_path(config_file: Path):
    result = runner.invoke(app, "config set --context production --certificate-authority ~/nemo-ca.pem")
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    production_cluster = next(c for c in data["clusters"] if c["name"] == "production")
    assert production_cluster["certificate_authority"] == str((Path.home() / "nemo-ca.pem").resolve())


def test_set_certificate_authority_empty_string_clears_value(config_file: Path):
    result = runner.invoke(app, "config set --context production --certificate-authority /tmp/nemo-ca.pem")
    assert_exit_code(result, 0)

    result = runner.invoke(app, ["config", "set", "--context", "production", "--certificate-authority", ""])
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    production_cluster = next(c for c in data["clusters"] if c["name"] == "production")
    assert "certificate_authority" not in production_cluster


def test_set_new_context_can_set_certificate_authority(config_file: Path):
    result = runner.invoke(
        app,
        "config set --context local-tls --base-url https://localhost:8443 --certificate-authority /tmp/local-ca.pem",
    )
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    context = next(c for c in data["contexts"] if c["name"] == "local-tls")
    cluster = next(c for c in data["clusters"] if c["name"] == context["cluster"])
    assert cluster["base_url"] == "https://localhost:8443/"
    assert cluster["certificate_authority"] == str(Path("/tmp/local-ca.pem").resolve())


def test_set_new_named_context_does_not_switch(config_file: Path):
    result = runner.invoke(app, "config set --context new-ctx --base-url https://new.example.com")
    assert_exit_code(result, 0)
    assert result.output.strip() == (
        "Context 'new-ctx' created\nTo make it current, run: nemo config use-context new-ctx"
    )

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "local"


def test_set_first_named_context_becomes_current(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_path))

    result = runner.invoke(app, "config set --context staging --base-url https://staging.example.com")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Context 'staging' created and set as current"

    with open(config_path) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "staging"


def test_set_new_context_hint_handles_leading_hyphen(config_file: Path):
    _ = config_file  # fixture sets up existing config so use-context below can succeed
    result = runner.invoke(
        app,
        ["config", "set", "--context", "-qa", "--base-url", "https://qa.example.com"],
    )
    assert_exit_code(result, 0)
    assert "To make it current, run: nemo config use-context -- -qa" in result.output

    switch_result = runner.invoke(app, ["config", "use-context", "--", "-qa"])
    assert_exit_code(switch_result, 0)
    assert Config.load(config_path=config_file).get_config_file().current_context == "-qa"


def test_set_existing_context_does_not_switch(config_file: Path):
    """Modifying an existing context should not change the current context."""
    result = runner.invoke(app, "config set --context production --workspace updated-ws")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Configuration updated for context 'production'"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "local", "Should stay on 'local', not switch to 'production'"


def test_set_reports_context_selected_by_environment(config_file: Path, monkeypatch):
    monkeypatch.setenv("NHX_CURRENT_CONTEXT", "production")

    result = runner.invoke(app, "config set --workspace updated-ws")
    assert_exit_code(result, 0)
    assert result.output.strip() == "Configuration updated for context 'production'"

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["current_context"] == "local"
    production = next(context for context in data["contexts"] if context["name"] == "production")
    assert production["workspace"] == "updated-ws"


def test_set_rejects_unknown_context_selected_by_environment(config_file: Path, monkeypatch):
    monkeypatch.setenv("NHX_CURRENT_CONTEXT", "prodction")
    with open(config_file) as f:
        before = yaml.safe_load(f)

    result = runner.invoke(app, "config set --workspace updated-ws")

    assert_exit_code(result, 1)
    assert "Context 'prodction' not found" in result.output

    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data == before


def test_set_new_context_creates_dedicated_cluster_and_user(config_file: Path):
    """New contexts should get their own cluster and user with descriptive names."""
    result = runner.invoke(app, "config set --context myctx --base-url https://my.example.com")
    assert_exit_code(result, 0)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    ctx = next(c for c in data["contexts"] if c["name"] == "myctx")
    assert ctx["cluster"] == "myctx-cluster"
    assert ctx["user"] == "myctx-user"

    cluster = next(c for c in data["clusters"] if c["name"] == "myctx-cluster")
    assert cluster["base_url"] == "https://my.example.com/"


@pytest.mark.parametrize("invalid_url", ["foo.com", "not-a-url", "example.com:8080"])
def test_set_rejects_invalid_base_url(config_file: Path, invalid_url: str):
    _ = config_file
    result = runner.invoke(app, f"config set --base-url {invalid_url}")
    assert_exit_code(result, 1)
    assert "Invalid base URL" in result.output
    assert "must include a scheme" in result.output


@pytest.mark.parametrize(
    "command_name",
    [
        "delete-cluster",
        "delete-user",
        "get-clusters",
        "get-contexts",
        "get-users",
        "set-cluster",
        "set-context",
        "set-user",
    ],
)
def test_advanced_commands_are_not_available(command_name: str):
    result = runner.invoke(app, f"config {command_name}")
    assert_exit_code(result, 2)
    assert "No such command" in result.output


def test_config_help_emphasizes_core_flow():
    result = runner.invoke(app, "config --help")
    assert_exit_code(result, 0)
    assert "set --base-url" in result.output
    assert "config view" in result.output
    assert "config view --minify" not in result.output
    assert "config use-context" in result.output
    assert "config delete-context" in result.output
    assert "Advanced:" not in result.output

    view_match = re.search(r"^\s*view\s+", result.output, re.MULTILINE)
    set_match = re.search(r"^\s*set\s+", result.output, re.MULTILINE)
    current_context_match = re.search(r"^\s*current-context\s+", result.output, re.MULTILINE)
    use_context_match = re.search(r"^\s*use-context\s+", result.output, re.MULTILINE)
    delete_context_match = re.search(r"^\s*delete-context\s+", result.output, re.MULTILINE)

    assert view_match is not None
    assert set_match is not None
    assert current_context_match is not None
    assert use_context_match is not None
    assert delete_context_match is not None
    assert view_match.start() < set_match.start()
    assert set_match.start() < current_context_match.start()
    assert current_context_match.start() < use_context_match.start()
    assert use_context_match.start() < delete_context_match.start()


def test_view_help_uses_all_contexts_option():
    result = runner.invoke(app, "config view --help")
    assert_exit_code(result, 0)
    assert "--all-contexts" in result.output
    assert "--minify" not in result.output
