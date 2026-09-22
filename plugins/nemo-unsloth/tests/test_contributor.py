# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for UnslothContributor.

Pin the contract the customization-router hub depends on:

- ``name`` and ``dependencies`` (used by the hub's dep merger).
- ``get_routers`` returns the jobs router under the right prefix, with
  ``@path_rule`` authz stamped on the generated job routes (the platform
  derives the policy from those rules — there is no ``get_authz_contribution``).
- ``get_cli`` exposes ``submit`` / ``explain`` and the submit group accepts the
  ``JOB_JSON`` positional.
"""

from __future__ import annotations

import re

import pytest
from nemo_helix_plugin.customization_contributor import CustomizationContributor
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture
def contributor() -> CustomizationContributor:
    from nemo_unsloth_plugin.contributor import UnslothContributor

    return UnslothContributor()


class TestIdentity:
    def test_name(self, contributor: CustomizationContributor) -> None:
        assert contributor.name == "unsloth"

    def test_dependencies_match_submit_path(self, contributor: CustomizationContributor) -> None:
        # Remote container submit needs the same set of platform services
        # automodel needs: workspace/auth, jobs API, secrets, files + models.
        for required in ("entities", "auth", "jobs", "files", "secrets", "models"):
            assert required in contributor.dependencies, f"{required!r} missing from {contributor.dependencies!r}"


class TestAuthz:
    def test_job_routes_carry_unsloth_path_rules(self, contributor: CustomizationContributor) -> None:
        """Authz is derived from ``@path_rule`` on the generated job routes
        (permission namespace ``customization.unsloth.jobs`` from
        ``AuthzScope("customization").child(name, "jobs")``), not a separate
        ``get_authz_contribution`` declaration."""
        from fastapi.routing import APIRoute
        from nemo_helix_plugin.authz import get_path_rules

        try:
            specs = contributor.get_routers()
        except ImportError as exc:
            pytest.skip(f"router deps unavailable in this env: {exc}")

        route_rules = [
            (route.path, get_path_rules(route.endpoint))
            for spec in specs
            for route in spec.router.routes
            if isinstance(route, APIRoute)
        ]
        assert route_rules
        unruled = [path for path, rules in route_rules if not rules]
        assert not unruled, f"routes without a @path_rule (would be denied fail-closed): {unruled}"

        perm_ids = {perm.id for _path, rules in route_rules for rule in rules for perm in rule.permissions}
        assert "customization.unsloth.jobs.create" in perm_ids


class TestRouters:
    def test_returns_jobs_router_spec(self, contributor: CustomizationContributor) -> None:
        specs = ()
        try:
            specs = contributor.get_routers()
        except ImportError as exc:
            pytest.skip(f"router deps unavailable in this env: {exc}")
        # Only the jobs router — health lives on the customization router hub,
        # not per contributor.
        assert len(specs) == 1
        prefixes = {s.prefix for s in specs}
        # The jobs router is mounted at the workspace prefix; add_job_routes
        # adds the /unsloth/jobs suffix internally based on
        # UnslothJob.job_collection_path.
        assert "/v2/workspaces/{workspace}" in prefixes


class TestCLI:
    def test_cli_root_help_lists_submit_and_explain_only(self, contributor: CustomizationContributor) -> None:
        try:
            cli = contributor.get_cli()
        except ImportError as exc:
            pytest.skip(f"CLI deps unavailable in this env: {exc}")
        assert cli is not None
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "submit" in plain
        assert "explain" in plain
        assert "run" not in plain

    def test_submit_help_shows_job_json_positional(self, contributor: CustomizationContributor) -> None:
        try:
            cli = contributor.get_cli()
        except ImportError as exc:
            pytest.skip(f"CLI deps unavailable in this env: {exc}")
        assert cli is not None
        runner = CliRunner()
        result = runner.invoke(cli, ["submit", "--help"])
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "JOB_JSON" in plain
        assert "--workspace" in plain or "-w" in plain
        assert "--profile" in plain
        assert "--base-url" in plain


class TestSDK:
    def test_exposes_sdk_resources(self, contributor: CustomizationContributor) -> None:
        from nemo_unsloth_plugin.sdk.resources import AsyncUnslothCustomization, UnslothCustomization

        sdk = contributor.get_sdk_resources()
        assert sdk is not None
        assert sdk.sync_resource is UnslothCustomization
        assert sdk.async_resource is AsyncUnslothCustomization
