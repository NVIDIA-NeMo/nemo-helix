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
from nemo_platform_plugin.customization_contributor import CustomizationContributor
from nemo_unsloth_plugin.contributor import UnslothContributor
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture
def contributor() -> CustomizationContributor:
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
        from nemo_platform_plugin.authz import get_path_rules

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
        # Match on the registered verbs, not the rendered text: the help prose
        # legitimately contains words like "runs".
        assert {cmd.name for cmd in cli.registered_commands} == {"submit", "explain"}

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


class TestCLIHelp:
    """Typed against the concrete class: ``cli_help`` is a backend detail, not protocol."""

    @pytest.fixture
    def unsloth(self) -> UnslothContributor:
        return UnslothContributor()

    def test_cli_summary_states_what_it_trains_and_where_it_runs(self, unsloth: UnslothContributor) -> None:
        summary = unsloth.get_cli_summary()
        assert summary is not None
        assert "SFT" in summary.trains and "LoRA" in summary.trains
        assert "One job uses one GPU." in summary.runs_on
        assert summary.command == "nemo customization unsloth submit job.json"

    def test_cli_summary_fits_the_rendered_width(self, unsloth: UnslothContributor) -> None:
        """The router prints the summary through an 80-column Rich console."""
        summary = unsloth.get_cli_summary()
        assert summary is not None
        rendered = summary.render("unsloth")
        assert [line for line in rendered.splitlines() if len(line) > 80] == []

    def test_backend_help_goes_deeper_than_the_top_level_summary(self, unsloth: UnslothContributor) -> None:
        summary = unsloth.get_cli_summary()
        assert summary is not None
        summary_text = summary.render(unsloth.name)
        help_text = unsloth.cli_help

        assert len(help_text) > len(summary_text)
        # Job JSON fields and schema names belong on the backend, not in the overview.
        for detail in ("UnslothJobInput", "gradient_accumulation_steps", "save_method", "explain"):
            assert detail in help_text, detail
            assert detail not in summary_text, detail

    def test_submit_help_explains_the_job_json(self, unsloth: UnslothContributor) -> None:
        cli = unsloth.get_cli()
        submit = next(cmd for cmd in cli.registered_commands if cmd.name == "submit")
        assert submit.help is not None
        assert "UnslothJobInput" in submit.help
        assert "nemo customization unsloth explain" in submit.help
