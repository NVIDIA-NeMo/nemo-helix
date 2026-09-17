# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from fastapi import FastAPI
from nemo_automodel_plugin.contributor import AutomodelContributor


@runtime_checkable
class _RouteWithPath(Protocol):
    path: str


@runtime_checkable
class _RouteWithEffectiveCandidates(Protocol):
    def effective_candidates(self) -> Iterable[object]: ...


def _route_paths(app: FastAPI) -> set[str]:
    """Collect all route paths, compatible with FastAPI 0.138+ _IncludedRouter."""
    paths: set[str] = set()
    queue: list[object] = list(app.routes)
    while queue:
        route = queue.pop()
        if isinstance(route, _RouteWithPath):
            paths.add(route.path)
        if isinstance(route, _RouteWithEffectiveCandidates):
            queue.extend(route.effective_candidates())
    return paths


def test_contributor_mounts_job_collection() -> None:
    contributor = AutomodelContributor()
    app = FastAPI()
    for spec in contributor.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    paths = _route_paths(app)
    assert "/v2/workspaces/{workspace}/automodel/jobs" in paths


def test_contributor_get_cli_exposes_flat_verbs() -> None:
    import typer

    cli = AutomodelContributor().get_cli()
    assert isinstance(cli, typer.Typer)
    assert cli.info.name == "automodel"
    assert not any(g.name == "jobs" for g in cli.registered_groups)
    assert {cmd.name for cmd in cli.registered_commands} == {"submit", "explain"}


def test_contributor_exposes_sdk_resources() -> None:
    from nemo_automodel_plugin.sdk.resources import AsyncAutomodelCustomization, AutomodelCustomization

    sdk = AutomodelContributor().get_sdk_resources()
    assert sdk is not None
    assert sdk.sync_resource is AutomodelCustomization
    assert sdk.async_resource is AsyncAutomodelCustomization


def test_cli_summary_states_what_it_trains_and_where_it_runs() -> None:
    summary = AutomodelContributor().get_cli_summary()
    assert summary is not None
    assert "SFT" in summary.trains and "LoRA" in summary.trains
    assert "Multi-node needs kubernetes_job or volcano_job." in summary.runs_on
    assert summary.command == "nemo customization automodel submit job.json"


def test_summary_and_help_agree_on_multi_node_backends() -> None:
    """The overview must not exclude a backend the detailed help accepts."""
    contributor = AutomodelContributor()
    summary = contributor.get_cli_summary()
    assert summary is not None
    for backend in ("kubernetes_job", "volcano_job"):
        assert backend in summary.runs_on, backend
        assert backend in contributor.cli_help, backend


def test_cli_summary_fits_the_rendered_width() -> None:
    """The router prints the summary through an 80-column Rich console."""
    summary = AutomodelContributor().get_cli_summary()
    assert summary is not None
    rendered = summary.render("automodel")
    assert [line for line in rendered.splitlines() if len(line) > 80] == []


def test_backend_help_goes_deeper_than_the_top_level_summary() -> None:
    contributor = AutomodelContributor()
    summary = contributor.get_cli_summary()
    assert summary is not None
    summary_text = summary.render(contributor.name)
    help_text = contributor.cli_help

    assert len(help_text) > len(summary_text)
    # Job JSON fields and schema names belong on the backend, not in the overview.
    for detail in ("AutomodelJobInput", "global_batch_size", "num_nodes", "explain"):
        assert detail in help_text, detail
        assert detail not in summary_text, detail


def test_submit_help_explains_the_job_json() -> None:
    cli = AutomodelContributor().get_cli()
    submit = next(cmd for cmd in cli.registered_commands if cmd.name == "submit")
    assert submit.help is not None
    assert "AutomodelJobInput" in submit.help
    assert "nemo customization automodel explain" in submit.help
