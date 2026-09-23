# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtering the workspace Evaluations list by the denormalized agent/model name facets.

The scalar ``agent_name``/``agent_version``/``model_name`` params match evaluations whose stored
list facet *contains* the value; the endpoint rewrites the parsed equality into a ``$contains``
membership match against the entity store (mirroring ``experiment_id`` over ``experiment_ids``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.intake.api.v2.experiments.dependencies import get_evaluation_rollup_repository
from nhx.intake.api.v2.experiments.endpoints import _rewrite_facet_filters
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.entities.experiments import Experiment
from nhx.intake.service import IntakeService
from nhx.testing import ClientContext, create_test_client

EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"

_FACET_FIELDS = ("data.agent_names", "data.agent_versions", "data.model_names")


def _eq(field: str, value: str) -> ComparisonOperation:
    return ComparisonOperation(operator=FilterOperator.EQ, field=field, value=value)


def test_rewrite_converts_facet_equality_to_contains() -> None:
    for field in _FACET_FIELDS:
        rewritten = _rewrite_facet_filters(_eq(field, "x"))
        assert isinstance(rewritten, ComparisonOperation)
        assert rewritten.operator == FilterOperator.CONTAINS
        assert rewritten.field == field
        assert rewritten.value == "x"


def test_rewrite_leaves_non_facet_and_non_eq_untouched() -> None:
    # A different field keeps its equality.
    assert _rewrite_facet_filters(_eq("data.name", "x")).operator == FilterOperator.EQ
    # A non-equality on a facet field is left alone (only the scalar-param equality is a membership match).
    already = ComparisonOperation(operator=FilterOperator.CONTAINS, field="data.agent_names", value="x")
    assert _rewrite_facet_filters(already).operator == FilterOperator.CONTAINS


def test_rewrite_recurses_through_logical_tree() -> None:
    op = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[_eq("data.agent_names", "a"), _eq("data.name", "n")],
    )
    rewritten = _rewrite_facet_filters(op)
    assert isinstance(rewritten, LogicalOperation)
    by_field = {o.field: o.operator for o in rewritten.operations}
    assert by_field["data.agent_names"] == FilterOperator.CONTAINS
    assert by_field["data.name"] == FilterOperator.EQ


def test_rewrite_none_passthrough() -> None:
    assert _rewrite_facet_filters(None) is None


@pytest.mark.asyncio
async def test_filter_by_agent_name_returns_only_matching_evaluations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(IntakeService, "is_ready", AsyncMock(return_value=True))
    intake_config = IntakeConfig(
        clickhouse_config=ClickHouseConfig(url="http://127.0.0.1:1"),
    )
    with create_test_client(
        IntakeService,
        client_type=ClientContext,
        dependency_overrides={get_evaluation_rollup_repository: lambda: None},
        service_configs={IntakeService: intake_config},
    ) as ctx:
        tc = ctx.test_client
        group = tc.post(EXPERIMENTS, json={"name": "facet-grp"}).json()
        for name in ("eval-x", "eval-y"):
            created = tc.post(
                EVALUATIONS,
                json={"name": name, "experiment_group_id": group["id"], "dataset_name": "ds"},
            )
            assert created.status_code == 201, created.text

        # Seed the system-managed facets directly (what the background refresher would write after ingest).
        for name, agents, models in (
            ("eval-x", ["agent-x"], ["provider/model-a"]),
            ("eval-y", ["agent-y"], ["provider/model-b"]),
        ):
            entity = await ctx.entity_client.get(Experiment, name=name, workspace="default")
            entity.agent_names = agents
            entity.model_names = models
            await ctx.entity_client.update(entity)

        # Filter by an agent name: only the evaluation whose facet list contains it comes back.
        resp = tc.get(EVALUATIONS, params={"filter[agent_name]": "agent-x"})
        assert resp.status_code == 200, resp.text
        assert {e["name"] for e in resp.json()["data"]} == {"eval-x"}

        # A model-name filter is independent and scopes to the other evaluation.
        resp = tc.get(EVALUATIONS, params={"filter[model_name]": "provider/model-b"})
        assert resp.status_code == 200, resp.text
        assert {e["name"] for e in resp.json()["data"]} == {"eval-y"}

        # A name nobody observed matches nothing.
        resp = tc.get(EVALUATIONS, params={"filter[agent_name]": "agent-z"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == []
