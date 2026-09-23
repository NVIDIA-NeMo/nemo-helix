# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment group default sort: string storage/validation and the sort helper.

``default_sort`` is a ``sort``-param string stored on the group: a comma-separated, ordered list of
fields (e.g. ``-evaluators.reward.mean,cost_usd.mean`` or ``-created_at``) where the first is the
primary sort and the rest break ties — any field the evaluations list can sort by. The client reads it
and applies it as the list ``sort`` param; the list endpoint itself never consults it. The
``_sort_evaluations`` helper is multi-key capable (pinned-first + tiebreaks), matching the multi-field
default.
"""

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from nhx.intake.api.v2.experiments.endpoints import _sort_evaluations, _validate_default_sort
from nhx.intake.api.v2.experiments.schemas import EvaluationResponse, EvaluatorAggregate

EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _exp(
    name: str,
    *,
    cost: float | None = None,
    latency: float | None = None,
    pinned: bool = False,
    created: datetime | None = None,
) -> EvaluationResponse:
    return EvaluationResponse(
        id=name,
        name=name,
        workspace="default",
        experiment_ids=["grp"],
        dataset_name="ds",
        cost_usd=EvaluatorAggregate(mean=cost) if cost is not None else None,
        latency_ms=EvaluatorAggregate(mean=latency) if latency is not None else None,
        pinned_at=datetime(2026, 1, 1, tzinfo=timezone.utc) if pinned else None,
        created_at=created,
    )


def _names(responses: list[EvaluationResponse]) -> list[str]:
    return [r.name for r in responses]


# ----------------------------- multi-key sort -----------------------------


def test_multi_key_primary_then_tiebreak() -> None:
    # Primary cost asc; ties on cost broken by latency asc.
    rows = [_exp("a", cost=1.0, latency=200), _exp("b", cost=1.0, latency=100), _exp("c", cost=0.5, latency=999)]
    ordered = _sort_evaluations(rows, keys=[("cost_usd.mean", False), ("latency_ms.mean", False)])
    assert _names(ordered) == ["c", "b", "a"]


def test_pinned_floats_to_top() -> None:
    rows = [_exp("a", cost=0.1), _exp("pinned", cost=0.9, pinned=True), _exp("b", cost=0.5)]
    ordered = _sort_evaluations(rows, keys=[("cost_usd.mean", False)], pinned_first=True)
    # Pinned first regardless of metric; unpinned follow in cost order.
    assert _names(ordered) == ["pinned", "a", "b"]


def test_falls_back_to_created_at_when_sorted_metric_missing() -> None:
    rows = [
        _exp("old", created=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _exp("new", created=datetime(2026, 6, 1, tzinfo=timezone.utc)),
    ]
    # No cost rollup on either -> the appended -created_at key decides (newest first).
    ordered = _sort_evaluations(rows, keys=[("cost_usd.mean", False), ("created_at", True)])
    assert _names(ordered) == ["new", "old"]


# ----------------------------- default-sort validation -----------------------------


def test_validate_default_sort_accepts_any_sortable_field() -> None:
    # Entity columns and rollup-metric paths, ascending or descending ('-'), are all valid.
    for value in (
        "name",
        "-created_at",
        "cost_usd.mean",
        "-latency_ms.p95",
        "run_count",
        "-evaluators.harbor.verifier.mean",
        # Multi-field: a comma-separated, ordered list where each token is independently validated.
        "-evaluators.reward.mean,cost_usd.mean",
        "cost_usd.mean,latency_ms.mean,-created_at",
        " -cost_usd.mean , latency_ms.mean ",  # surrounding whitespace tolerated
    ):
        _validate_default_sort(value)
    _validate_default_sort(None)  # absent default sort is fine


def test_validate_default_sort_rejects_unsortable_fields() -> None:
    for value in (
        "bogus",
        "-description",
        "cost_usd.bogus",
        "evaluators.reward",
        # A single bad token anywhere in a multi-field list fails the whole value.
        "cost_usd.mean,bogus",
        "-evaluators.reward.mean,cost_usd.bogus",
        # An empty or all-blank value is not a usable sort and must be rejected, not silently accepted.
        "",
        ",",
        "  ,  ",
    ):
        with pytest.raises(HTTPException) as exc:
            _validate_default_sort(value)
        assert exc.value.status_code == 400


def test_entity_coerces_legacy_default_sort_to_created_at() -> None:
    # Schema-on-read: rows persisted before default_sort was a non-null string stored it as null (or,
    # earlier, a SortCriterion list). Both must deserialize to the default, not raise.
    from nhx.intake.entities.experiments import ExperimentGroup

    for stored in (None, [{"field": "cost_usd.mean", "direction": "asc"}]):
        group = ExperimentGroup.model_validate({"name": "g", "workspace": "default", "default_sort": stored})
        assert group.default_sort == "-created_at"


def test_entity_coerces_legacy_non_string_metadata_values() -> None:
    # Schema-on-read: metadata tightened from dict[str, Any] to dict[str, str]. Rows persisted with
    # non-string values (the old type) must deserialize — stringified, JSON-encoding structured values.
    from nhx.intake.entities.experiments import Experiment, ExperimentGroup

    exp = Experiment.model_validate(
        {
            "name": "e",
            "workspace": "default",
            "experiment_group_id": "grp",
            "dataset_name": "ds",
            "metadata": {"attempt": 1, "config": {"lr": 0.1}, "ok": "yes"},
        }
    )
    assert exp.metadata == {"attempt": "1", "config": '{"lr": 0.1}', "ok": "yes"}

    group = ExperimentGroup.model_validate({"name": "g", "workspace": "default", "metadata": {"n": 3}})
    assert group.metadata == {"n": "3"}


# ----------------------------- endpoint wiring -----------------------------


def test_create_group_defaults_sort_to_created_at(client: TestClient) -> None:
    resp = client.post(EXPERIMENTS, json={"name": "g-default"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["default_sort"] == "-created_at"


def test_create_group_with_default_sort_round_trips(client: TestClient) -> None:
    for i, value in enumerate(
        ("-cost_usd.mean", "-created_at", "-evaluators.reward.mean,cost_usd.mean"),
    ):
        resp = client.post(EXPERIMENTS, json={"name": f"g-sort-{i}", "default_sort": value})
        assert resp.status_code == 201, resp.text
        assert resp.json()["default_sort"] == value


def test_create_group_rejects_unsortable_field(client: TestClient) -> None:
    for i, value in enumerate(("bogus", "cost_usd.bogus")):
        resp = client.post(EXPERIMENTS, json={"name": f"g-bad-{i}", "default_sort": value})
        assert resp.status_code == 400, resp.text


def test_default_order_floats_pinned_first(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "g-pin"}).json()
    for name in ("exp-a", "exp-b"):
        created = client.post(
            EVALUATIONS, json={"name": name, "experiment_group_id": group["id"], "dataset_name": "ds"}
        )
        assert created.status_code == 201, created.text
    # Pin the OLDER evaluation (exp-a): by the -created_at fallback it would sort LAST, so seeing it
    # first proves pinned-first actually overrides the fallback rather than coinciding with newest-first.
    assert client.post(f"{EVALUATIONS}/exp-a/pin").status_code == 200
    # No explicit sort -> default path floats pinned to top (entity-only, no rollups needed).
    listed = client.get(EVALUATIONS, params={"filter[experiment_group_id]": group["id"]})
    assert listed.status_code == 200, listed.text
    assert [r["name"] for r in listed.json()["data"]] == ["exp-a", "exp-b"]
