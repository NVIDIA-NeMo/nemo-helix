# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD tests for the Evaluations and Experiments endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.intake.api.v2.experiments.dependencies import get_evaluation_rollup_repository

EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"
# The deprecated pre-rename path; kept as a hidden alias for backwards compatibility.
LEGACY_GROUPS = "/apis/intake/v2/workspaces/default/experiment-groups"
EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"


def _evaluation_body(*, experiment_group_id: str, **overrides: Any) -> dict:
    body = {
        "name": "terminal-bench-2_claude-code_opus_baseline",
        "experiment_group_id": experiment_group_id,
        "dataset_name": "terminal-bench-2",
        "dataset_version": "v1",
        "source_link": "https://example.com/evaluations/tb2-baseline",
        "metadata": {"job_name": "tb2-baseline"},
    }
    body.update(overrides)
    return body


def _create_group(client: TestClient, name: str = "default-test-group") -> dict:
    """Create or fetch an experiment group; returns the JSON body."""
    response = client.post(EXPERIMENTS, json={"name": name})
    if response.status_code == 409:
        response = client.get(f"{EXPERIMENTS}/{name}")
    response.raise_for_status()
    return response.json()


def test_legacy_experiment_groups_alias_is_backwards_compatible(client: TestClient) -> None:
    """The pre-rename `/experiment-groups` path still works and shares the `/experiments` handlers."""
    # Create via the legacy path; read it back via the canonical path.
    created = client.post(LEGACY_GROUPS, json={"name": "legacy-alias-exp"})
    assert created.status_code == 201, created.text
    assert client.get(f"{EXPERIMENTS}/legacy-alias-exp").status_code == 200

    # Create via the canonical path; read it back via the legacy path.
    assert client.post(EXPERIMENTS, json={"name": "canonical-exp"}).status_code == 201
    legacy_get = client.get(f"{LEGACY_GROUPS}/canonical-exp")
    assert legacy_get.status_code == 200, legacy_get.text
    assert legacy_get.json()["name"] == "canonical-exp"


def test_experiment_group_crud(client: TestClient) -> None:
    created = client.post(EXPERIMENTS, json={"name": "tb2-routing-research", "description": "routing sweep"})
    assert created.status_code == 201, created.text
    group = created.json()
    assert group["name"] == "tb2-routing-research"
    assert group["description"] == "routing sweep"
    assert group["id"]

    # Duplicate name conflicts.
    duplicate = client.post(EXPERIMENTS, json={"name": "tb2-routing-research"})
    assert duplicate.status_code == 409

    fetched = client.get(f"{EXPERIMENTS}/tb2-routing-research")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == group["id"]

    listed = client.get(EXPERIMENTS)
    assert listed.status_code == 200
    assert any(g["name"] == "tb2-routing-research" for g in listed.json()["data"])

    deleted = client.delete(f"{EXPERIMENTS}/tb2-routing-research")
    assert deleted.status_code == 204
    missing = client.get(f"{EXPERIMENTS}/tb2-routing-research")
    assert missing.status_code == 404


def test_filter_groups_by_insight_id(client: TestClient) -> None:
    """The groups list can be filtered server-side by the seeding insight id."""
    seeded = client.post(EXPERIMENTS, json={"name": "seeded-group", "insight_id": "insight-abc"})
    assert seeded.status_code == 201, seeded.text
    other = client.post(EXPERIMENTS, json={"name": "unseeded-group"})
    assert other.status_code == 201, other.text

    listed = client.get(EXPERIMENTS, params={"filter[insight_id]": "insight-abc"})
    assert listed.status_code == 200, listed.text
    names = {g["name"] for g in listed.json()["data"]}
    assert names == {"seeded-group"}


def test_experiment_group_update_description(client: TestClient) -> None:
    client.post(EXPERIMENTS, json={"name": "grp", "description": "old"})
    updated = client.put(f"{EXPERIMENTS}/grp", json={"name": "grp", "description": "new"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["description"] == "new"
    # Renaming via PUT is rejected.
    renamed = client.put(f"{EXPERIMENTS}/grp", json={"name": "renamed"})
    assert renamed.status_code == 409
    missing = client.put(f"{EXPERIMENTS}/missing", json={"name": "missing"})
    assert missing.status_code == 404


def test_experiment_flags_round_trip_and_preserve_omitted_values(client: TestClient) -> None:
    created = client.post(
        EXPERIMENTS,
        json={
            "name": "flagged-experiment",
            "default_sort": "name",
            "is_favorite": True,
            "show_evaluations_over_time": True,
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["is_favorite"] is True
    assert created.json()["show_evaluations_over_time"] is True

    # Omitting defaulted settings from a full update must not reset values that were already customized.
    updated = client.put(
        f"{EXPERIMENTS}/flagged-experiment",
        json={"name": "flagged-experiment", "description": "updated by an older client"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["default_sort"] == "name"
    assert updated.json()["is_favorite"] is True
    assert updated.json()["show_evaluations_over_time"] is True

    cleared = client.put(
        f"{EXPERIMENTS}/flagged-experiment",
        json={"name": "flagged-experiment", "is_favorite": False, "show_evaluations_over_time": False},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["is_favorite"] is False
    assert cleared.json()["show_evaluations_over_time"] is False


def test_filter_experiments_by_new_flags(client: TestClient) -> None:
    assert client.post(EXPERIMENTS, json={"name": "flagged", "is_favorite": True}).status_code == 201
    assert client.post(EXPERIMENTS, json={"name": "timeline", "show_evaluations_over_time": True}).status_code == 201
    assert client.post(EXPERIMENTS, json={"name": "plain"}).status_code == 201

    favorites = client.get(EXPERIMENTS, params={"filter[is_favorite]": "true"})
    assert favorites.status_code == 200, favorites.text
    assert {item["name"] for item in favorites.json()["data"]} == {"flagged"}

    not_timeline = client.get(EXPERIMENTS, params={"filter[show_evaluations_over_time]": "false"})
    assert not_timeline.status_code == 200, not_timeline.text
    names = {item["name"] for item in not_timeline.json()["data"]}
    assert "plain" in names
    assert "timeline" not in names


def test_experiment_group_pareto_defaults_and_round_trips(client: TestClient) -> None:
    # Omitting pareto defaults to cost (x) vs latency (y), so the chart always has something to render.
    created = client.post(EXPERIMENTS, json={"name": "pareto-cfg"})
    assert created.status_code == 201, created.text
    assert created.json()["pareto"] == {"x_metric": "cost_usd", "y_metric": "latency_ms"}
    assert client.get(f"{EXPERIMENTS}/pareto-cfg").json()["pareto"] == {
        "x_metric": "cost_usd",
        "y_metric": "latency_ms",
    }

    # A custom selection round-trips through PUT.
    updated = client.put(
        f"{EXPERIMENTS}/pareto-cfg",
        json={"name": "pareto-cfg", "pareto": {"x_metric": "cost_usd", "y_metric": "evaluators.reward"}},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["pareto"] == {"x_metric": "cost_usd", "y_metric": "evaluators.reward"}

    # An update that omits pareto preserves the saved axes rather than resetting them to the default.
    preserved = client.put(f"{EXPERIMENTS}/pareto-cfg", json={"name": "pareto-cfg", "summary": "unrelated edit"})
    assert preserved.status_code == 200, preserved.text
    assert preserved.json()["pareto"] == {"x_metric": "cost_usd", "y_metric": "evaluators.reward"}


def test_experiment_group_pareto_rejects_unknown_metric(client: TestClient) -> None:
    # Studio can only plot cost_usd, latency_ms, or evaluators.<name>; anything else is rejected so it
    # can't be persisted as an unusable chart default.
    rejected = client.post(
        EXPERIMENTS, json={"name": "bad-pareto", "pareto": {"x_metric": "made_up", "y_metric": "latency_ms"}}
    )
    assert rejected.status_code == 422, rejected.text
    # A dynamic evaluator metric is allowed.
    ok = client.post(
        EXPERIMENTS, json={"name": "ok-pareto", "pareto": {"x_metric": "evaluators.safety", "y_metric": "cost_usd"}}
    )
    assert ok.status_code == 201, ok.text


def test_evaluation_update_moves_between_groups_and_edits(client: TestClient) -> None:
    group_a = client.post(EXPERIMENTS, json={"name": "grp-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "grp-b"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-a", experiment_group_id=group_a["id"]))

    # Move the evaluation from group_a to group_b and edit its description.
    body = _evaluation_body(name="exp-a", experiment_group_id=group_b["id"], description="looks good")
    updated = client.put(f"{EVALUATIONS}/exp-a", json=body)
    assert updated.status_code == 200, updated.text
    assert updated.json()["experiment_group_id"] == group_b["id"]
    assert updated.json()["description"] == "looks good"


def test_evaluation_update_rejects_immutable_change(client: TestClient) -> None:
    group = _create_group(client)
    client.post(
        EVALUATIONS,
        json=_evaluation_body(name="exp-a", dataset_name="tb2", experiment_group_id=group["id"]),
    )
    changed = _evaluation_body(name="exp-a", dataset_name="tb3", experiment_group_id=group["id"])
    resp = client.put(f"{EVALUATIONS}/exp-a", json=changed)
    assert resp.status_code == 409, resp.text
    assert "dataset_name" in resp.json()["detail"]
    missing = client.put(
        f"{EVALUATIONS}/missing",
        json=_evaluation_body(name="missing", experiment_group_id=group["id"]),
    )
    assert missing.status_code == 404


def test_evaluation_crud_and_empty_rollups(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "grp"}).json()

    created = client.post(EVALUATIONS, json=_evaluation_body(experiment_group_id=group["id"]))
    assert created.status_code == 201, created.text
    exp = created.json()
    assert exp["name"] == "terminal-bench-2_claude-code_opus_baseline"
    assert exp["experiment_group_id"] == group["id"]
    assert exp["dataset_name"] == "terminal-bench-2"
    assert exp["source_link"] == "https://example.com/evaluations/tb2-baseline"
    assert exp["metadata"] == {"job_name": "tb2-baseline"}

    # Rollups exist on the read model but are empty until ClickHouse hydration lands.
    assert exp["evaluator_names"] == []
    assert exp["model_names"] == []
    assert exp["agent_names"] == []
    assert exp["agent_versions"] == []
    assert exp["aggregate_scores"] is None
    assert exp["run_count"] == 0


def test_experiment_selects_one_baseline_evaluation(client: TestClient) -> None:
    experiment = client.post(EXPERIMENTS, json={"name": "baseline-experiment"}).json()
    first = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="first-baseline", experiment_group_id=experiment["id"]),
    ).json()
    second = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="second-baseline", experiment_group_id=experiment["id"]),
    ).json()

    selected = client.put(
        f"{EXPERIMENTS}/baseline-experiment",
        json={"name": "baseline-experiment", "baseline_evaluation_name": first["name"]},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["baseline_evaluation_name"] == first["name"]

    # Replacing the pointer leaves only one selected baseline by construction.
    replaced = client.put(
        f"{EXPERIMENTS}/baseline-experiment",
        json={"name": "baseline-experiment", "baseline_evaluation_name": second["name"]},
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["baseline_evaluation_name"] == second["name"]


def test_shared_evaluation_can_be_baseline_for_a_subset_of_experiments(client: TestClient) -> None:
    experiment_a = client.post(EXPERIMENTS, json={"name": "baseline-a"}).json()
    experiment_b = client.post(EXPERIMENTS, json={"name": "baseline-b"}).json()
    body = _evaluation_body(name="shared-baseline", experiment_group_id="placeholder")
    body.pop("experiment_group_id")
    body["experiment_ids"] = [experiment_a["id"], experiment_b["id"]]
    evaluation = client.post(EVALUATIONS, json=body).json()

    selected = client.put(
        f"{EXPERIMENTS}/baseline-a",
        json={"name": "baseline-a", "baseline_evaluation_name": evaluation["name"]},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["baseline_evaluation_name"] == evaluation["name"]
    assert client.get(f"{EXPERIMENTS}/baseline-b").json()["baseline_evaluation_name"] is None

    also_selected = client.put(
        f"{EXPERIMENTS}/baseline-b",
        json={"name": "baseline-b", "baseline_evaluation_name": evaluation["name"]},
    )
    assert also_selected.status_code == 200, also_selected.text
    assert also_selected.json()["baseline_evaluation_name"] == evaluation["name"]
    assert client.get(f"{EXPERIMENTS}/baseline-a").json()["baseline_evaluation_name"] == evaluation["name"]


def test_experiment_rejects_invalid_baseline_evaluation(client: TestClient) -> None:
    on_create = client.post(
        EXPERIMENTS,
        json={"name": "baseline-too-early", "baseline_evaluation_name": "evaluation-nope"},
    )
    assert on_create.status_code == 422, on_create.text

    experiment = client.post(EXPERIMENTS, json={"name": "baseline-owner"}).json()
    other = client.post(EXPERIMENTS, json={"name": "baseline-other"}).json()
    non_member = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="non-member", experiment_group_id=other["id"]),
    ).json()

    missing = client.put(
        f"{EXPERIMENTS}/baseline-owner",
        json={"name": "baseline-owner", "baseline_evaluation_name": "evaluation-nope"},
    )
    assert missing.status_code == 400, missing.text

    wrong_experiment = client.put(
        f"{EXPERIMENTS}/baseline-owner",
        json={"name": "baseline-owner", "baseline_evaluation_name": non_member["name"]},
    )
    assert wrong_experiment.status_code == 400, wrong_experiment.text
    assert experiment["baseline_evaluation_name"] is None


def test_baseline_reference_blocks_membership_removal_and_deletion(client: TestClient) -> None:
    experiment = client.post(EXPERIMENTS, json={"name": "protected-baseline"}).json()
    other = client.post(EXPERIMENTS, json={"name": "protected-other"}).json()
    body = _evaluation_body(name="protected-evaluation", experiment_group_id="placeholder")
    body.pop("experiment_group_id")
    body["experiment_ids"] = [experiment["id"], other["id"]]
    evaluation = client.post(EVALUATIONS, json=body).json()
    selected = client.put(
        f"{EXPERIMENTS}/protected-baseline",
        json={"name": "protected-baseline", "baseline_evaluation_name": evaluation["name"]},
    )
    assert selected.status_code == 200, selected.text

    removed = client.patch(
        f"{EVALUATIONS}/protected-evaluation",
        json={"experiment_ids": [other["id"]]},
    )
    assert removed.status_code == 409, removed.text
    assert client.delete(f"{EVALUATIONS}/protected-evaluation").status_code == 409

    cleared = client.put(
        f"{EXPERIMENTS}/protected-baseline",
        json={"name": "protected-baseline", "baseline_evaluation_name": None},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["baseline_evaluation_name"] is None
    removed_after_clear = client.patch(
        f"{EVALUATIONS}/protected-evaluation",
        json={"experiment_ids": [other["id"]]},
    )
    assert removed_after_clear.status_code == 200, removed_after_clear.text
    assert client.delete(f"{EVALUATIONS}/protected-evaluation").status_code == 204


def test_filter_experiments_by_baseline_evaluation_name(client: TestClient) -> None:
    experiment = client.post(EXPERIMENTS, json={"name": "filter-baseline"}).json()
    evaluation = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="filter-baseline-evaluation", experiment_group_id=experiment["id"]),
    ).json()
    assert (
        client.put(
            f"{EXPERIMENTS}/filter-baseline",
            json={"name": "filter-baseline", "baseline_evaluation_name": evaluation["name"]},
        ).status_code
        == 200
    )

    filtered = client.get(EXPERIMENTS, params={"filter[baseline_evaluation_name]": evaluation["name"]})
    assert filtered.status_code == 200, filtered.text
    assert {item["name"] for item in filtered.json()["data"]} == {"filter-baseline"}


def test_evaluation_read_degrades_when_rollup_hydration_fails(client: TestClient) -> None:
    class FailingRollupRepository:
        async def get_rollups(self, *, workspace: str, evaluation_names: list[str]) -> dict:
            raise RuntimeError("clickhouse unavailable")

    group = _create_group(client)
    app = cast(FastAPI, client.app)
    app.dependency_overrides[get_evaluation_rollup_repository] = lambda: FailingRollupRepository()
    try:
        created = client.post(
            EVALUATIONS,
            json=_evaluation_body(name="exp-rollup-fails", experiment_group_id=group["id"]),
        )
        assert created.status_code == 201, created.text

        fetched = client.get(f"{EVALUATIONS}/exp-rollup-fails")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["name"] == "exp-rollup-fails"
        assert fetched.json()["run_count"] == 0
        assert fetched.json()["aggregate_scores"] is None
    finally:
        app.dependency_overrides.pop(get_evaluation_rollup_repository, None)


def test_evaluation_create_rejects_missing_group_id(client: TestClient) -> None:
    body = _evaluation_body(name="exp-no-group", experiment_group_id="placeholder")
    body.pop("experiment_group_id")
    response = client.post(EVALUATIONS, json=body)
    assert response.status_code == 422, response.text


def test_evaluation_create_rejects_unknown_group_id(client: TestClient) -> None:
    response = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="exp-bad-group", experiment_group_id="experiment_group-does-not-exist"),
    )
    assert response.status_code == 400, response.text
    assert "must be created before" in response.json()["detail"]


def test_delete_group_cascades_to_evaluations(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "doomed-group"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-doomed-1", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-doomed-2", experiment_group_id=group["id"]))

    deleted = client.delete(f"{EXPERIMENTS}/doomed-group")
    assert deleted.status_code == 204, deleted.text

    # Both child evaluations are gone with the group.
    for child_name in ("exp-doomed-1", "exp-doomed-2"):
        missing = client.get(f"{EVALUATIONS}/{child_name}")
        assert missing.status_code == 404


def test_evaluation_conflict_and_not_found(client: TestClient) -> None:
    group = _create_group(client)
    created = client.post(EVALUATIONS, json=_evaluation_body(experiment_group_id=group["id"]))
    assert created.status_code == 201
    duplicate = client.post(EVALUATIONS, json=_evaluation_body(experiment_group_id=group["id"]))
    assert duplicate.status_code == 409
    missing = client.get(f"{EVALUATIONS}/does-not-exist")
    assert missing.status_code == 404
    missing_delete = client.delete(f"{EVALUATIONS}/does-not-exist")
    assert missing_delete.status_code == 404


def test_evaluation_list_and_scope_to_group(client: TestClient) -> None:
    group_a = client.post(EXPERIMENTS, json={"name": "grp-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "grp-b"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-a", experiment_group_id=group_a["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-b", experiment_group_id=group_b["id"]))

    all_resp = client.get(EVALUATIONS)
    assert all_resp.status_code == 200
    names = {e["name"] for e in all_resp.json()["data"]}
    assert {"exp-a", "exp-b"} <= names

    in_group_a = client.get(EVALUATIONS, params={"filter[experiment_group_id]": group_a["id"]})
    assert in_group_a.status_code == 200
    assert {e["name"] for e in in_group_a.json()["data"]} == {"exp-a"}

    deleted = client.delete(f"{EVALUATIONS}/exp-a")
    assert deleted.status_code == 204
    missing = client.get(f"{EVALUATIONS}/exp-a")
    assert missing.status_code == 404


def test_evaluation_filter_experiment_id_matches_deprecated_alias(client: TestClient) -> None:
    """The canonical `experiment_id` filter returns the same set as the deprecated `experiment_group_id`,
    both resolving membership over the evaluation's `experiment_ids` list."""
    group = client.post(EXPERIMENTS, json={"name": "grp-canon"}).json()
    other = client.post(EXPERIMENTS, json={"name": "grp-other"}).json()
    # Membership is set via the canonical `experiment_ids` write field (no legacy scalar).
    client.post(EVALUATIONS, json={"name": "in-group", "experiment_ids": [group["id"]], "dataset_name": "ds"})
    client.post(EVALUATIONS, json={"name": "in-other", "experiment_ids": [other["id"]], "dataset_name": "ds"})

    by_canonical = client.get(EVALUATIONS, params={"filter[experiment_id]": group["id"]})
    assert by_canonical.status_code == 200, by_canonical.text
    canonical_names = {e["name"] for e in by_canonical.json()["data"]}
    assert canonical_names == {"in-group"}

    by_deprecated = client.get(EVALUATIONS, params={"filter[experiment_group_id]": group["id"]})
    assert by_deprecated.status_code == 200, by_deprecated.text
    deprecated_names = {e["name"] for e in by_deprecated.json()["data"]}

    # The canonical key and the deprecated alias resolve to the same membership set.
    assert canonical_names == deprecated_names


def test_evaluation_filter_by_dataset_version(client: TestClient) -> None:
    group = _create_group(client)
    client.post(
        EVALUATIONS,
        json=_evaluation_body(name="exp-v1", dataset_version="v1", experiment_group_id=group["id"]),
    )
    client.post(
        EVALUATIONS,
        json=_evaluation_body(name="exp-v2", dataset_version="v1", experiment_group_id=group["id"]),
    )
    client.post(
        EVALUATIONS,
        json=_evaluation_body(name="exp-v3", dataset_version="v2", experiment_group_id=group["id"]),
    )

    by_dataset_version = client.get(EVALUATIONS, params={"filter[dataset_version]": "v1"})
    assert by_dataset_version.status_code == 200
    assert {e["name"] for e in by_dataset_version.json()["data"]} == {"exp-v1", "exp-v2"}


def test_evaluation_filter_by_created_at_range(client: TestClient) -> None:
    from datetime import datetime, timedelta, timezone

    group = _create_group(client)
    before_create = datetime.now(timezone.utc) - timedelta(seconds=2)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-recent", experiment_group_id=group["id"]))
    after_create = datetime.now(timezone.utc) + timedelta(seconds=2)

    # Range that brackets the create timestamp -> the evaluation is included.
    in_range = client.get(
        EVALUATIONS,
        params={
            "filter[name]": "exp-recent",
            "filter[created_at][$gte]": before_create.isoformat(),
            "filter[created_at][$lte]": after_create.isoformat(),
        },
    )
    assert in_range.status_code == 200, in_range.text
    assert {e["name"] for e in in_range.json()["data"]} == {"exp-recent"}

    # Range entirely after the create timestamp -> excluded.
    future_only = client.get(
        EVALUATIONS,
        params={
            "filter[name]": "exp-recent",
            "filter[created_at][$gte]": (after_create + timedelta(hours=1)).isoformat(),
        },
    )
    assert future_only.status_code == 200, future_only.text
    assert future_only.json()["data"] == []


def test_evaluation_filter_by_created_by(client: TestClient) -> None:
    # The test harness doesn't set an authenticated principal, so we only verify the filter
    # parameter is accepted and routed through the entity store without erroring.
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-cb", experiment_group_id=group["id"]))
    response = client.get(EVALUATIONS, params={"filter[created_by]": "someone@example.com"})
    assert response.status_code == 200, response.text


def test_soft_delete_frees_name_for_reuse(client: TestClient) -> None:
    group = _create_group(client)
    first = client.post(EVALUATIONS, json=_evaluation_body(name="reusable", experiment_group_id=group["id"]))
    assert first.status_code == 201, first.text

    deleted = client.delete(f"{EVALUATIONS}/reusable")
    assert deleted.status_code == 204, deleted.text

    # The original name is now free; a new evaluation can claim it.
    second = client.post(EVALUATIONS, json=_evaluation_body(name="reusable", experiment_group_id=group["id"]))
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]


def test_list_hides_soft_deleted_by_default(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-live", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-gone", experiment_group_id=group["id"]))
    client.delete(f"{EVALUATIONS}/exp-gone")

    listed = client.get(EVALUATIONS)
    assert listed.status_code == 200
    names = {e["name"] for e in listed.json()["data"]}
    assert "exp-live" in names
    assert "exp-gone" not in names
    assert not any(name.startswith("exp-gone-deleted-") for name in names)


def test_filter_is_deleted_true_returns_only_deleted(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-still-here", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-trash", experiment_group_id=group["id"]))
    client.delete(f"{EVALUATIONS}/exp-trash")

    response = client.get(EVALUATIONS, params={"filter[is_deleted]": "true"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    # Live evaluations are excluded; only deleted rows (with mangled names) appear. The
    # response body intentionally omits ``is_deleted``; the filter context and mangled name
    # are the signal that these are trash-bin rows.
    assert any(e["name"].startswith("exp-trash-deleted-") for e in data)
    assert not any(e["name"] == "exp-still-here" for e in data)


def test_group_soft_delete_cascades_and_frees_names(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "doomed-group-v2"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="child-1", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="child-2", experiment_group_id=group["id"]))

    deleted = client.delete(f"{EXPERIMENTS}/doomed-group-v2")
    assert deleted.status_code == 204, deleted.text

    # Group and children all read as 404 (live view).
    assert client.get(f"{EXPERIMENTS}/doomed-group-v2").status_code == 404
    for child_name in ("child-1", "child-2"):
        assert client.get(f"{EVALUATIONS}/{child_name}").status_code == 404

    # Names are reusable in a fresh group.
    fresh_group = client.post(EXPERIMENTS, json={"name": "doomed-group-v2"})
    assert fresh_group.status_code == 201, fresh_group.text
    revived = client.post(
        EVALUATIONS, json=_evaluation_body(name="child-1", experiment_group_id=fresh_group.json()["id"])
    )
    assert revived.status_code == 201, revived.text

    # Trash view still surfaces the cascaded rows.
    deleted_groups = client.get(EXPERIMENTS, params={"filter[is_deleted]": "true"})
    assert any(g["name"].startswith("doomed-group-v2-deleted-") for g in deleted_groups.json()["data"])
    deleted_exps = client.get(EVALUATIONS, params={"filter[is_deleted]": "true"})
    deleted_names = {e["name"] for e in deleted_exps.json()["data"]}
    assert any(n.startswith("child-1-deleted-") for n in deleted_names)
    assert any(n.startswith("child-2-deleted-") for n in deleted_names)


def test_create_evaluation_in_deleted_group_rejected(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "ephemeral-group"}).json()
    client.delete(f"{EXPERIMENTS}/ephemeral-group")

    response = client.post(
        EVALUATIONS,
        json=_evaluation_body(name="orphan", experiment_group_id=group["id"]),
    )
    assert response.status_code == 400, response.text
    assert "deleted" in response.json()["detail"].lower()


def test_update_or_delete_deleted_evaluation_returns_404(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-once", experiment_group_id=group["id"]))
    client.delete(f"{EVALUATIONS}/exp-once")

    # GET, PUT, and a second DELETE all 404 on the (now-renamed) deleted row.
    assert client.get(f"{EVALUATIONS}/exp-once").status_code == 404
    assert (
        client.put(
            f"{EVALUATIONS}/exp-once",
            json=_evaluation_body(name="exp-once", experiment_group_id=group["id"]),
        ).status_code
        == 404
    )
    delete_response = client.delete(f"{EVALUATIONS}/exp-once")
    assert delete_response.status_code == 404


def test_pin_and_unpin_evaluation(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-pin", experiment_group_id=group["id"]))

    # New evaluations start unpinned.
    fetched = client.get(f"{EVALUATIONS}/exp-pin")
    assert fetched.status_code == 200
    assert fetched.json()["pinned_at"] is None

    # Pin sets pinned_at to a timestamp.
    pinned = client.post(f"{EVALUATIONS}/exp-pin/pin")
    assert pinned.status_code == 200, pinned.text
    first_pinned_at = pinned.json()["pinned_at"]
    assert first_pinned_at is not None

    # Re-pinning refreshes pinned_at (most-recently-pinned-first ordering).
    re_pinned = client.post(f"{EVALUATIONS}/exp-pin/pin")
    assert re_pinned.status_code == 200, re_pinned.text
    assert re_pinned.json()["pinned_at"] >= first_pinned_at

    # Unpin clears pinned_at.
    unpinned = client.delete(f"{EVALUATIONS}/exp-pin/pin")
    assert unpinned.status_code == 200, unpinned.text
    assert unpinned.json()["pinned_at"] is None

    # Unpin on an already-unpinned evaluation is a no-op.
    again = client.delete(f"{EVALUATIONS}/exp-pin/pin")
    assert again.status_code == 200, again.text
    assert again.json()["pinned_at"] is None


def test_pin_unknown_evaluation_returns_404(client: TestClient) -> None:
    assert client.post(f"{EVALUATIONS}/missing/pin").status_code == 404
    assert client.delete(f"{EVALUATIONS}/missing/pin").status_code == 404


def test_pin_rejects_deleted_evaluation(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-soft", experiment_group_id=group["id"]))
    assert client.delete(f"{EVALUATIONS}/exp-soft").status_code == 204
    assert client.post(f"{EVALUATIONS}/exp-soft/pin").status_code == 404


def test_filter_by_is_pinned(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-pinned-a", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-pinned-b", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-not-pinned", experiment_group_id=group["id"]))
    client.post(f"{EVALUATIONS}/exp-pinned-a/pin")
    client.post(f"{EVALUATIONS}/exp-pinned-b/pin")

    only_pinned = client.get(EVALUATIONS, params={"filter[is_pinned]": "true"})
    assert only_pinned.status_code == 200, only_pinned.text
    pinned_names = {e["name"] for e in only_pinned.json()["data"]}
    assert pinned_names == {"exp-pinned-a", "exp-pinned-b"}

    only_unpinned = client.get(EVALUATIONS, params={"filter[is_pinned]": "false"})
    assert only_unpinned.status_code == 200, only_unpinned.text
    unpinned_names = {e["name"] for e in only_unpinned.json()["data"]}
    assert "exp-not-pinned" in unpinned_names
    assert "exp-pinned-a" not in unpinned_names
    assert "exp-pinned-b" not in unpinned_names

    no_filter = client.get(EVALUATIONS)
    all_names = {e["name"] for e in no_filter.json()["data"]}
    assert {"exp-pinned-a", "exp-pinned-b", "exp-not-pinned"} <= all_names


def test_sort_by_pinned_at_most_recent_first(client: TestClient) -> None:
    group = _create_group(client)
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-old-pin", experiment_group_id=group["id"]))
    client.post(EVALUATIONS, json=_evaluation_body(name="exp-new-pin", experiment_group_id=group["id"]))
    client.post(f"{EVALUATIONS}/exp-old-pin/pin")
    client.post(f"{EVALUATIONS}/exp-new-pin/pin")

    pinned_desc = client.get(EVALUATIONS, params={"filter[is_pinned]": "true", "sort": "-pinned_at"})
    assert pinned_desc.status_code == 200, pinned_desc.text
    names_desc = [e["name"] for e in pinned_desc.json()["data"]]
    assert names_desc.index("exp-new-pin") < names_desc.index("exp-old-pin")

    pinned_asc = client.get(EVALUATIONS, params={"filter[is_pinned]": "true", "sort": "pinned_at"})
    assert pinned_asc.status_code == 200, pinned_asc.text
    names_asc = [e["name"] for e in pinned_asc.json()["data"]]
    assert names_asc.index("exp-old-pin") < names_asc.index("exp-new-pin")


def test_evaluation_belongs_to_multiple_groups(client: TestClient) -> None:
    """An evaluation created with multiple experiment_ids appears in each group's leaderboard."""
    group_a = client.post(EXPERIMENTS, json={"name": "multi-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "multi-b"}).json()
    body = _evaluation_body(name="multi-member", experiment_group_id="placeholder")
    body.pop("experiment_group_id")
    body["experiment_ids"] = [group_a["id"], group_b["id"]]

    created = client.post(EVALUATIONS, json=body)
    assert created.status_code == 201, created.text
    assert set(created.json()["experiment_ids"]) == {group_a["id"], group_b["id"]}
    # Deprecated single-group alias still resolves (to the first membership).
    assert created.json()["experiment_group_id"] == group_a["id"]

    for gid in (group_a["id"], group_b["id"]):
        listed = client.get(EVALUATIONS, params={"filter[experiment_group_id]": gid})
        assert listed.status_code == 200, listed.text
        assert any(e["name"] == "multi-member" for e in listed.json()["data"])


def test_evaluation_create_rejects_empty_experiment_ids(client: TestClient) -> None:
    """The >=1-group invariant: an empty experiment_ids list is rejected."""
    body = _evaluation_body(name="no-groups", experiment_group_id="placeholder")
    body.pop("experiment_group_id")
    body["experiment_ids"] = []
    resp = client.post(EVALUATIONS, json=body)
    assert resp.status_code == 422, resp.text


def test_delete_group_reference_counted_cascade(client: TestClient) -> None:
    """Deleting a group deletes only its sole-membership members; shared members survive elsewhere."""
    group_a = client.post(EXPERIMENTS, json={"name": "rc-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "rc-b"}).json()

    shared = _evaluation_body(name="rc-shared", experiment_group_id="placeholder")
    shared.pop("experiment_group_id")
    shared["experiment_ids"] = [group_a["id"], group_b["id"]]
    client.post(EVALUATIONS, json=shared)
    client.post(EVALUATIONS, json=_evaluation_body(name="rc-sole", experiment_group_id=group_a["id"]))

    assert client.delete(f"{EXPERIMENTS}/rc-a").status_code == 204

    # Sole member gone; shared member survives, now only in group b.
    assert client.get(f"{EVALUATIONS}/rc-sole").status_code == 404
    survivor = client.get(f"{EVALUATIONS}/rc-shared")
    assert survivor.status_code == 200, survivor.text
    assert survivor.json()["experiment_ids"] == [group_b["id"]]

    in_b = client.get(EVALUATIONS, params={"filter[experiment_group_id]": group_b["id"]})
    assert any(e["name"] == "rc-shared" for e in in_b.json()["data"])


def test_patch_evaluation_adds_to_group(client: TestClient) -> None:
    """PATCH .../evaluations/{name} with a merged experiment_ids adds a membership; the run shows on the board."""
    group_a = client.post(EXPERIMENTS, json={"name": "add-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "add-b"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="add-eval", experiment_group_id=group_a["id"]))

    added = client.patch(f"{EVALUATIONS}/add-eval", json={"experiment_ids": [group_a["id"], group_b["id"]]})
    assert added.status_code == 200, added.text
    assert set(added.json()["experiment_ids"]) == {group_a["id"], group_b["id"]}

    in_b = client.get(EVALUATIONS, params={"filter[experiment_group_id]": group_b["id"]})
    assert any(e["name"] == "add-eval" for e in in_b.json()["data"])


def test_patch_evaluation_unknown_group_rejected(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "add-known"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="add-eval-2", experiment_group_id=group["id"]))
    resp = client.patch(f"{EVALUATIONS}/add-eval-2", json={"experiment_ids": [group["id"], "experiment_group-nope"]})
    assert resp.status_code == 400, resp.text


def test_patch_evaluation_rejects_empty_experiment_ids(client: TestClient) -> None:
    group = client.post(EXPERIMENTS, json={"name": "empty-ids"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="empty-ids-eval", experiment_group_id=group["id"]))
    resp = client.patch(f"{EVALUATIONS}/empty-ids-eval", json={"experiment_ids": []})
    assert resp.status_code == 400, resp.text
    # The >=1-group invariant holds: membership is unchanged.
    assert client.get(f"{EVALUATIONS}/empty-ids-eval").json()["experiment_ids"] == [group["id"]]


def test_patch_evaluation_leaves_omitted_fields_unchanged(client: TestClient) -> None:
    """Partial semantics: PATCHing experiment_ids alone must not clobber omitted fields (e.g. description)."""
    group_a = client.post(EXPERIMENTS, json={"name": "patch-a"}).json()
    group_b = client.post(EXPERIMENTS, json={"name": "patch-b"}).json()
    body = _evaluation_body(name="patch-eval", experiment_group_id=group_a["id"])
    body["description"] = "keep me"
    body["metadata"] = {"team": "switchyard"}
    client.post(EVALUATIONS, json=body)

    patched = client.patch(f"{EVALUATIONS}/patch-eval", json={"experiment_ids": [group_a["id"], group_b["id"]]})
    assert patched.status_code == 200, patched.text
    data = patched.json()
    assert set(data["experiment_ids"]) == {group_a["id"], group_b["id"]}
    assert data["description"] == "keep me"  # omitted from the PATCH -> unchanged
    assert data["metadata"] == {"team": "switchyard"}


def test_patch_evaluation_dedupes_experiment_ids(client: TestClient) -> None:
    """Membership is a set: duplicate ids are stored once, so a group can't double-count an evaluation."""
    group = client.post(EXPERIMENTS, json={"name": "dedupe-grp"}).json()
    client.post(EVALUATIONS, json=_evaluation_body(name="dedupe-eval", experiment_group_id=group["id"]))

    patched = client.patch(f"{EVALUATIONS}/dedupe-eval", json={"experiment_ids": [group["id"], group["id"]]})
    assert patched.status_code == 200, patched.text
    assert patched.json()["experiment_ids"] == [group["id"]]

    # The group counts the evaluation once, not twice.
    listed = client.get(EXPERIMENTS)
    grp = next(g for g in listed.json()["data"] if g["id"] == group["id"])
    assert grp["evaluation_count"] == 1


def test_experiment_group_column_layout_defaults_and_round_trips(client: TestClient) -> None:
    created = client.post(EXPERIMENTS, json={"name": "columns-cfg"})
    assert created.status_code == 201, created.text
    assert created.json()["column_layout"] is None
    assert client.get(f"{EXPERIMENTS}/columns-cfg").json()["column_layout"] is None

    layout = {
        "order": ["name", "evaluator-llm-judge.answers_question", "created_at"],
        "hidden": ["created_by", "metadata-run_id"],
    }
    updated = client.put(f"{EXPERIMENTS}/columns-cfg", json={"name": "columns-cfg", "column_layout": layout})
    assert updated.status_code == 200, updated.text
    assert updated.json()["column_layout"] == layout

    # An update that omits column_layout preserves the saved layout rather than clearing it.
    preserved = client.put(f"{EXPERIMENTS}/columns-cfg", json={"name": "columns-cfg", "summary": "unrelated edit"})
    assert preserved.status_code == 200, preserved.text
    assert preserved.json()["column_layout"] == layout

    # An empty layout is a saved layout, not an absent one, so the client keeps its defaults off.
    reset = client.put(
        f"{EXPERIMENTS}/columns-cfg",
        json={"name": "columns-cfg", "column_layout": {"order": [], "hidden": []}},
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["column_layout"] == {"order": [], "hidden": []}


def test_experiment_group_partial_update_preserves_unsent_fields(client: TestClient) -> None:
    # A layout-only save must not blank unsent fields, nor echo back stale copies to avoid it.
    created = client.post(
        EXPERIMENTS,
        json={
            "name": "partial-update",
            "description": "why this experiment exists",
            "summary": "what it found",
            "insight_id": "insight-1",
            "metadata": {"owner": "team-a"},
        },
    )
    assert created.status_code == 201, created.text

    layout = {"order": ["name", "created_at"], "hidden": ["created_by"]}
    updated = client.put(
        f"{EXPERIMENTS}/partial-update",
        json={"name": "partial-update", "column_layout": layout},
    )
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["column_layout"] == layout
    assert body["description"] == "why this experiment exists"
    assert body["summary"] == "what it found"
    assert body["insight_id"] == "insight-1"
    assert body["metadata"] == {"owner": "team-a"}

    # An explicit null still clears, so nothing loses the ability to unset a field.
    cleared = client.put(
        f"{EXPERIMENTS}/partial-update",
        json={"name": "partial-update", "summary": None},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["summary"] is None
    assert cleared.json()["description"] == "why this experiment exists"
