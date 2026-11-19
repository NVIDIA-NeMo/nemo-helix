# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
from nemo_helix_ext.cli.core.job_watch_graph import DagLayout
from nemo_helix_plugin.jobs.workflows import WorkflowNodeView
from prompt_toolkit.utils import get_cwidth


def views(dependencies: dict[str, tuple[str, ...]]) -> dict[str, WorkflowNodeView]:
    return {
        name: WorkflowNodeView(
            name=name,
            depends_on=list(parents),
            freshness="dirty",
            scheduling_state="running",
            reason=None,
            blocked_by=[],
            requested_run_revision="rev-1",
            active_execution_id=None,
            latest_execution_id=None,
            successful_execution_id=None,
            last_outcome=None,
            has_pending_run=False,
            last_inputs_checked_at=datetime.now(timezone.utc),
            last_poll_attempt_at=None,
            last_poll_error=None,
        )
        for name, parents in dependencies.items()
    }


def test_diamond_has_all_directed_edges_and_disconnected_nodes() -> None:
    dependencies = {"root": (), "left": ("root",), "right": ("root",), "merge": ("left", "right"), "alone": ()}
    layout = DagLayout(dependencies)
    assert set(layout.boxes) == set(dependencies)
    assert layout.boxes["root"].x < layout.boxes["left"].x < layout.boxes["merge"].x
    assert layout.boxes["left"].x == layout.boxes["right"].x
    assert {(segment.source, segment.target) for segment in layout.segments} == {
        (parent, name) for name, parents in dependencies.items() for parent in parents
    }
    rows = layout.viewport(views(dependencies), "merge", 0, 0, layout.width, layout.height)
    text = "\n".join("".join(fragment[1] for fragment in row) for row in rows)
    for name in dependencies:
        assert name in text
    assert "▶" in text
    assert any("reverse" in fragment[0] for row in rows for fragment in row)


def test_skipped_layer_routes_do_not_cross_node_boxes() -> None:
    dependencies = {"a": (), "b": ("a",), "c": ("b",), "d": ("a", "c")}
    layout = DagLayout(dependencies)
    for segment in layout.segments:
        for box in layout.boxes.values():
            if segment.y1 == segment.y2 and box.y <= segment.y1 <= box.y + 3:
                assert max(segment.x1, segment.x2) < box.x or min(segment.x1, segment.x2) >= box.x + 24
    assert sum(segment.source == "a" and segment.target == "d" for segment in layout.segments) == 9


@pytest.mark.parametrize("x,y", [(0, 0), (20, 1), (30, 3), (50, 7)])
def test_viewport_is_bounded_and_supports_wide_names(x: int, y: int) -> None:
    dependencies = {"根节点": (), "left": ("根节点",), "right": ("根节点",), "merge": ("left", "right")}
    layout = DagLayout(dependencies)
    rows = layout.viewport(views(dependencies), "根节点", x, y, 31, 9)
    assert len(rows) == 9
    assert all(get_cwidth("".join(fragment[1] for fragment in row)) == 31 for row in rows)


def test_large_graph_renders_small_viewport() -> None:
    dependencies: dict[str, tuple[str, ...]] = {f"input-{index}": () for index in range(500)}
    dependencies["merge"] = tuple(dependencies)
    layout = DagLayout(dependencies)
    assert len(layout.boxes) == 501
    assert layout.height >= 2500
    rows = layout.viewport(views(dependencies), "merge", 100, 100, 80, 20)
    assert len(rows) == 20
    assert all(len(row) == 80 for row in rows)


def test_invalid_graph_reports_error() -> None:
    with pytest.raises(ValueError, match="cyclic"):
        DagLayout({"a": ("b",), "b": ("a",)})
    with pytest.raises(ValueError, match="Unknown"):
        DagLayout({"a": ("missing",)})


def test_geometry_is_independent_of_definition_order() -> None:
    dependencies = {"root": (), "left": ("root",), "right": ("root",), "merge": ("left", "right")}
    assert DagLayout(dependencies).boxes == DagLayout(dict(reversed(list(dependencies.items())))).boxes


def test_combining_characters_in_node_names_are_preserved() -> None:
    dependencies: dict[str, tuple[str, ...]] = {"cafe\u0301": ()}
    layout = DagLayout(dependencies)
    rows = layout.viewport(views(dependencies), "cafe\u0301", 0, 0, 30, 8)
    text = "\n".join("".join(fragment[1] for fragment in row) for row in rows)
    assert "cafe\u0301" in text
    assert all(get_cwidth("".join(fragment[1] for fragment in row)) == 30 for row in rows)


def test_unrelated_edge_crossings_have_a_distinct_marker() -> None:
    dependencies = {"a": (), "b": (), "x": ("a",), "y": ("b",), "d": ("x", "b"), "e": ("y", "a")}
    layout = DagLayout(dependencies)
    rows = layout.viewport(views(dependencies), "d", 0, 0, layout.width, layout.height)
    text = "\n".join("".join(fragment[1] for fragment in row) for row in rows)
    assert "╳" in text
