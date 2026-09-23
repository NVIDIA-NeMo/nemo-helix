# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolve_service_loading_order in nhx.common.service.deptree."""

import pytest
from nhx.common.service.deptree import CircularDependencyError, resolve_service_loading_order


@pytest.mark.unit
class TestResolveServiceLoadingOrder:
    """Tests for resolve_service_loading_order (pure function, no default tree)."""

    def test_empty_services_returns_empty(self):
        """Empty input returns empty list."""
        assert resolve_service_loading_order([], {}) == []

    def test_single_service_with_no_deps(self):
        """Service with no dependencies returns just that service."""
        tree = {"entities": []}
        assert resolve_service_loading_order(["entities"], tree) == ["entities"]

    def test_single_service_with_deps(self):
        """Single service returns deps first, then the service."""
        tree = {
            "jobs": ["entities", "auth", "secrets", "files"],
            "entities": [],
            "auth": ["entities"],
            "secrets": ["entities", "auth"],
            "files": ["entities", "auth", "secrets"],
        }
        result = resolve_service_loading_order(["jobs"], tree)
        expected = ["entities", "auth", "secrets", "files", "jobs"]
        assert result == expected

    def test_dependency_order_respected(self):
        """Dependencies appear in valid order (e.g. auth after entities)."""
        tree = {"auth": ["entities"], "entities": []}
        assert resolve_service_loading_order(["auth"], tree) == ["entities", "auth"]

    def test_transitive_deps_included(self):
        """Transitive dependencies are included and ordered correctly."""
        tree = {
            "files": ["entities", "auth", "secrets"],
            "entities": [],
            "auth": ["entities"],
            "secrets": ["entities", "auth"],
        }
        result = resolve_service_loading_order(["files"], tree)
        expected = ["entities", "auth", "secrets", "files"]
        assert result == expected

    def test_multiple_services_shared_deps_once(self):
        """Multiple services share dependencies; each dep appears once."""
        tree = {
            "files": ["entities", "auth", "secrets"],
            "jobs": ["entities", "auth", "secrets", "files"],
            "entities": [],
            "auth": ["entities"],
            "secrets": ["entities", "auth"],
        }
        result = resolve_service_loading_order(["files", "jobs"], tree)
        expected = ["entities", "auth", "secrets", "files", "jobs"]
        assert result == expected

    def test_service_not_in_tree_appears_at_end(self):
        """Services not in dependency_tree are appended at end in input order."""
        tree = {"auth": ["entities"], "entities": []}
        result = resolve_service_loading_order(["studio", "hello-world"], tree)
        assert result == ["studio", "hello-world"]

    def test_mixed_known_and_unknown(self):
        """Known services are ordered by deps; unknown are at end."""
        tree = {"entities": [], "auth": ["entities"]}
        result = resolve_service_loading_order(["entities", "studio", "auth"], tree)
        expected = ["entities", "auth", "studio"]
        assert result == expected

    def test_custom_dependency_tree(self):
        """Custom dependency_tree is used."""
        tree = {"a": ["b"], "b": []}
        assert resolve_service_loading_order(["a"], tree) == ["b", "a"]
        assert resolve_service_loading_order(["b"], tree) == ["b"]

    def test_entities_first_root_services_ordered_first(self):
        """Services with no dependencies (e.g. entities) appear first in order."""
        tree = {
            "entities": [],
            "auth": ["entities"],
            "files": ["entities", "auth", "secrets"],
            "secrets": ["entities", "auth"],
        }
        result = resolve_service_loading_order(["auth", "files"], tree)
        assert result[0] == "entities", "entities has no deps and must start first"
        # Order among the rest: auth after entities, secrets after auth, files after secrets
        assert result.index("entities") < result.index("auth")
        assert result.index("auth") < result.index("secrets")
        assert result.index("secrets") < result.index("files")


@pytest.mark.unit
class TestCircularDependency:
    """Tests for circular dependency detection and CircularDependencyError."""

    def test_two_node_cycle_raises(self):
        """A -> B -> A raises CircularDependencyError."""
        tree = {"a": ["b"], "b": ["a"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolve_service_loading_order(["a"], tree)
        err = exc_info.value
        assert "Circular dependency" in str(err)
        assert err.nodes == {"a", "b"}
        assert err.cycle is not None
        assert len(err.cycle) == 3  # e.g. ["a", "b", "a"]
        assert err.cycle[0] == err.cycle[-1]

    def test_three_node_cycle_raises(self):
        """A -> B -> C -> A raises with cycle reported."""
        tree = {"a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolve_service_loading_order(["a"], tree)
        err = exc_info.value
        assert err.nodes == {"a", "b", "c"}
        assert err.cycle is not None
        assert err.cycle[0] == err.cycle[-1]
        assert set(err.cycle) == {"a", "b", "c"}

    def test_self_loop_raises(self):
        """A -> A raises CircularDependencyError."""
        tree = {"a": ["a"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolve_service_loading_order(["a"], tree)
        err = exc_info.value
        assert err.nodes == {"a"}
        assert err.cycle == ["a", "a"]

    def test_cycle_in_transitive_deps_raises(self):
        """DAG with a cycle among requested + deps raises."""
        tree = {"x": ["a"], "a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolve_service_loading_order(["x"], tree)
        err = exc_info.value
        assert err.nodes is not None
        assert "a" in err.nodes and "b" in err.nodes and "c" in err.nodes
        assert "Circular dependency" in str(err)

    def test_error_message_includes_cycle_string(self):
        """Exception message includes 'Cycle: a -> b -> a' for inspection."""
        tree = {"a": ["b"], "b": ["a"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolve_service_loading_order(["a"], tree)
        msg = str(exc_info.value)
        assert "Cycle:" in msg
        assert "a" in msg and "b" in msg
