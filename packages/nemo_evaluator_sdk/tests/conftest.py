# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
_CATEGORY_MARKERS = {
    "unit": "Unit tests for the SDK package.",
    "e2e": "End-to-end tests.",
    "integration": "Integration tests.",
    "regression": "Regression tests.",
    "canary": "Canary tests.",
    "slow": "Slow-running tests.",
    "skip_in_ci": "Tests skipped in CI environments.",
}
_OTHER_MARKERS = {
    "real_codex_home": "Live test that needs the developer's real Codex login (skips the Codex home guard).",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker_name, description in {**_CATEGORY_MARKERS, **_OTHER_MARKERS}.items():
        config.addinivalue_line("markers", f"{marker_name}: {description}")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        # Pytest passes the full repo item list here once this conftest is loaded,
        # so restrict the auto-unit marker to the SDK's own tests.
        if item_path != _TESTS_DIR and _TESTS_DIR not in item_path.parents:
            continue

        marker_names = {marker.name for marker in item.iter_markers()}
        if not marker_names.intersection(_CATEGORY_MARKERS):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _guard_codex_home(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the Fabric runtime's per-trial Codex homes and base login inside the test's tmp dir.

    A Codex-adapter trial creates a temp Codex home and links the base ``auth.json`` into it; without
    this, unit tests would do that against the developer's real ``~/.codex``.
    """
    if request.node.get_closest_marker("real_codex_home"):
        return
    # A sibling of the test's tmp_path, so tests that inspect tmp_path don't see these directories.
    guard = tmp_path_factory.mktemp("codex-home-guard")
    monkeypatch.setenv("CODEX_HOME", str(guard / "base-codex-home"))
    temp_root = guard / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
