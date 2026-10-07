# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-boundary tests protecting the CLI startup path.

``nemo_helix_plugin.cli`` and ``nemo_helix_plugin.discovery`` are both on the hot
path of every ``nemo`` CLI invocation (the entrypoint imports them before any
command runs). They must NOT eagerly import the heavy plugin-primitive surface
(FastAPI, Pydantic, the typed client, NemoJob/NemoService/...), which is why those
imports live under ``TYPE_CHECKING`` with string-form ``cast`` and in-body imports
for the few genuine runtime uses.

These tests run in a fresh subprocess (so sibling suites that already loaded the
heavy stack can't mask a regression) and assert on MODULE BOUNDARIES — which
modules end up in ``sys.modules`` — rather than fragile timing thresholds. If a
future edit re-adds an eager top-level import of a deferred module, these fail.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Modules that importing the CLI hot-path modules must NOT drag in. ``fastapi`` and
# ``pydantic`` are the expensive leaves; ``nemo_helix_plugin.client.types`` is the
# ~100ms typed-client module the eager chain used to pull via ``job -> client``.
_FORBIDDEN = (
    "fastapi",
    "nemo_helix_plugin.client.types",
    "nemo_helix_plugin.job",
    "nemo_helix_plugin.service",
    "nemo_helix_plugin.customization_contributor",
)


def _assert_light_import(module: str) -> None:
    code = textwrap.dedent(
        f"""
        import sys

        import {module}  # noqa: F401

        forbidden = {_FORBIDDEN!r}
        leaked = sorted(m for m in forbidden if m in sys.modules)
        assert not leaked, f"{module} eagerly imported heavy modules: " + repr(leaked)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_importing_plugin_cli_stays_light() -> None:
    _assert_light_import("nemo_helix_plugin.cli")


def test_importing_discovery_stays_light() -> None:
    _assert_light_import("nemo_helix_plugin.discovery")
