# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-time dependencies shared by the analyst agent and its read methods.

The CLI builds one :class:`AnalystDeps` for each run. The Nooa agent keeps it
hidden from generated code and injects it into every scoped backend method.
Keeping it in its own module avoids an import cycle between the agent and the
read-method implementations.
"""

from dataclasses import dataclass
from datetime import datetime

from nemo_insights_plugin.analyst.analyst_backend import AnalystBackend


@dataclass
class AnalystDeps:
    """Per-run configuration injected into every analyst read method.

    Every tool talks to the platform through a single :class:`AnalystBackend`
    built once by the CLI and shared here, rather than constructing an SDK
    client per call. The CLI owns the backend's client lifecycle (closing it
    when the run finishes), so tools must not close it.

    Attributes:
        agent: Agent under test. Used as the default ``agent_name`` filter for
            span/insight tools.
        workspace: NHX workspace the analyst operates in.
        base_url: Base URL of the running NHX instance (run metadata; tools go
            through ``backend``).
        insights_output: When set, the backend also persists insights to this
            local YAML file, mirroring the rows the platform stored (run
            metadata).
        backend: Shared data-access backend used by every tool.
        since: Optional lower bound for scheduled incremental analysis. Backend
            reads enforce this even if the model omits a time filter.
        max_results: Hard ceiling on items any single fetch may pull across
            pages. A fetch's ``limit`` is clamped to this so a wide filter
            can't flood the model's context; the analyst paginates or narrows
            its filter to see more.
    """

    agent: str = ""
    workspace: str = "default"
    base_url: str | None = None
    insights_output: str | None = None
    backend: AnalystBackend | None = None  # set by the CLI per run
    since: datetime | None = None
    evaluation_id: str | None = None  # run scope; AND-pinned onto every span read
    max_results: int = 200
