# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""say-hello job — registered under ``nemo.jobs``.

Writes a greeting to persistent storage and registers it as a result
via :attr:`~nemo_platform_plugin.job_context.JobContext.results`. Programmatic
tests can provide :class:`~nemo_platform_plugin.job_results.LocalJobResults` so
the artefact lands under ``<persistent>/results/greeting`` with no clients.
"""

from __future__ import annotations

from nemo_example_plugin.core import say_hello
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext

DEFAULT_RESULT_NAME = "greeting"
DEFAULT_FILE_NAME = "greeting.txt"


class SayHelloJob(NemoJob):
    """Greet a name, save the greeting as a registered job result."""

    name = "say-hello"
    description = "Greet a name and save the greeting as a registered job result."
    container = "cpu-tasks"

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
    ) -> dict:
        name = config.get("name", "world")
        greeting = say_hello(name)

        greeting_path = ctx.storage.persistent / DEFAULT_FILE_NAME
        greeting_path.write_text(greeting, encoding="utf-8")
        result = ctx.results.save(DEFAULT_RESULT_NAME, greeting_path)

        self.report_progress(
            ctx,
            work_done=1,
            work_total=1,
            status="completed",
            details={"name": name},
        )

        return {
            "result": greeting,
            "artifact": result.model_dump(),
        }
