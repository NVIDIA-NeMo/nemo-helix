# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface: the first-eval job collection under ``/apis/eval-author``."""

from __future__ import annotations

from typing import ClassVar

from nemo_eval_author_plugin.authz import scope
from nemo_eval_author_plugin.jobs.first_eval import FirstEvalJob
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

#: The ``source`` tag on first-eval job records.
FIRST_EVAL_JOB_SOURCE = "nemo-eval-author-plugin-first-eval"


class EvalAuthorService(NemoService):
    name: ClassVar[str] = "eval-author"
    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "files"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                # POST /apis/eval-author/v2/workspaces/{workspace}/jobs/first-eval
                router=add_job_routes(
                    FirstEvalJob, service_name=FIRST_EVAL_JOB_SOURCE, authz=scope.child("first-eval")
                ),
                tag="Eval Author",
                description="Author a platform agent's first evaluation suite as a job.",
                prefix="/v2/workspaces/{workspace}",
            ),
        ]
