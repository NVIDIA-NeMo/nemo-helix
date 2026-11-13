# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A NeMo Gym resources server that scores financial document QA as the Nemotron finance cookbook does.

Each score the cookbook records for a row is a reward component, using Gym's multi-reward response: the mapping
in ``reward_components``, and each score again as a top-level field, which is what Gym's own aggregates read.
``reward`` is ``task_correct``, the cookbook's headline score.
"""

import sys
from pathlib import Path
from typing import Any, Optional

from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import]
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    SimpleResourcesServer,
)

# Gym starts the server from this directory; the explicit path also covers an import from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring  # noqa: E402


class FinanceQAConfig(BaseResourcesServerConfig):
    tolerance: float = 0.05


class FinanceQAVerifyRequest(BaseVerifyRequest):
    expected_answer: str
    task: str


class FinanceQAVerifyResponse(BaseMultiRewardVerifyResponse):
    # Repeated as top-level fields: Gym's aggregate metrics do not descend into reward_components.
    exact: float = 0.0
    relaxed: float = 0.0
    task_correct: float = 0.0
    answer_f1: float = 0.0
    trace_closed: float = 0.0
    hit_max_new_tokens: float = 0.0
    program_syntax_valid: Optional[float] = None  # code tasks only
    task: str


class FinanceQAResourcesServer(SimpleResourcesServer):
    config: FinanceQAConfig

    async def verify(self, body: FinanceQAVerifyRequest) -> FinanceQAVerifyResponse:
        components = scoring.score(
            task=body.task,
            expected_answer=body.expected_answer,
            response=body.response.model_dump(mode="json"),
            tolerance=self.config.tolerance,
        )
        return FinanceQAVerifyResponse(
            **body.model_dump(),
            reward=components["task_correct"],
            reward_components=components,
            **components,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """The cookbook's ``by_task`` numbers, when Gym runs the environment itself."""
        return scoring.by_task(rollout for rollouts in tasks for rollout in rollouts)


if __name__ == "__main__":
    FinanceQAResourcesServer.run_webserver()
