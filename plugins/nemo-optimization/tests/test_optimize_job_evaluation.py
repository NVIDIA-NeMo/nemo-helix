# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import yaml
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.refs import FilesetRef
from nemo_optimization.jobs.optimize import OptimizeJob
from nemo_optimization.schemas.optimize import OptimizeSpec, OptimizeSubmitSpec
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.values.models import ModelRef
from nhx_evals_sdk.values.scores import RangeScore

CONFIG_URL = "/apis/files/v2/workspaces/ws/filesets/eval-data/-/eval-config.yaml"
EVALUATION_CONFIG = "ws/eval-data#eval-config.yaml"


def eval_config(maximum: int = 1) -> str:
    judge = LLMJudgeMetric(
        model=ModelRef("models-ws/judge"),
        scores=[RangeScore(name="quality", minimum=0, maximum=maximum)],
        prompt_template="Grade {{ sample.output_text }} for {{ item.q }}",
    )
    return yaml.safe_dump(
        {
            "dataset": [{"id": "r1", "q": "hello"}],
            "prompt_template": "{{ item.q }}",
            "metrics": [bundle_metric(judge, InlineMetricBundlePackager()).model_dump(mode="json")],
        }
    )


def serve_config(text: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == CONFIG_URL, request.url
        return httpx.Response(200, content=text.encode())

    return handler


async def submit(text: str) -> OptimizeSpec:
    client = AsyncNemoClient(
        base_url="http://test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve_config(text)), base_url="http://test"),
    )
    return await OptimizeJob.to_spec(
        OptimizeSubmitSpec(
            optimize_config="optimize.yaml",
            optimize_config_fileset=FilesetRef("ws/bundle"),
            evaluation_config=EVALUATION_CONFIG,
        ),
        workspace="ws",
        entity_client=None,
        async_sdk=client,
        is_local=False,
    )


async def test_submit_accepts_an_evaluation_the_study_can_reproduce() -> None:
    assert (await submit(eval_config())).evaluation_config == EVALUATION_CONFIG


async def test_submit_rejects_an_evaluation_the_study_cannot_reproduce() -> None:
    with pytest.raises(HelixJobCompilationError, match="ranging from 0 to 1"):
        await submit(eval_config(maximum=100))


def test_spec_rejects_an_evaluation_config_that_is_not_a_fileset_object_ref() -> None:
    with pytest.raises(ValueError, match="evaluation_config"):
        OptimizeSubmitSpec(
            optimize_config="optimize.yaml",
            optimize_config_fileset=FilesetRef("ws/bundle"),
            evaluation_config="eval-config.yaml",
        )


def test_run_scores_the_study_from_the_evaluation(
    tmp_path: Path,
    ctx: JobContext,
    make_platform_client: Callable[..., NemoClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    config_path = tmp_path / "optimize.yaml"
    config_path.write_text(
        yaml.safe_dump({"models": {"default": {"model": "agent"}}, "optimizer": {"numeric": {"enabled": True}}})
    )
    seen: dict[str, Any] = {}

    def dispatch(**kwargs: Any) -> dict[str, Any]:
        config = kwargs["optimize_config"]
        seen["config"] = config
        seen["rows"] = json.loads(Path(config["eval"]["general"]["dataset"]["file_path"]).read_text())
        return {"status": "completed"}

    with (
        patch("nemo_optimization.jobs.optimize.preflight_validate_llm_models"),
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=dispatch),
    ):
        result = OptimizeJob().run(
            {"optimize_config": str(config_path), "workspace": "ws", "evaluation_config": EVALUATION_CONFIG},
            ctx=ctx,
            sdk=make_platform_client(serve_config(eval_config())),
        )

    assert result["status"] == "completed"
    assert seen["config"]["models"]["judge"]["model"] == "models-ws/judge"
    assert seen["config"]["eval"]["evaluators"]["accuracy"]["_type"] == "tunable_rag_evaluator"
    assert [(row["id"], row["question"]) for row in seen["rows"]] == [("r1", "hello")]
