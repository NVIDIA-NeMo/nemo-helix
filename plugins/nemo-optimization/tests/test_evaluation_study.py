# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from nemo_evals.filesets import FilesetRef as DatasetRef
from nemo_evals.jobs.evaluate import EvaluateInputSpec
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_optimization import evaluation_study
from nemo_optimization.evaluation_study import (
    CANDIDATE_MARKER,
    MIMICKED_JUDGE_PROMPT,
    EvaluationStudyError,
    TaskEvalConfig,
    apply_study_evaluation,
    build_study_evaluation,
    judge_model_config,
    load_fileset_dataset,
    parse_eval_config,
)
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from nhx_evals_sdk.values.models import Model, ModelRef
from nhx_evals_sdk.values.params import InferenceParams
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore
from pydantic import ValidationError

JUDGE_PROMPT = "Grade {{ sample.output_text }} for {{ item.question }} against {{ item.reference }}."


def judge_bundle(
    *,
    prompt_template: str | dict[str, Any] | None = JUDGE_PROMPT,
    score: RangeScore | None = None,
) -> dict[str, Any]:
    metric = LLMJudgeMetric(
        model=ModelRef("models-ws/judge"),
        scores=[score or RangeScore(name="quality", minimum=0, maximum=1)],
        prompt_template=prompt_template,
        inference=InferenceParams(temperature=0.0),
    )
    return bundle_metric(metric, InlineMetricBundlePackager()).model_dump(mode="json")


def string_check_bundle() -> dict[str, Any]:
    metric = StringCheckMetric(
        left_template="{{ sample.output_text }}", operation="equals", right_template="{{ item.reference }}"
    )
    return bundle_metric(metric, InlineMetricBundlePackager()).model_dump(mode="json")


def dataset_config(**overrides: Any) -> dict[str, Any]:
    return {
        "dataset": [{"id": "r1", "q": "is this phishing?", "gold": "yes"}],
        "prompt_template": "{{ item.question | upper }}",
        "field_mapping": {"custom": {"question": "q"}, "reference": "gold"},
        "metrics": [judge_bundle(), string_check_bundle()],
        **overrides,
    }


def no_dataset(_: DatasetRef) -> list[dict[str, Any]]:
    raise AssertionError("unexpected dataset read")


def study_of(config: dict[str, Any], load_dataset=no_dataset):
    return build_study_evaluation(parse_eval_config(json.dumps(config)), load_dataset)


def test_parse_eval_config_reads_json_and_yaml_with_the_evaluator_schemas() -> None:
    assert isinstance(parse_eval_config(json.dumps(dataset_config())), EvaluateInputSpec)
    task_config = {"tasks": [{"id": "t1", "intent": "say hi", "metrics": [judge_bundle()]}]}
    assert isinstance(parse_eval_config(yaml.safe_dump(task_config)), TaskEvalConfig)


def test_parse_eval_config_rejects_fields_the_evaluator_rejects() -> None:
    with pytest.raises(ValidationError):
        parse_eval_config(json.dumps({**dataset_config(), "unknown": 1}))


def test_dataset_rows_render_prompt_and_rubric_with_jinja_and_field_mapping() -> None:
    study = study_of(dataset_config())

    assert study.judge.score.name == "quality"
    [row] = study.rows
    assert row["id"] == "r1"
    assert row["question"] == "IS THIS PHISHING?"
    assert row["answer"].startswith(f"Grade {CANDIDATE_MARKER} for is this phishing? against yes.")
    assert row["answer"].endswith(
        'return the original "quality" value as "score", with a brief "reasoning" string. '
        "Do not return the original response keys."
    )


def test_rows_without_an_id_are_numbered() -> None:
    config = dataset_config(dataset=[{"q": "a", "gold": "b"}, {"q": "c", "gold": "d"}])
    assert [row["id"] for row in study_of(config).rows] == ["0", "1"]


def test_message_templates_use_the_last_user_message_and_flatten_the_judge() -> None:
    config = dataset_config(
        prompt_template={
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "{{ item.question }}"},
            ]
        },
        metrics=[
            judge_bundle(
                prompt_template={
                    "messages": [
                        {"role": "system", "content": "You grade."},
                        {"role": "user", "content": "{{ sample.output_text }} vs {{ item.reference }}"},
                    ]
                }
            )
        ],
    )
    [row] = study_of(config).rows
    assert row["question"] == "is this phishing?"
    assert row["answer"].startswith(f"system:\nYou grade.\n\nuser:\n{CANDIDATE_MARKER} vs yes")


def test_judge_template_sees_the_score_definitions() -> None:
    config = dataset_config(
        metrics=[judge_bundle(prompt_template="{{ scores.quality.maximum }} {{ sample.output_text }}")]
    )
    assert study_of(config).rows[0]["answer"].startswith(f"1 {CANDIDATE_MARKER}")


def test_fileset_datasets_are_loaded_through_the_ref() -> None:
    seen: list[str] = []

    def load(dataset: DatasetRef) -> list[dict[str, Any]]:
        seen.append(dataset.root)
        return [{"q": "from fileset", "gold": "g"}]

    study = study_of(dataset_config(dataset="ws/data#dataset/*.parquet"), load)
    assert seen == ["ws/data#dataset/*.parquet"]
    assert study.rows[0]["question"] == "FROM FILESET"


def test_load_fileset_dataset_reads_parquet_like_the_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "download"
    source.mkdir()
    pq.write_table(pa.table({"q": ["a", "b"], "n": pa.array([1, 2], type=pa.int64())}), source / "part-0.parquet")
    monkeypatch.setattr(evaluation_study, "download_dataset_sync", lambda *_args: source)

    rows = load_fileset_dataset(DatasetRef("ws/data#dataset/*.parquet"), sdk=object(), destination=tmp_path)  # ty: ignore[invalid-argument-type]

    assert rows == [{"q": "a", "n": 1}, {"q": "b", "n": 2}]


def test_task_rows_score_against_inputs_reference_and_task() -> None:
    config = {
        "tasks": [
            {
                "id": "t1",
                "intent": "Triage the inbox",
                "inputs": {"instruction": "Triage it"},
                "reference": {"label": "phishing"},
                "metrics": [
                    judge_bundle(prompt_template="{{ task.intent }}/{{ reference.label }}/{{ sample.output_text }}")
                ],
            },
            {
                "id": "t2",
                "intent": "Fallback intent",
                "metrics": [
                    judge_bundle(prompt_template="{{ task.intent }}/{{ reference.label }}/{{ sample.output_text }}")
                ],
                "reference": {"label": "safe"},
            },
        ]
    }
    first, second = study_of(config).rows
    assert (first["id"], first["question"]) == ("t1", "Triage it")
    assert first["answer"].startswith(f"Triage the inbox/phishing/{CANDIDATE_MARKER}")
    assert second["question"] == "Fallback intent"


def test_tasks_with_different_judges_are_rejected() -> None:
    config = {
        "tasks": [
            {"id": "t1", "intent": "a", "metrics": [judge_bundle()]},
            {"id": "t2", "intent": "b", "metrics": [judge_bundle(prompt_template="other {{ sample.output_text }}")]},
        ]
    }
    with pytest.raises(EvaluationStudyError, match="different judge settings per task"):
        study_of(config)


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ([string_check_bundle()], "exactly one inline LLM judge"),
        ([judge_bundle(), judge_bundle()], "exactly one inline LLM judge"),
        ([judge_bundle(score=RangeScore(name="q", minimum=1, maximum=5))], "ranging from 0 to 1"),
        (
            [
                judge_bundle(
                    score=RangeScore(name="q", minimum=0, maximum=1, parser=JSONScoreParser(json_path="result.q"))
                )
            ],
            "default parser",
        ),
        ([judge_bundle(prompt_template=None)], "explicit prompt template"),
    ],
)
def test_judges_the_study_cannot_reproduce_are_rejected(metrics: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(EvaluationStudyError, match=message):
        study_of(dataset_config(metrics=metrics))


def test_non_inline_judge_bundles_are_never_unbundled() -> None:
    bundle = judge_bundle()
    bundle["payload"] = {"kind": "cloudpickle", "data": "untrusted"}
    with pytest.raises((EvaluationStudyError, ValidationError)):
        study_of(dataset_config(metrics=[bundle]))


def test_registered_judges_route_through_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    assert judge_model_config("ws", "models-ws/judge") == {
        "provider": "nvidia",
        "model": "models-ws/judge",
        "base_url": "http://platform:8080/apis/inference-gateway/v2/workspaces/models-ws/openai/-/v1",
    }


def test_inline_judges_are_called_directly_unless_they_need_custom_auth() -> None:
    model = Model(url="http://judge:8000/v1", name="judge", served_model_name="served")
    assert judge_model_config("ws", model) == {
        "provider": "openai",
        "model": "served",
        "base_url": "http://judge:8000/v1",
    }
    with pytest.raises(EvaluationStudyError, match="custom authentication"):
        judge_model_config("ws", Model(url="http://judge", name="judge", default_headers={"x": "y"}))


def test_apply_study_evaluation_generates_scoring_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    optimize_config = {
        "models": {"default": {"model": "agent"}, "judge": {"model": "stale"}},
        "optimizer": {"numeric": {"n_trials": 4}},
        "eval": {"general": {"dataset": "old.json", "max_concurrency": 2}, "evaluators": {"old": {}}},
    }
    dataset_path = tmp_path / "dataset.json"

    updated = apply_study_evaluation(
        optimize_config, study_of(dataset_config()), dataset_path=dataset_path, workspace="ws"
    )

    assert updated["models"]["default"] == {"model": "agent"}
    assert updated["models"]["judge"]["model"] == "models-ws/judge"
    assert updated["optimizer"] == {"numeric": {"n_trials": 4}}
    assert updated["eval"]["general"] == {"dataset": {"file_path": str(dataset_path)}, "max_concurrency": 2}
    assert updated["eval"]["evaluators"] == {
        "accuracy": {
            "_type": "tunable_rag_evaluator",
            "llm_name": "judge",
            "default_scoring": False,
            "inference": {"temperature": 0.0},
            "judge_llm_prompt": MIMICKED_JUDGE_PROMPT,
        }
    }
    assert optimize_config["eval"]["evaluators"] == {"old": {}}


def test_an_explicit_default_parser_is_accepted() -> None:
    score = RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))
    assert study_of(dataset_config(metrics=[judge_bundle(score=score)])).judge.score.name == "quality"
