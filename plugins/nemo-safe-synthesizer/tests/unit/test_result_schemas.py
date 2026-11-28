# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from nemo_helix_plugin.jobs.api_factory import PydanticResultSerializer
from nemo_safe_synthesizer.config.evaluation_report_data import Metrics
from nemo_safe_synthesizer.config.external_results import SafeSynthesizerSummary, SafeSynthesizerTiming
from nemo_safe_synthesizer_plugin.jobs.generate import RESULT_ROUTES

SPEC_PATH = Path(__file__).parents[2] / "openapi" / "openapi.yaml"

UNAVAILABLE_SCORE = {"raw_score": None, "score": None, "grade": "Unavailable", "notes": None}
COLUMN_PROFILE = {
    "type": "categorical",
    "count": 1,
    "unique_count": 1,
    "unique_percent": None,
    "missing_count": 0,
    "missing_percent": None,
    "avg_str_length": 3.0,
}


def _report_data_with_nulls() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "library_version": "test",
        "job_id": None,
        "scores": {"synthetic_quality": UNAVAILABLE_SCORE, "data_privacy": None},
        "metrics": {
            **dict.fromkeys(Metrics.model_fields),
            "pii_replay": {
                "score": UNAVAILABLE_SCORE,
                "enabled": None,
                "training_total_records": 0,
                "synthetic_total_records": 0,
                "entries": [],
                "skipped_reason": None,
            },
        },
        "dataset": None,
        "columns": [
            {
                "name": "color",
                "training": COLUMN_PROFILE,
                "synthetic": COLUMN_PROFILE,
                "distribution_distance": None,
                "distribution_stability": None,
                "distribution": {
                    "labels": ["red"],
                    "bin_edges": None,
                    "training_percent": [None],
                    "synthetic_percent": [100.0],
                },
                "pii": None,
            }
        ],
        "any_column_transformed": False,
        "differential_privacy": {"enabled": False, "epsilon": None, "delta": None},
        "pii_replacement": None,
    }


def _nullable_as_json_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_nullable_as_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    converted = {key: _nullable_as_json_schema(value) for key, value in node.items() if key != "nullable"}
    if node.get("nullable") is True:
        return {"anyOf": [converted, {"type": "null"}]}
    return converted


def _summary_with_nulls() -> dict[str, Any]:
    return {
        **dict.fromkeys(SafeSynthesizerSummary.model_fields),
        "timing": dict.fromkeys(SafeSynthesizerTiming.model_fields),
    }


def _serialize_result(result_name: str, data: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    serializer = next(route.serializer for route in RESULT_ROUTES if route.name == result_name)
    assert isinstance(serializer, PydanticResultSerializer)
    result_path = tmp_path / f"{result_name}.json"
    result_path.write_text(json.dumps(data), encoding="utf-8")
    return json.loads(bytes(serializer.serialize(result_path).body))


@pytest.mark.parametrize(
    ("schema_name", "data"),
    [
        ("EvaluationReportData", _report_data_with_nulls()),
        ("SafeSynthesizerSummary", _summary_with_nulls()),
    ],
)
def test_committed_spec_accepts_result_nulls(schema_name: str, data: dict[str, Any]):
    components = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))["components"]
    schema = {
        "$ref": f"#/components/schemas/{schema_name}",
        "components": _nullable_as_json_schema(components),
    }

    errors = [error.message for error in Draft202012Validator(schema).iter_errors(data)]

    assert errors == []


def test_report_data_route_returns_nulls(tmp_path):
    body = _serialize_result("evaluation-report-data", _report_data_with_nulls(), tmp_path)

    assert body["scores"]["synthetic_quality"] == UNAVAILABLE_SCORE
    assert body["metrics"]["pii_replay"]["score"] == UNAVAILABLE_SCORE
    assert body["columns"][0]["distribution"]["bin_edges"] is None
    assert body["dataset"] is None
    assert body["pii_replacement"] is None


def test_summary_route_returns_nulls(tmp_path):
    body = _serialize_result("summary", _summary_with_nulls(), tmp_path)

    assert body == _summary_with_nulls()
