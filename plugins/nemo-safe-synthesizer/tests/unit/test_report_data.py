# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nemo_safe_synthesizer.package_info import __version__ as nss_version
from nemo_safe_synthesizer_plugin.report_data import (
    REPORT_DATA_SCHEMA_VERSION,
    build_report_data,
    evaluation_report_of,
    to_json_safe,
)

RENDERED_HTML = "<div>rendered plotly</div>"


class FakeFigure:
    def __init__(self, label: str):
        self.label = label

    def to_json(self, validate: bool = True) -> str:
        return json.dumps(
            {
                "data": [{"type": "bar", "name": self.label, "y": [1, None]}],
                "layout": {"title": {"text": self.label}, "template": {"layout": {"font": {"size": 12}}}},
            }
        )


def _score(grade: str = "Good", score: float | None = 7.0) -> dict:
    return {"raw_score": 0.7, "grade": grade, "score": score, "notes": None, "figure": RENDERED_HTML}


@pytest.fixture
def fake_figure_modules(monkeypatch):
    fakes = {
        "multi_modal_figures": {
            "generate_aia_figure": lambda df: FakeFigure("aia"),
            "generate_mia_figure": lambda df: FakeFigure("mia"),
            "generate_combined_correlation_figure": lambda **kwargs: FakeFigure("correlation"),
            "structure_stability_figure": lambda **kwargs: FakeFigure("pca"),
            "generate_text_semantic_similarity_figures": lambda training, synthetic: FakeFigure("semantic"),
            "generate_text_structure_similarity_figures": lambda training, synthetic: (
                FakeFigure("structure") if not training.empty else None
            ),
        },
        "column_distribution": {
            "ColumnDistributionPlotRow": SimpleNamespace(
                _get_figure_for_field=lambda field, training, synthetic: FakeFigure(f"distribution:{field.name}")
            ),
        },
        "autocorrelation_similarity_figures": {
            "generate_autocorrelation_summary_figure": lambda profiles: FakeFigure("typical"),
            "generate_autocorrelation_lag_error_figure": lambda profiles: FakeFigure("lag"),
            "generate_autocorrelation_pair_score_figure": lambda profiles, score: FakeFigure("pairs"),
        },
    }
    for module_name, attributes in fakes.items():
        module = ModuleType(module_name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, f"nemo_safe_synthesizer.evaluation.components.{module_name}", module)


@pytest.fixture
def report():
    pii_column_statistics = {
        "assigned_type": "email",
        "detected_entity_counts": {"email": 3},
        "detected_entity_values": {"email": {"alice@example.com"}},
        "is_transformed": True,
        "transform_functions": {"hash", "fake"},
    }
    context = {
        "tooltips": {"membership_inference_info": "How likely an attacker can tell a record was used."},
        "job_id": "safe-synth-job",
        "with_synthesizer": True,
        "with_data_privacy": True,
        "with_transform": True,
        "with_time_series": True,
        "dp_enabled": True,
        "epsilon": 8.0,
        "delta": "auto",
        "time_series_cards": [{"name": "duplicated for the template"}],
        "synthetic_quality_score": {
            "name": "Synthetic Quality Score",
            "score": _score(),
            "figure": RENDERED_HTML,
            "figure_overview": RENDERED_HTML,
        },
        "membership_inference_protection": {
            "name": "Membership Inference Protection",
            "score": _score("Excellent", 9.4),
            "anchor_link": "#mia",
            "excluded_columns": ["id"],
            "figure": RENDERED_HTML,
        },
        "attribute_inference_protection": {
            "name": "Attribute Inference Protection",
            "score": _score("Unavailable", None),
            "figure": None,
        },
        "dataset_statistics": {
            "name": "Dataset Statistics",
            "score": _score("Unavailable", None),
            "training_rows": np.int64(100),
            "synthetic_rows": 90,
            "memorized_lines": 0,
        },
        "column_distribution_stability": {
            "name": "Column Distribution Stability",
            "score": _score(),
            "evaluation_fields": [
                {"name": "age", "distribution_distance": 0.1, "column_statistics": None},
                {"name": "email", "distribution_distance": float("nan"), "column_statistics": pii_column_statistics},
            ],
            "column_statistics": {"email": pii_column_statistics},
            "figures": [{"name1": "age", "name2": "email", "figure": RENDERED_HTML}],
        },
        "text_structure_similarity": {
            "name": "Text Structure Similarity",
            "score": _score(),
            "figures": [{"title": "bio", "html": RENDERED_HTML}],
        },
        "pii_replay": {
            "name": "PII Replay",
            "score": _score("Unavailable", None),
            "pii_types": ["email"],
            "pii_replay_data": [{"column_name": "email", "pii_type": "email", "unique_synthetic_data_percentage": 0.0}],
        },
        "autocorrelation_similarity": {
            "name": "Autocorrelation Similarity",
            "score": _score(),
            "details": {"profiles": [{"column": "value", "acf": np.array([1.0, np.nan])}]},
            "template": "jinja/components/autocorrelation_similarity.j2",
            "figures": [
                {
                    "title": "Typical autocorrelation",
                    "html": RENDERED_HTML,
                    "column_select": True,
                    "tooltip_id": "autocorrelation-typical-info",
                },
            ],
        },
    }
    text_statistics = SimpleNamespace(empty=False)
    components = [
        SimpleNamespace(name="Synthetic Quality Score"),
        SimpleNamespace(name="Membership Inference Protection", attack_sum_df=pd.DataFrame({"attack": [1]})),
        SimpleNamespace(name="Attribute Inference Protection", col_accuracy_df=pd.DataFrame()),
        SimpleNamespace(name="Dataset Statistics"),
        SimpleNamespace(name="Column Distribution Stability"),
        SimpleNamespace(
            name="Text Structure Similarity",
            training_statistics={"bio": text_statistics, "notes": SimpleNamespace(empty=True)},
            synthetic_statistics={"bio": text_statistics, "notes": SimpleNamespace(empty=True)},
        ),
        SimpleNamespace(name="PII Replay"),
        SimpleNamespace(
            name="Autocorrelation Similarity",
            details=context["autocorrelation_similarity"]["details"],
            score=SimpleNamespace(score=7.0),
        ),
    ]
    evaluation_datasets = SimpleNamespace(
        get_tabular_columns=lambda: ["age", "email"],
        evaluation_fields=[SimpleNamespace(name="age"), SimpleNamespace(name="bio"), SimpleNamespace(name="email")],
        training=pd.DataFrame({"age": [1], "bio": ["x"], "email": ["e"]}),
        synthetic=pd.DataFrame({"age": [2], "bio": ["y"], "email": ["f"]}),
    )
    return SimpleNamespace(jinja_context=context, components=components, evaluation_datasets=evaluation_datasets)


def test_build_report_data_carries_report_level_fields(report, fake_figure_modules):
    data = build_report_data(report)

    assert data.schema_version == REPORT_DATA_SCHEMA_VERSION
    assert data.library_version == nss_version
    assert data.job_id == "safe-synth-job"
    assert data.flags.model_dump() == {
        "with_synthesizer": True,
        "with_data_privacy": True,
        "with_transform": True,
        "with_time_series": True,
    }
    assert data.differential_privacy.model_dump() == {"enabled": True, "epsilon": 8.0, "delta": "auto"}
    assert data.tooltips["membership_inference_info"].startswith("How likely")
    assert list(data.components) == [
        "synthetic_quality_score",
        "membership_inference_protection",
        "attribute_inference_protection",
        "dataset_statistics",
        "column_distribution_stability",
        "text_structure_similarity",
        "pii_replay",
        "autocorrelation_similarity",
    ]


def test_build_report_data_replaces_rendered_html_with_plotly_json(report, fake_figure_modules):
    payload = build_report_data(report).model_dump(mode="json")
    components = payload["components"]

    assert RENDERED_HTML not in json.dumps(payload, allow_nan=False)
    assert components["synthetic_quality_score"]["score"] == {
        "raw_score": 0.7,
        "grade": "Good",
        "score": 7.0,
        "notes": None,
    }
    assert components["synthetic_quality_score"]["figures"] == []
    assert components["membership_inference_protection"]["excluded_columns"] == ["id"]
    assert components["membership_inference_protection"]["figures"] == [
        {
            "title": None,
            "figure": {"data": [{"type": "bar", "name": "mia", "y": [1, None]}], "layout": {"title": {"text": "mia"}}},
        }
    ]
    assert components["attribute_inference_protection"]["figures"] == []
    assert [f["title"] for f in components["column_distribution_stability"]["figures"]] == ["age", "email"]
    assert [f["title"] for f in components["text_structure_similarity"]["figures"]] == ["bio"]
    assert components["autocorrelation_similarity"]["figures"][0] == {
        "title": "Typical autocorrelation",
        "column_select": True,
        "tooltip_id": "autocorrelation-typical-info",
        "figure": {
            "data": [{"type": "bar", "name": "typical", "y": [1, None]}],
            "layout": {"title": {"text": "typical"}},
        },
    }
    assert [f["title"] for f in components["autocorrelation_similarity"]["figures"]] == [
        "Typical autocorrelation",
        "Difference by lag",
        "Pair scores",
    ]
    assert "template" not in components["autocorrelation_similarity"]


def test_build_report_data_drops_detected_pii_values(report, fake_figure_modules):
    payload = build_report_data(report).model_dump(mode="json")
    column_distribution = payload["components"]["column_distribution_stability"]

    assert "alice@example.com" not in json.dumps(payload)
    assert column_distribution["column_statistics"]["email"] == {
        "assigned_type": "email",
        "detected_entity_counts": {"email": 3},
        "is_transformed": True,
        "transform_functions": ["fake", "hash"],
    }
    assert "detected_entity_values" not in column_distribution["evaluation_fields"][1]["column_statistics"]


def test_build_report_data_emits_strict_json(report, fake_figure_modules):
    payload = json.loads(build_report_data(report).model_dump_json())
    components = payload["components"]

    assert components["dataset_statistics"]["training_rows"] == 100
    assert components["column_distribution_stability"]["evaluation_fields"][1]["distribution_distance"] is None
    assert components["autocorrelation_similarity"]["details"]["profiles"][0]["acf"] == [1.0, None]


def test_to_json_safe_rejects_unknown_types():
    with pytest.raises(TypeError, match="DataFrame"):
        to_json_safe({"frame": pd.DataFrame({"a": [1]})})


def test_evaluation_report_of_returns_the_evaluator_report():
    report = object()

    assert evaluation_report_of(SimpleNamespace(evaluator=SimpleNamespace(report=report))) is report
    assert evaluation_report_of(SimpleNamespace(evaluator=None)) is None
    assert evaluation_report_of(object()) is None
