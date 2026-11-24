# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data behind the Safe Synthesizer evaluation report, for rendering it outside the HTML template."""

import datetime
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd
from nemo_safe_synthesizer.package_info import __version__ as nss_version
from pydantic import BaseModel, ConfigDict, Field

REPORT_DATA_RESULT_NAME = "evaluation-report-data"
REPORT_DATA_SCHEMA_VERSION = 1

_RENDERED_HTML_KEYS = frozenset({"figure", "figure_overview", "figures", "html", "template"})
_PII_VALUE_KEYS = frozenset({"detected_entity_values"})

# Field profiles also hold every distinct raw training value, so only the keys the report shows are kept.
_EVALUATION_FIELD_KEYS = (
    "name",
    "distribution_distance",
    "distribution_stability",
    "training_field_features",
    "synthetic_field_features",
    "column_statistics",
)
_FIELD_FEATURE_KEYS = (
    "type",
    "count",
    "unique_count",
    "unique_percent",
    "missing_count",
    "missing_percent",
    "avg_str_length",
)
_COLUMN_STATISTICS_KEYS = (
    "assigned_type",
    "assigned_entity",
    "detected_entity_counts",
    "is_transformed",
    "transform_functions",
)

_FIGURES_MODULE = "nemo_safe_synthesizer.evaluation.components.multi_modal_figures"
_AUTOCORRELATION_FIGURES_MODULE = "nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures"
_COLUMN_DISTRIBUTION_MODULE = "nemo_safe_synthesizer.evaluation.components.column_distribution"


class EvaluationReportLike(Protocol):
    """The parts of the library's ``MultimodalReport`` the payload is built from."""

    components: Sequence[Any]
    evaluation_datasets: Any

    @property
    def jinja_context(self) -> dict[str, Any]: ...


class ReportScore(BaseModel):
    """A component score as the library reports it."""

    raw_score: float | None = None
    grade: str = Field(description="Qualitative grade, e.g. 'Unavailable', 'Poor', 'Good', 'Excellent'.")
    score: float | None = Field(default=None, description="Scaled score from 0 to 10.")
    notes: str | None = None


class ReportFigure(BaseModel):
    """One chart in a report card, as Plotly figure JSON."""

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    figure: dict[str, Any] = Field(description="Plotly figure JSON with `data` and `layout`; layout.template removed.")


class ReportCard(BaseModel):
    """One report component: its score, charts, and the component-specific fields the HTML report renders."""

    model_config = ConfigDict(extra="allow")

    name: str
    score: ReportScore
    figures: list[ReportFigure] = Field(default_factory=list)


class ReportFlags(BaseModel):
    """Which report sections have data."""

    with_synthesizer: bool = False
    with_data_privacy: bool = False
    with_transform: bool = False
    with_time_series: bool = False


class ReportDifferentialPrivacy(BaseModel):
    enabled: bool = False
    epsilon: float | None = None
    delta: float | str | None = None


class SafeSynthesizerReportData(BaseModel):
    """Everything the Safe Synthesizer Quality & Privacy report shows, keyed by component."""

    schema_version: int = REPORT_DATA_SCHEMA_VERSION
    library_version: str | None = Field(default=None, description="nemo-safe-synthesizer version that built it.")
    job_id: str | None = None
    flags: ReportFlags = Field(default_factory=ReportFlags)
    differential_privacy: ReportDifferentialPrivacy = Field(default_factory=ReportDifferentialPrivacy)
    tooltips: dict[str, str] = Field(default_factory=dict)
    components: dict[str, ReportCard] = Field(
        default_factory=dict,
        description="Cards keyed by snake-cased component name, e.g. `synthetic_quality_score`, in report order.",
    )


FigureEntry = tuple[str | None, Any]
FigureBuilder = Callable[[Any, EvaluationReportLike], list[FigureEntry]]


def evaluation_report_of(ss: object) -> EvaluationReportLike | None:
    """Return the report a ``SafeSynthesizer`` run evaluated, or ``None`` when evaluation did not run."""
    evaluator = getattr(ss, "evaluator", None)
    return getattr(evaluator, "report", None)


def build_report_data(report: EvaluationReportLike) -> SafeSynthesizerReportData:
    """Build the report payload from the evaluation report the job just rendered to HTML."""
    context = report.jinja_context
    components: dict[str, ReportCard] = {}
    for component in report.components:
        key = _component_key(component.name)
        builder = _FIGURE_BUILDERS.get(key)
        figures = builder(component, report) if builder is not None else []
        components[key] = _build_card(context[key], figures)

    return SafeSynthesizerReportData(
        library_version=nss_version,
        job_id=context.get("job_id"),
        flags=ReportFlags(**{name: bool(context.get(name)) for name in ReportFlags.model_fields}),
        differential_privacy=ReportDifferentialPrivacy(
            enabled=bool(context.get("dp_enabled")),
            epsilon=context.get("epsilon"),
            delta=context.get("delta"),
        ),
        tooltips={str(key): str(value) for key, value in context.get("tooltips", {}).items()},
        components=components,
    )


def _component_key(name: str) -> str:
    # Same keying as the library's EvaluationReport.jinja_context.
    return re.sub(" ", "_", name).lower()


def _build_card(card_context: Mapping[str, Any], figures: list[FigureEntry]) -> ReportCard:
    fields = {
        key: to_json_safe(_allowlisted(key, value))
        for key, value in card_context.items()
        if key not in _RENDERED_HTML_KEYS
    }
    score = card_context["score"]
    fields["score"] = to_json_safe({key: value for key, value in score.items() if key != "figure"})

    row_extras = {
        row["title"]: {key: to_json_safe(value) for key, value in row.items() if key not in _RENDERED_HTML_KEYS}
        for row in card_context.get("figures") or []
        if isinstance(row, Mapping) and "html" in row
    }
    fields["figures"] = [
        {**row_extras.get(title, {}), "title": title, "figure": figure_to_json(figure)} for title, figure in figures
    ]
    return ReportCard.model_validate(fields)


def _allowlisted(key: str, value: Any) -> Any:
    if key == "evaluation_fields":
        return [_evaluation_field(field) for field in value or []]
    if key == "column_statistics":
        return {column: _pick(statistics, _COLUMN_STATISTICS_KEYS) for column, statistics in (value or {}).items()}
    return value


def _evaluation_field(field: Mapping[str, Any]) -> dict[str, Any] | None:
    picked = _pick(field, _EVALUATION_FIELD_KEYS)
    if picked is None:
        return None
    for features_key in ("training_field_features", "synthetic_field_features"):
        if features_key in picked:
            picked[features_key] = _pick(picked[features_key], _FIELD_FEATURE_KEYS)
    if "column_statistics" in picked:
        picked["column_statistics"] = _pick(picked["column_statistics"], _COLUMN_STATISTICS_KEYS)
    return picked


def _pick(value: Mapping[str, Any] | None, keys: Sequence[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: value[key] for key in keys if key in value}


def figure_to_json(figure: Any) -> dict[str, Any]:
    """Serialize a Plotly figure with Plotly's own encoder, dropping the theme template."""
    data: dict[str, Any] = json.loads(figure.to_json(validate=False))
    layout = data.get("layout")
    if isinstance(layout, dict):
        layout.pop("template", None)
    return data


def to_json_safe(value: Any) -> Any:
    """Convert library context values to strict JSON, dropping raw detected PII values."""
    if isinstance(value, BaseModel):
        return to_json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items() if key not in _PII_VALUE_KEYS}
    if isinstance(value, (set, frozenset)):
        return sorted((to_json_safe(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    elif isinstance(value, (np.timedelta64, datetime.timedelta)):
        value = pd.Timedelta(value)
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} in evaluation report data")


def _nonempty(*frames: pd.DataFrame | None) -> bool:
    return all(frame is not None and not frame.empty for frame in frames)


def _figures_module() -> Any:
    return importlib.import_module(_FIGURES_MODULE)


def _attribute_inference_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    if not _nonempty(component.col_accuracy_df):
        return []
    return [(None, _figures_module().generate_aia_figure(component.col_accuracy_df))]


def _membership_inference_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    if not _nonempty(component.attack_sum_df):
        return []
    return [(None, _figures_module().generate_mia_figure(df=component.attack_sum_df))]


def _correlation_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    if not _nonempty(component.training_correlation, component.synthetic_correlation, component.correlation_difference):
        return []
    figure = _figures_module().generate_combined_correlation_figure(
        training_correlation_df=component.training_correlation,
        synthetic_correlation_df=component.synthetic_correlation,
        correlation_difference=component.correlation_difference,
    )
    return [(None, figure)]


def _deep_structure_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    if not _nonempty(component.training_pca, component.synthetic_pca):
        return []
    figure = _figures_module().structure_stability_figure(
        training_df=component.training_pca,
        synthetic_df=component.synthetic_pca,
    )
    return [(None, figure)]


def _text_semantic_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    figures = _figures_module()
    entries: list[FigureEntry] = []
    for column, datum in (component.text_semantic_similarity_dict or {}).items():
        figure = figures.generate_text_semantic_similarity_figures(
            datum.training_pca.iloc[:, :2],
            datum.synthetic_pca.iloc[:, :2],
        )
        if figure is not None:
            entries.append((column, figure))
    return entries


def _text_structure_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    figures = _figures_module()
    entries: list[FigureEntry] = []
    for column, training_statistics in (component.training_statistics or {}).items():
        figure = figures.generate_text_structure_similarity_figures(
            training_statistics, component.synthetic_statistics[column]
        )
        if figure is not None:
            entries.append((column, figure))
    return entries


def _column_distribution_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    plot_row = importlib.import_module(_COLUMN_DISTRIBUTION_MODULE).ColumnDistributionPlotRow
    datasets = report.evaluation_datasets
    tabular_columns = set(datasets.get_tabular_columns())
    entries: list[FigureEntry] = []
    for field in datasets.evaluation_fields:
        if field.name not in tabular_columns:
            continue
        figure = plot_row._get_figure_for_field(field, datasets.training[field.name], datasets.synthetic[field.name])
        if figure is not None:
            entries.append((field.name, figure))
    return entries


def _autocorrelation_figures(component: Any, report: EvaluationReportLike) -> list[FigureEntry]:
    profiles = component.details.get("profiles", [])
    if not profiles:
        return []
    figures = importlib.import_module(_AUTOCORRELATION_FIGURES_MODULE)
    return [
        ("Typical autocorrelation", figures.generate_autocorrelation_summary_figure(profiles)),
        ("Difference by lag", figures.generate_autocorrelation_lag_error_figure(profiles)),
        ("Pair scores", figures.generate_autocorrelation_pair_score_figure(profiles, component.score.score)),
    ]


_FIGURE_BUILDERS: dict[str, FigureBuilder] = {
    "attribute_inference_protection": _attribute_inference_figures,
    "membership_inference_protection": _membership_inference_figures,
    "column_correlation_stability": _correlation_figures,
    "deep_structure_stability": _deep_structure_figures,
    "text_semantic_similarity": _text_semantic_figures,
    "text_structure_similarity": _text_structure_figures,
    "column_distribution_stability": _column_distribution_figures,
    "autocorrelation_similarity": _autocorrelation_figures,
}
