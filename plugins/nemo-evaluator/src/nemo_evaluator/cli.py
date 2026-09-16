# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface for the evaluator plugin scaffold."""

from __future__ import annotations

import inspect
import json
from enum import Enum
from types import UnionType
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin

import typer
from nemo_evaluator_sdk.metrics.types import MetricVariants
from nemo_evaluator_sdk.values.metrics import _RAGASBase
from nemo_helix_plugin.cli import NemoCLI
from pydantic import BaseModel


def _unwrap_metric_model_classes(type_hint: object) -> list[type[BaseModel]]:
    """Return Pydantic model classes from an annotated metric union."""
    origin = get_origin(type_hint)
    if origin is Annotated:
        return _unwrap_metric_model_classes(get_args(type_hint)[0])
    if origin in {Union, UnionType}:
        model_classes: list[type[BaseModel]] = []
        for union_member in get_args(type_hint):
            model_classes.extend(_unwrap_metric_model_classes(union_member))
        return model_classes
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return [type_hint]
    return []


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _metric_type_values(model_cls: type[BaseModel]) -> list[str]:
    type_field = model_cls.model_fields["type"]
    annotation_args = get_args(type_field.annotation)
    if annotation_args:
        return [str(_json_value(value)) for value in annotation_args]
    return [str(_json_value(type_field.default))]


def _metric_type_models() -> dict[str, type[BaseModel]]:
    metric_types: dict[str, type[BaseModel]] = {}
    for model_cls in _unwrap_metric_model_classes(MetricVariants):
        for metric_type in _metric_type_values(model_cls):
            existing = metric_types.get(metric_type)
            if existing is not None and existing is not model_cls:
                raise ValueError(
                    f"Duplicate metric type '{metric_type}' mapped to both {existing.__name__} and {model_cls.__name__}"
                )
            metric_types[metric_type] = model_cls
    return dict(sorted(metric_types.items()))


def _is_ragas_metric(model_cls: type[BaseModel]) -> bool:
    return issubclass(model_cls, _RAGASBase)


def _metric_type_entries() -> list[dict[str, str]]:
    return [
        {
            "name": metric_type,
            "description": inspect.getdoc(model_cls) or "",
        }
        for metric_type, model_cls in sorted(
            _metric_type_models().items(),
            key=lambda item: (_is_ragas_metric(item[1]), item[0]),
        )
    ]


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2))


class EvaluatorPluginCLI(NemoCLI):
    """CLI surface for the evaluator plugin scaffold."""

    name: ClassVar[str] = "evaluator"
    description: ClassVar[str] = "Evaluator plugin commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(
            name=self.name,
            help=self.description,
            no_args_is_help=True,
        )

        @app.command("info")
        def info() -> None:
            """Print the current plugin status."""
            _echo_json(
                {
                    "plugin": self.name,
                    "status": "ready",
                    "service": "/apis/evaluator/v1/healthz",
                    "jobs": [
                        "evaluator.evaluate",
                        "evaluator.agent-evaluate",
                        "evaluator.retrieve-eval",
                    ],
                    "sdk": "nemo_evaluator_sdk.Evaluator",
                }
            )

        @app.command("metric-types")
        def metric_types(
            metric_types_name: str | None = typer.Argument(None, metavar="<metric-name>"),
        ) -> None:
            """Print available evaluator metric names or a metric JSON schema."""
            if metric_types_name is None:
                _echo_json({"metric_types": _metric_type_entries()})
                return

            metric_types_map = _metric_type_models()
            model_cls = metric_types_map.get(metric_types_name)
            if model_cls is None:
                typer.echo(
                    f"Unknown metric name '{metric_types_name}'. Run `nemo evaluator metric-types` to list available metric names.",
                    err=True,
                )
                raise typer.Exit(code=1)
            _echo_json(model_cls.model_json_schema())

        return app
