# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface for the evaluator plugin scaffold."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin

import typer
from nemo_evaluator_sdk.metrics.types import MetricVariants
from nemo_evaluator_sdk.values.metrics import _RAGASBase
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.job import NemoJob
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

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob

        if job_cls is not AgentEvalJob:
            return
        for command in group.registered_commands:
            if command.name in {"submit", job_cls.name} and command.callback is not None:
                command.callback = _with_agent_source(command.callback, job_cls)

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


def _with_agent_source(callback: Callable[..., Any], job_cls: type[NemoJob]) -> Callable[..., Any]:
    """Extend generated submission without changing its spec/flag precedence or transport."""
    from nemo_platform_plugin import commands
    from nemo_platform_plugin._spec_flags import UNSET, build_overlay, deep_merge, walk_spec_leaves

    leaves = walk_spec_leaves(commands._job_input_schema(job_cls), reserved=commands._JOB_SUBMIT_RESERVED_FLAGS)

    def submit(typer_ctx: typer.Context, **kwargs: Any) -> Any:
        agent_dir = kwargs.pop("agent_dir", None)
        if agent_dir is not None:
            from nemo_evaluator.harbor.agent_source import publish_agent_source, validate_import_path
            from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, HarborRunnerTarget
            from nemo_platform_plugin.files.client import FilesClient

            base = commands._load_spec(
                str(kwargs["config"]) if kwargs.get("config") is not None else str(kwargs.get("spec", "{}")),
                kwargs.get("config_file") if kwargs.get("config_file") is not None else kwargs.get("spec_file"),
            )
            spec = AgentEvalInputSpec.model_validate(
                deep_merge(base, build_overlay(leaves, kwargs, unset_sentinel=UNSET))
            )
            if not isinstance(spec.target, HarborRunnerTarget) or not spec.target.agent_import_path:
                raise typer.BadParameter("--agent-dir requires a Harbor target with agent_import_path")
            if spec.target.agent_source is not None:
                raise typer.BadParameter("--agent-dir conflicts with target.agent_source")
            validate_import_path(spec.target.agent_import_path)
            commands._merge_options_inputs(kwargs.get("options", []), kwargs.get("options_file"))
            workspace = kwargs.get("workspace", "default")
            client = FilesClient(
                base_url=commands._resolve_submit_base_url(
                    typer_ctx, base_url=kwargs.get("base_url"), cluster=kwargs.get("cluster")
                ),
                workspace=workspace,
                default_headers=commands._resolve_submit_auth_headers(typer_ctx),
            )
            try:
                spec.target.agent_source = publish_agent_source(
                    Path(agent_dir), files_client=client, fileset_ref=f"{workspace}/harbor-agent-sources"
                )
            finally:
                client.close()
            typer.echo(
                f"Verified agent source retained for reuse: {spec.target.agent_source.model_dump_json()}",
                err=True,
            )
            kwargs["spec"] = spec.model_dump_json()
            kwargs["config"] = None
            kwargs["spec_file"] = None
            kwargs["config_file"] = None
            # The merged spec now owns all leaf values; avoid a second overlay undoing publication.
            for leaf in leaves:
                kwargs.pop(leaf.param_name, None)
        try:
            return callback(typer_ctx, **kwargs)
        except Exception:
            if agent_dir is not None:
                typer.echo(
                    "Job creation may have succeeded; inspect job state before resubmitting. "
                    "The verified agent source above remains available for reuse.",
                    err=True,
                )
            raise

    parameters = list(inspect.signature(callback).parameters.values())
    parameters.append(
        inspect.Parameter(
            "agent_dir",
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=Path | None,
            default=typer.Option(
                None, "--agent-dir", help="Capture custom Harbor source relative to the current directory."
            ),
        )
    )
    setattr(submit, "__signature__", inspect.Signature(parameters))
    return submit
