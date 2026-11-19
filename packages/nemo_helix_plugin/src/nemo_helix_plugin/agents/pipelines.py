# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent pipeline configs: models, static checks, and helpers a pipeline runner uses.

Parsing checks types only and keeps unknown keys; every rule lives in :func:`collect_static_issues`,
which runs on create, update and validate. Stored configs therefore stay readable if a rule tightens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any

from nemo_helix_plugin.agents import pipeline_template as tpl
from nemo_helix_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Discriminator, Field, RootModel, Tag, model_validator

PIPELINE_SUPPORTED_VERSIONS: tuple[int, ...] = (1,)
PIPELINE_MAX_STEPS = 20
PIPELINE_MAX_RETRIES = 5
STEP_NAME_PATTERN = r"^[a-z][a-z0-9_-]{0,62}$"
_STEP_NAME = re.compile(STEP_NAME_PATTERN)
_AGENT_REF = re.compile(ENTITY_REF_PATTERN)
_INPUT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
_CONDITION_OPERATORS = ("equals", "not_equals", "in")
_STEP_KEY_HINT = "Agent settings go in the agent's agent.yaml; run settings are given when the pipeline runs."


class ResponseFormat(StrEnum):
    """How a pipeline reads a step's ``response``."""

    TEXT = "text"
    JSON = "json"


class InputType(StrEnum):
    """Value type of a pipeline input."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class PipelineIssue(BaseModel):
    """One problem found while checking a pipeline config."""

    location: str = Field(description="Where the problem is, e.g. 'steps[reply].input'.")
    message: str = Field(description="What is wrong and how to fix it.")


class PipelineValidationResult(BaseModel):
    """Response of ``POST .../pipeline-configs/validate``."""

    valid: bool = Field(description="True when there are no issues.")
    issues: list[PipelineIssue] = Field(default_factory=list, description="Problems found, in config order.")


class ValidatePipelineConfigRequest(RootModel[dict[str, Any]]):
    """Request body for ``POST .../pipeline-configs/validate``: a pipeline.yaml as a JSON object."""


class PipelineInputParam(BaseModel):
    """A named, typed pipeline input."""

    model_config = ConfigDict(extra="allow")

    type: InputType = Field(default=InputType.STRING, description="Value type of the input.")
    description: str = Field(default="", description="What the input is.")
    required: bool | None = Field(default=None, description="Whether a run must supply it. Default: true.")
    default: Any = Field(default=None, description="Value used when a run omits the input.")

    @property
    def is_required(self) -> bool:
        return self.required if self.required is not None else self.default is None


class _ConditionBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = Field(description="A placeholder path without braces, e.g. 'steps.triage.response.verdict'.")


class ConditionEquals(_ConditionBase):
    """Run the step when the value at ``path`` equals ``equals``."""

    equals: Any = Field(description="Run when the value equals this.")

    def matches(self, value: Any) -> bool:
        return value == self.equals


class ConditionNotEquals(_ConditionBase):
    """Run the step when the value at ``path`` differs from ``not_equals``."""

    not_equals: Any = Field(description="Run when the value differs from this.")

    def matches(self, value: Any) -> bool:
        return value != self.not_equals


class ConditionIn(_ConditionBase):
    """Run the step when the value at ``path`` is one of ``in``."""

    model_config = ConfigDict(extra="allow", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    in_: list[Any] = Field(alias="in", description="Run when the value is one of these.")

    def matches(self, value: Any) -> bool:
        return value in self.in_


def _condition_operator(value: Any) -> str | None:
    """Pick the condition shape from its operator key, so errors come from that shape only."""
    if isinstance(value, BaseModel):
        return {ConditionEquals: "equals", ConditionNotEquals: "not_equals", ConditionIn: "in"}.get(type(value))
    if isinstance(value, Mapping):
        for operator in ("equals", "not_equals"):
            if operator in value:
                return operator
        if "in" in value or "in_" in value:
            return "in"
    return None


StepCondition = Annotated[
    Annotated[ConditionEquals, Tag("equals")]
    | Annotated[ConditionNotEquals, Tag("not_equals")]
    | Annotated[ConditionIn, Tag("in")],
    Discriminator(
        _condition_operator,
        custom_error_type="condition_operator",
        custom_error_message="Set exactly one of 'equals', 'not_equals' or 'in'.",
    ),
]


class PipelineStepConfig(BaseModel):
    """One agent run in a pipeline."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Step name, unique within the pipeline.")
    agent: str = Field(description="The registered Agent to run: 'name' or 'workspace/name'.")
    input: str | dict[str, Any] | None = Field(
        default=None, description="Text template or JSON object of templates for the agent's input."
    )
    response_artifact: str | None = Field(
        default=None, description="Optional relative file in output_workdir to use instead of the final text response."
    )
    expect: StepCondition | None = Field(
        default=None,
        description="Required output condition, evaluated against response, e.g. response.status equals ready.",
    )
    context: dict[str, Any] = Field(default_factory=dict, description="Per-run data for the agent; may hold templates.")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.TEXT, description="'json' lets later steps read fields of the response."
    )
    when: StepCondition | None = Field(default=None, description="Run the step only when this condition holds.")
    retries: int = Field(default=0, description="Retries after a failure.")

    @model_validator(mode="before")
    @classmethod
    def _no_inline_agent(cls, data: Any) -> Any:
        # Without this, an inline agent object would fail with a bare "Input should be a valid string".
        if isinstance(data, Mapping) and isinstance(data.get("agent"), Mapping):
            raise ValueError(
                "agent must reference a registered Agent ('name' or 'workspace/name'); register it from its "
                "agent.yaml first."
            )
        return data


class PipelineInline(BaseModel):
    """A pipeline config without entity identity: the body of ``pipeline.yaml`` minus ``name``."""

    model_config = ConfigDict(extra="allow")

    version: int = Field(description="Pipeline format version (1).")
    description: str = Field(default="", description="What the pipeline does.")
    inputs: dict[str, PipelineInputParam] = Field(
        default_factory=dict, description="Named inputs a run supplies, referenced as {{ inputs.<name> }}."
    )
    steps: list[PipelineStepConfig] = Field(description="Steps to run, in order.")


class _StaticallyChecked(PipelineInline):
    """Rejects configs that fail the static checks. Used by request models only."""

    @model_validator(mode="after")
    def _static_checks(self) -> _StaticallyChecked:
        issues = collect_static_issues(self)
        if issues:
            raise ValueError("; ".join(f"{issue.location}: {issue.message}" for issue in issues))
        return self


class PipelineConfig(_StaticallyChecked):
    """A complete ``pipeline.yaml``: the pipeline body plus its name."""

    model_config = ConfigDict(extra="allow", regex_engine="python-re")

    name: str = Field(pattern=NAME_PATTERN, description=f"Pipeline name. {NAME_PATTERN_DESCRIPTION}")


class PipelineConfigUpdate(_StaticallyChecked):
    """A replacement pipeline body. ``name`` is optional and must match the stored name."""

    name: str | None = Field(default=None, description="Must equal the pipeline name in the path when set.")


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def collect_static_issues(pipeline: PipelineInline) -> list[PipelineIssue]:
    """Every problem in *pipeline* that can be found without looking anything up in a workspace."""
    if pipeline.version not in PIPELINE_SUPPORTED_VERSIONS:
        supported = ", ".join(str(v) for v in PIPELINE_SUPPORTED_VERSIONS)
        return [PipelineIssue(location="version", message=f"Unsupported version {pipeline.version}; use {supported}.")]
    issues = _shape_issues(pipeline) + _input_issues(pipeline)

    seen: dict[str, int] = {}
    for index, step in enumerate(pipeline.steps):
        if step.name in seen:
            issues.append(
                PipelineIssue(
                    location=f"steps[{index}].name",
                    message=f"Duplicate step name {step.name!r} (also step {seen[step.name]}).",
                )
            )
        seen.setdefault(step.name, index)

    for index, step in enumerate(pipeline.steps):
        where = f"steps[{step.name}]"
        template = step_input_template(pipeline, index)
        if template is None:
            if pipeline.inputs:  # with no inputs at all, the missing-inputs issue already says what to fix
                issues.append(
                    PipelineIssue(
                        location=f"{where}.input",
                        message="The first step must set input: there is no single text input to use by default.",
                    )
                )
        else:
            for location, value in tpl.tree_strings(template, f"{where}.input"):
                issues.extend(_template_issues(pipeline, index, value, location))
        for location, value in tpl.tree_strings(step.context, f"{where}.context"):
            issues.extend(_template_issues(pipeline, index, value, location))
        if step.when is not None:
            try:
                problem = _path_problem(pipeline, index, tpl.parse_path(step.when.path))
            except tpl.TemplateSyntaxError as exc:
                problem = str(exc)
            if problem:
                issues.append(PipelineIssue(location=f"{where}.when.path", message=problem))
    return issues


def _unknown_keys(
    model: BaseModel, location: str, *, ignore: tuple[str, ...] = (), hint: str = ""
) -> list[PipelineIssue]:
    unknown = sorted(set(model.model_extra or {}) - set(ignore))
    if not unknown:
        return []
    return [PipelineIssue(location=location, message=f"Unknown key(s): {', '.join(unknown)}. {hint}".rstrip())]


def _shape_issues(pipeline: PipelineInline) -> list[PipelineIssue]:
    """Rules on individual fields: unknown keys, limits and name patterns."""
    issues = _unknown_keys(pipeline, "pipeline")
    if not 1 <= len(pipeline.steps) <= PIPELINE_MAX_STEPS:
        issues.append(PipelineIssue(location="steps", message=f"A pipeline has 1 to {PIPELINE_MAX_STEPS} steps."))
    for index, step in enumerate(pipeline.steps):
        where = f"steps[{index}]"
        issues.extend(_unknown_keys(step, where, hint=_STEP_KEY_HINT))
        if not _STEP_NAME.fullmatch(step.name):
            issues.append(
                PipelineIssue(
                    location=f"{where}.name",
                    message="Step names use lowercase letters, digits, '-' and '_', start with a letter, and have no '.'.",
                )
            )
        if not _AGENT_REF.fullmatch(step.agent):
            issues.append(PipelineIssue(location=f"{where}.agent", message="Use 'name' or 'workspace/name'."))
        if not 0 <= step.retries <= PIPELINE_MAX_RETRIES:
            issues.append(
                PipelineIssue(location=f"{where}.retries", message=f"retries is between 0 and {PIPELINE_MAX_RETRIES}.")
            )
        if step.response_artifact is not None:
            path = step.response_artifact
            if (
                not path
                or PurePosixPath(path).is_absolute()
                or ".." in path.split("/")
                or "\\" in path
                or "\x00" in path
                or PurePosixPath(path).as_posix() != path
                or path == "."
            ):
                issues.append(
                    PipelineIssue(
                        location=f"{where}.response_artifact",
                        message="Use a normalized relative file path inside output_workdir.",
                    )
                )
        if step.expect is not None:
            issues.extend(_unknown_keys(step.expect, f"{where}.expect"))
            try:
                path = tpl.parse_path(step.expect.path)
                if path[0] != "response":
                    raise ValueError("Output checks must start with response.")
                if len(path) > 1 and step.response_format is not ResponseFormat.JSON:
                    raise ValueError("Set response_format: json to check response fields.")
            except ValueError as exc:
                issues.append(PipelineIssue(location=f"{where}.expect.path", message=str(exc)))
        if step.when is not None:
            if any(op in (step.when.model_extra or {}) for op in _CONDITION_OPERATORS):
                issues.append(
                    PipelineIssue(
                        location=f"{where}.when", message="Set exactly one of 'equals', 'not_equals' or 'in'."
                    )
                )
            issues.extend(_unknown_keys(step.when, f"{where}.when", ignore=_CONDITION_OPERATORS))
    return issues


def _input_issues(pipeline: PipelineInline) -> list[PipelineIssue]:
    if not pipeline.inputs:
        return [PipelineIssue(location="inputs", message="Declare at least one input, e.g. inputs: {input: {}}.")]
    issues = []
    for name, param in pipeline.inputs.items():
        where = f"inputs.{name}"
        issues.extend(_unknown_keys(param, where))
        if not _INPUT_NAME.fullmatch(name):
            issues.append(
                PipelineIssue(
                    location=where, message="Input names use letters, digits and '_', not starting with a digit."
                )
            )
        if param.default is not None:
            if param.required:
                issues.append(
                    PipelineIssue(location=where, message="An input cannot be both required and have a default.")
                )
            if not _matches_type(param.default, param.type):
                issues.append(
                    PipelineIssue(location=f"{where}.default", message=f"default must be of type {param.type.value}.")
                )
    return issues


def _template_issues(pipeline: PipelineInline, index: int, template: str, location: str) -> list[PipelineIssue]:
    try:
        found = tpl.placeholders(template)
    except tpl.TemplateSyntaxError as exc:
        return [PipelineIssue(location=location, message=str(exc))]
    return [
        PipelineIssue(location=location, message=f"{{{{ {placeholder.source} }}}}: {problem}")
        for placeholder in found
        if (problem := _path_problem(pipeline, index, placeholder.path))
    ]


def _path_problem(pipeline: PipelineInline, index: int, path: tuple[tpl.PathSegment, ...]) -> str | None:
    """Why *path* cannot be used in step *index*, or None when it can."""
    if path[0] == "input":  # {{ input }} is short for {{ inputs.input }}
        path = ("inputs", *path)
    root, rest = path[0], path[1:]
    if root == "inputs":
        if not rest or not isinstance(rest[0], str):
            return "Name an input, e.g. inputs.email."
        param = pipeline.inputs.get(rest[0])
        if param is None:
            return f"Unknown input {rest[0]!r}."
        if len(rest) > 1 and param.type not in (InputType.OBJECT, InputType.ARRAY):
            return f"Input {rest[0]!r} is a {param.type.value}, so it has no fields."
        return None
    if root == "previous":
        if index == 0:
            return "'previous' is not available in the first step."
        if not rest or rest[0] != "response":
            return "Only previous.response is available; use steps.<name>.output for the full output."
        return _step_reference_problem(pipeline, index - 1, rest)
    if root == "steps":
        if not rest or not isinstance(rest[0], str):
            return "Name a step, e.g. steps.triage.response."
        names = [step.name for step in pipeline.steps]
        if rest[0] not in names:
            return f"Unknown step {rest[0]!r}."
        target = names.index(rest[0])
        if target == index:
            return "A step cannot reference itself."
        if target > index:
            return f"Step {rest[0]!r} runs later; only earlier steps can be referenced."
        if len(rest) < 2 or rest[1] not in ("response", "output"):
            return f"Use steps.{rest[0]}.response or steps.{rest[0]}.output."
        return _step_reference_problem(pipeline, target, rest[1:])
    return f"Unknown name {root!r}; placeholders start with inputs, input, previous or steps."


def _step_reference_problem(pipeline: PipelineInline, target: int, rest: tuple[tpl.PathSegment, ...]) -> str | None:
    """Problems shared by steps.<x>.* and previous.*: *rest* starts with 'response' or 'output'."""
    step = pipeline.steps[target]
    if step.when is not None:
        return f"Step {step.name!r} has a 'when' condition and may be skipped, so it cannot be referenced."
    if rest[0] == "response" and len(rest) > 1 and step.response_format is not ResponseFormat.JSON:
        return f"Set response_format: json on step {step.name!r} to read fields of its response."
    return None


def _matches_type(value: Any, input_type: InputType) -> bool:
    if input_type is InputType.STRING:
        return isinstance(value, str)
    if input_type is InputType.BOOLEAN:
        return isinstance(value, bool)
    if input_type is InputType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if input_type is InputType.OBJECT:
        return isinstance(value, dict)
    return isinstance(value, list)


# ---------------------------------------------------------------------------
# Helpers for a pipeline runner
# ---------------------------------------------------------------------------


def step_input_template(pipeline: PipelineInline, index: int) -> str | dict[str, Any] | None:
    """The input template step *index* uses: its own, or the default. None when there is no default.

    Defaults: the first step gets the pipeline's single text input; later steps get the previous response.
    """
    step = pipeline.steps[index]
    if step.input is not None:
        return step.input
    if index > 0:
        return "{{ previous.response }}"
    text_inputs = [name for name, param in pipeline.inputs.items() if param.type is InputType.STRING]
    return f"{{{{ inputs.{text_inputs[0]} }}}}" if len(text_inputs) == 1 else None


def ensure_runnable(pipeline: PipelineInline) -> None:
    """Raise ``ValueError`` if a stored *pipeline* fails the static checks; runners call this first."""
    issues = collect_static_issues(pipeline)
    if issues:
        raise ValueError(
            "The pipeline config cannot run: " + "; ".join(f"{issue.location}: {issue.message}" for issue in issues)
        )


def resolve_run_inputs(pipeline: PipelineInline, supplied: Any) -> dict[str, Any]:
    """Check *pipeline* can run and the run's inputs match it; return the render context for the inputs.

    *supplied* is a mapping of input values, or a string for the pipeline's single text input.
    """
    ensure_runnable(pipeline)
    text_inputs = [name for name, param in pipeline.inputs.items() if param.type is InputType.STRING]
    if isinstance(supplied, str) and len(text_inputs) == 1:
        supplied = {text_inputs[0]: supplied}
    if not isinstance(supplied, Mapping):
        raise ValueError("Pass the inputs as an object of input values.")
    unknown = sorted(set(supplied) - set(pipeline.inputs))
    if unknown:
        raise ValueError(f"Unknown inputs: {', '.join(unknown)}.")
    values: dict[str, Any] = {}
    for name, param in pipeline.inputs.items():
        if name in supplied:
            if not _matches_type(supplied[name], param.type):
                raise ValueError(f"Input {name!r} must be of type {param.type.value}.")
            values[name] = supplied[name]
        elif param.is_required:
            raise ValueError(f"Missing required input {name!r}.")
        else:
            values[name] = param.default
    context: dict[str, Any] = {"inputs": values}
    if "input" in values:
        context["input"] = values["input"]
    return context


def parse_step_response(step: PipelineStepConfig, response: Any) -> Any:
    """The value later steps see as ``steps.<name>.response``, applying the step's response_format."""
    if step.response_format is ResponseFormat.TEXT or not isinstance(response, str):
        return response
    try:
        return json.loads(response)
    except ValueError as exc:
        raise ValueError(f"Step {step.name!r} has response_format json, but its response is not valid JSON.") from exc


def render_step(pipeline: PipelineInline, index: int, context: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The ``(input, context)`` to send to step *index*'s agent.

    *context* holds the inputs (from :func:`resolve_run_inputs`), ``steps`` (each finished step's
    ``{"response": ..., "output": ...}``) and ``previous`` (the last finished step's).
    """
    template = step_input_template(pipeline, index)
    if template is None:
        raise ValueError(f"Step {pipeline.steps[index].name!r} has no input.")
    prompt = (
        tpl.render(template, context) if isinstance(template, str) else json.dumps(tpl.render_tree(template, context))
    )
    return prompt, tpl.render_tree(pipeline.steps[index].context, context)


def should_run(step: PipelineStepConfig, context: Mapping[str, Any]) -> bool:
    """Whether *step* runs, given its ``when`` condition. A step without a condition always runs."""
    if step.when is None:
        return True
    return step.when.matches(tpl.resolve(tpl.parse_path(step.when.path), context))
