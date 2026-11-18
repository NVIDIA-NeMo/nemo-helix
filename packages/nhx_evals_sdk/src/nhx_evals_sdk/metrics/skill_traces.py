# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checks over adapter-supplied, ordered skill/tool evidence."""

from collections.abc import Iterable
from fnmatch import fnmatchcase
from typing import Literal, Self

from jsonschema import validators
from jsonschema.exceptions import SchemaError
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.metrics.protocol import MetricDiagnostic, MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.skill_inputs import mapping_at_path, trial_trace
from nhx_evals_sdk.metrics.skills import SkillActivationMetric, _unmeasured
from nhx_evals_sdk.values.metrics import MetricBase
from nhx_evals_sdk.values.skill_evidence import SkillToolCall, SkillToolTrace
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def _matches(call: SkillToolCall, pattern: str) -> bool:
    return fnmatchcase(call.component or call.tool, pattern)


def _result(
    trace: SkillToolTrace, values: dict[str, int | float], diagnostics: list[MetricDiagnostic] | None = None
) -> MetricResult:
    return MetricResult(
        outputs=[
            MetricOutput(name=name, value=value)
            for name, value in {"complete": float(trace.complete), **values}.items()
        ],
        diagnostics=diagnostics or [],
    )


def _spec(counts: Iterable[str] = (), scores: Iterable[str] = ()) -> list[MetricOutputSpec]:
    return [
        MetricOutputSpec.continuous_score("complete", required=False),
        *(MetricOutputSpec.discrete_score(name, required=False) for name in counts),
        *(MetricOutputSpec.continuous_score(name, required=False) for name in scores),
    ]


class SkillOrderRule(BaseModel):
    """Every matching 'after' event needs an earlier 'before' event; both are required."""

    model_config = ConfigDict(extra="forbid")
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)


def _ordered(trace: SkillToolTrace, rule: SkillOrderRule) -> bool:
    before = [call.step for call in trace.calls if _matches(call, rule.before)]
    after = [call.step for call in trace.calls if _matches(call, rule.after)]
    return bool(before and after) and all(any(first < second for first in before) for second in after)


class SkillOveractivationMetric(MetricBase):
    """Report activation on an authored negative case, using the first PR's activation contract."""

    type: Literal[MetricType.SKILL_OVERACTIVATION] = MetricType.SKILL_OVERACTIVATION
    skill_name: str = Field(min_length=1)

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score(name, required=False) for name in ("overactivated", "as_expected")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        result = await SkillActivationMetric(skill_name=self.skill_name, should_activate=False).compute_scores(input)
        for output in result.outputs:
            if output.name == "activated":
                output.name = "overactivated"
        return result


class SkillRoutingMetric(MetricBase):
    """Distinguish target, acceptable alternate, and wrong skill invocations."""

    type: Literal[MetricType.SKILL_ROUTING] = MetricType.SKILL_ROUTING
    expected_skill: str = Field(min_length=1, description="Canonical component identity, e.g. Skill:search.")
    acceptable_skills: list[str] = Field(default_factory=list)

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("target_calls", "alternate_calls", "wrong_calls", "unknown_component_calls"), ("as_expected",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Routing requires normalized skill events.")
        skills = [call.component for call in trace.calls if call.component and call.component.startswith("Skill:")]
        target = skills.count(self.expected_skill)
        alternate = sum(skill != self.expected_skill and skill in self.acceptable_skills for skill in skills)
        wrong = len(skills) - target - alternate
        unknown = sum(call.component is None for call in trace.calls)
        values: dict[str, int | float] = {
            "target_calls": target,
            "alternate_calls": alternate,
            "wrong_calls": wrong,
            "unknown_component_calls": unknown,
        }
        if trace.complete and not unknown:
            values["as_expected"] = float(bool(target or alternate) and wrong == 0)
        return _result(trace, values)


class ExpectedScriptInvocationMetric(MetricBase):
    """Measure invocation independently of whether the invoked script succeeded."""

    type: Literal[MetricType.EXPECTED_SCRIPT_INVOCATION] = MetricType.EXPECTED_SCRIPT_INVOCATION
    expected_script: str = Field(min_length=1, description="Package-relative script path, matched exactly.")

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("invocations", "successful", "failed", "unknown_outcome"), ("invoked",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Script invocation requires correlated tool evidence.")
        calls = [call for call in trace.calls if call.script == self.expected_script]
        values: dict[str, int | float] = {
            "invocations": len(calls),
            "successful": sum(call.succeeded is True for call in calls),
            "failed": sum(call.succeeded is False for call in calls),
            "unknown_outcome": sum(call.succeeded is None for call in calls),
        }
        if calls or trace.complete:
            values["invoked"] = float(bool(calls))
        return _result(trace, values)


class WorkflowComplianceMetric(MetricBase):
    """Check authored precedence constraints without imposing order within a parallel step."""

    type: Literal[MetricType.WORKFLOW_COMPLIANCE] = MetricType.WORKFLOW_COMPLIANCE
    rules: list[SkillOrderRule] = Field(min_length=1)

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("rules", "satisfied"), ("score",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Workflow checks require ordered tool evidence.")
        passed = sum(_ordered(trace, rule) for rule in self.rules)
        values: dict[str, int | float] = {"rules": len(self.rules), "satisfied": passed}
        if trace.complete:
            values["score"] = passed / len(self.rules)
        return _result(trace, values)


class ErrorRecoveryMetric(MetricBase):
    """Attribute failures explicitly and follow correlated retry chains to known success."""

    type: Literal[MetricType.ERROR_RECOVERY] = MetricType.ERROR_RECOVERY

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(
            (
                "failures",
                "retried_failures",
                "recovered",
                "unresolved",
                "unknown_outcome",
                "skill_faults",
                "agent_faults",
                "unknown_faults",
            ),
            ("recovery_rate",),
        )

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Recovery requires correlated outcomes and retries.")
        failures = [call for call in trace.calls if call.succeeded is False]
        retried = recovered = 0
        for failure in failures:
            descendants = {failure.id}
            attempts = []
            for call in trace.calls:
                if call.retry_of in descendants:
                    descendants.add(call.id)
                    attempts.append(call)
            retried += bool(attempts)
            recovered += any(call.succeeded is True for call in attempts)
        unknown = sum(call.succeeded is None for call in trace.calls)
        values: dict[str, int | float] = {
            "failures": len(failures),
            "retried_failures": retried,
            "recovered": recovered,
            "unknown_outcome": unknown,
            **{
                f"{owner}_faults": sum(call.fault_owner == owner for call in failures)
                for owner in ("skill", "agent", "unknown")
            },
        }
        if trace.complete and unknown == 0:
            values["unresolved"] = len(failures) - recovered
            if failures:
                values["recovery_rate"] = recovered / len(failures)
        return _result(trace, values)


class ToolProductivityMetric(MetricBase):
    """Aggregate explicit producer/judge labels rather than guessing waste from repeated calls."""

    type: Literal[MetricType.TOOL_PRODUCTIVITY] = MetricType.TOOL_PRODUCTIVITY

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("productive", "wasted", "neutral", "unclassified"), ("score",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Productivity requires classified tool evidence.")
        values: dict[str, int | float] = {
            label: sum(call.productivity == label for call in trace.calls)
            for label in ("productive", "wasted", "neutral")
        }
        values["unclassified"] = sum(call.productivity is None for call in trace.calls)
        denominator = values["productive"] + values["wasted"]
        if trace.complete and not values["unclassified"] and denominator:
            values["score"] = values["productive"] / denominator
        return _result(trace, values)


class ToolSelectionMetric(MetricBase):
    """Set precision/recall/F1 for expected component/tool patterns, with optional alternatives and decoys."""

    type: Literal[MetricType.TOOL_SELECTION] = MetricType.TOOL_SELECTION
    expected_tools: list[str] = Field(min_length=1)
    acceptable_tools: list[str] = Field(default_factory=list)
    decoy_tools: list[str] = Field(default_factory=list)

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("distinct_called", "expected_matched", "unexpected", "decoys"), ("precision", "recall", "f1"))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Tool selection requires normalized tool identities.")
        called = {call.component or call.tool for call in trace.calls}
        decoys = {tool for tool in called if any(fnmatchcase(tool, pattern) for pattern in self.decoy_tools)}
        eligible = called - decoys
        accepted = {
            tool
            for tool in eligible
            if any(fnmatchcase(tool, pattern) for pattern in self.expected_tools + self.acceptable_tools)
        }
        matched = sum(any(fnmatchcase(tool, pattern) for tool in eligible) for pattern in self.expected_tools)
        values: dict[str, int | float] = {
            "distinct_called": len(called),
            "expected_matched": matched,
            "unexpected": len(called - accepted),
            "decoys": len(decoys),
        }
        if trace.complete:
            precision = len(accepted) / len(called) if called else 0.0
            recall = matched / len(self.expected_tools)
            values.update(
                precision=precision,
                recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            )
        return _result(trace, values)


class SkillArgumentRule(BaseModel):
    """One tool pattern and a self-contained JSON Schema for its arguments."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    tool: str = Field(min_length=1)
    schema_: dict[str, JsonValue] = Field(validation_alias="schema", serialization_alias="schema")

    @model_validator(mode="after")
    def valid_local_schema(self) -> Self:
        try:
            validators.validator_for(self.schema_).check_schema(self.schema_)
        except SchemaError as error:
            raise ValueError("invalid argument JSON Schema") from error
        pending: list[object] = [self.schema_]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "$id":
                        raise ValueError("argument schemas must not change the reference base with $id")
                    if (
                        key in {"$ref", "$dynamicRef", "$recursiveRef"}
                        and isinstance(value, str)
                        and not value.startswith("#")
                    ):
                        raise ValueError("argument schemas must use local references only")
                    pending.append(value)
            elif isinstance(item, list):
                pending.extend(item)
        return self


class ToolArgumentComplianceMetric(MetricBase):
    """Check all matching call/rule pairs; an uncalled tool has no argument-compliance score."""

    type: Literal[MetricType.TOOL_ARGUMENT_COMPLIANCE] = MetricType.TOOL_ARGUMENT_COMPLIANCE
    rules: list[SkillArgumentRule] = Field(min_length=1)

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("checked", "passed", "failed", "unmatched_rules"), ("score",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Argument checks require normalized tool inputs.")
        checked = passed = unmatched = 0
        for rule in self.rules:
            calls = [call for call in trace.calls if _matches(call, rule.tool)]
            unmatched += not calls
            validator = validators.validator_for(rule.schema_)(rule.schema_)
            checked += len(calls)
            passed += sum(validator.is_valid(call.arguments) for call in calls)
        values: dict[str, int | float] = {
            "checked": checked,
            "passed": passed,
            "failed": checked - passed,
            "unmatched_rules": unmatched,
        }
        if checked and trace.complete:
            values["score"] = passed / checked
        return _result(trace, values)


class ToolExecutionOutcomeMetric(MetricBase):
    """Known-outcome success rate; unknown outcomes remain visible and outside its denominator."""

    type: Literal[MetricType.TOOL_EXECUTION_OUTCOME] = MetricType.TOOL_EXECUTION_OUTCOME
    tool_pattern: str = Field(default="MCP:*", min_length=1)

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(("successful", "failed", "unknown"), ("success_rate",))

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Outcome scoring requires correlated tool results.")
        calls = [call for call in trace.calls if _matches(call, self.tool_pattern)]
        successful = sum(call.succeeded is True for call in calls)
        failed = sum(call.succeeded is False for call in calls)
        values: dict[str, int | float] = {
            "successful": successful,
            "failed": failed,
            "unknown": sum(call.succeeded is None for call in calls),
        }
        if successful + failed:
            values["success_rate"] = successful / (successful + failed)
        return _result(trace, values)


class SkillHandoffRule(BaseModel):
    """Require an earlier producer's value or artifact in a later consumer's input."""

    model_config = ConfigDict(extra="forbid")
    producer: str = Field(min_length=1)
    consumer: str = Field(min_length=1)
    kind: Literal["value", "artifact"] = "value"
    source_path: str = Field(
        default="", description="Dotted path in the producer observation; empty selects the whole value."
    )
    target_path: str = Field(default="", description="Dotted path in consumer arguments; empty selects all arguments.")
    minimum_value_characters: int = Field(default=8, ge=1, strict=True)


def _contains(value: object, context: object) -> bool:
    if value == context:
        return True
    if isinstance(value, str) and isinstance(context, str):
        return value in context
    if isinstance(context, dict):
        return any(_contains(value, item) for item in context.values())
    if isinstance(context, list):
        return any(_contains(value, item) for item in context)
    return False


def _handoff(trace: SkillToolTrace, rule: SkillHandoffRule, instruction: object) -> bool | None:
    if rule.kind == "value" and trace.initial_context is None:
        return None
    uncertain = False
    for producer in trace.calls:
        if not _matches(producer, rule.producer) or producer.succeeded is False:
            continue
        for consumer in trace.calls:
            if consumer.step <= producer.step or not _matches(consumer, rule.consumer):
                continue
            if producer.succeeded is None:
                uncertain = True
                continue
            if rule.kind == "artifact":
                if set(producer.writes) & set(consumer.reads):
                    return True
                continue
            observed = producer.observation
            if observed is None:
                uncertain = True
                continue
            value = (
                mapping_at_path(observed, rule.source_path)
                if rule.source_path and isinstance(observed, dict)
                else observed
                if not rule.source_path
                else None
            )
            if not isinstance(value, str) or len(value.strip()) < rule.minimum_value_characters:
                continue
            if _contains(value, trace.initial_context) or _contains(value, instruction):
                continue
            target = mapping_at_path(consumer.arguments, rule.target_path) if rule.target_path else consumer.arguments
            if _contains(value, target):
                return True
    return None if uncertain else False


class SkillConflictRule(BaseModel):
    """Two component patterns that must not both be activated in one trial."""

    model_config = ConfigDict(extra="forbid")
    first: str = Field(min_length=1)
    second: str = Field(min_length=1)


class ComponentCoordinationMetric(MetricBase):
    """Report authored order, handoff, and conflict probes separately and as an equal-weight score."""

    type: Literal[MetricType.COMPONENT_COORDINATION] = MetricType.COMPONENT_COORDINATION
    order_rules: list[SkillOrderRule] = Field(default_factory=list)
    handoffs: list[SkillHandoffRule] = Field(default_factory=list)
    conflicts: list[SkillConflictRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if not (self.order_rules or self.handoffs or self.conflicts):
            raise ValueError("provide at least one coordination probe")
        return self

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(
            (
                "order_checks",
                "order_passed",
                "handoff_checks",
                "handoffs_observed",
                "handoffs_unmeasured",
                "conflict_checks",
                "conflicts_observed",
            ),
            ("score",),
        )

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Coordination requires component and ordered handoff evidence.")
        ordered = sum(_ordered(trace, rule) for rule in self.order_rules)
        verdicts = [
            _handoff(trace, rule, input.row.data.get("inputs", {}).get("instruction")) for rule in self.handoffs
        ]
        handoffs = sum(verdict is True for verdict in verdicts)
        unmeasured = sum(verdict is None for verdict in verdicts)
        conflicts = sum(
            any(_matches(call, rule.first) for call in trace.calls)
            and any(_matches(call, rule.second) for call in trace.calls)
            for rule in self.conflicts
        )
        values: dict[str, int | float] = {
            "order_checks": len(self.order_rules),
            "order_passed": ordered,
            "handoff_checks": len(self.handoffs),
            "handoffs_observed": handoffs,
            "handoffs_unmeasured": unmeasured,
            "conflict_checks": len(self.conflicts),
            "conflicts_observed": conflicts,
        }
        if trace.complete and not unmeasured:
            values["score"] = (ordered + handoffs + len(self.conflicts) - conflicts) / (
                len(self.order_rules) + len(self.handoffs) + len(self.conflicts)
            )
        return _result(trace, values)


class ComponentUsageMetric(MetricBase):
    """Coverage of declared available components; environment omissions stay visible."""

    type: Literal[MetricType.COMPONENT_USAGE] = MetricType.COMPONENT_USAGE
    declared_components: list[str] = Field(min_length=1)
    unavailable_components: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_inventory(self) -> Self:
        if len(set(self.declared_components)) != len(self.declared_components):
            raise ValueError("declared components must be unique")
        if not set(self.unavailable_components) <= set(self.declared_components):
            raise ValueError("unavailable components must be declared")
        return self

    def output_spec(self) -> list[MetricOutputSpec]:
        return _spec(
            (
                "declared",
                "available",
                "unavailable",
                "exercised",
                "never_activated",
                "unknown_component_calls",
                "undeclared_exercised",
            ),
            ("coverage", "multi_component"),
        )

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trace = trial_trace(input)
        if trace is None:
            return _unmeasured("missing_trace", "Component coverage requires an inventory and observed activations.")
        available = set(self.declared_components) - set(self.unavailable_components)
        observed = {call.component for call in trace.calls if call.component is not None}
        if observed & set(self.unavailable_components):
            raise ValueError("trace activated a component marked unavailable")
        exercised = observed & available
        unknown = sum(call.component is None for call in trace.calls)
        values: dict[str, int | float] = {
            "declared": len(self.declared_components),
            "available": len(available),
            "unavailable": len(set(self.unavailable_components)),
            "exercised": len(exercised),
            "unknown_component_calls": unknown,
            "undeclared_exercised": len(observed - set(self.declared_components)),
        }
        if len(exercised) >= 2:
            values["multi_component"] = 1.0
        if trace.complete and not unknown:
            values["never_activated"] = len(available - exercised)
            values["multi_component"] = float(len(exercised) >= 2)
            if available:
                values["coverage"] = len(exercised) / len(available)
        return _result(trace, values)
