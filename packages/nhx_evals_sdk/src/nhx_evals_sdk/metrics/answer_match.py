# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer matching metric runtime implementation.

The scoring functions generalize the ones ``finance-qa`` ported from the Nemotron finance LoRA cookbook: each
constant the cookbook fixed (the normalization, the number format, the separators, the token pattern, the
delimiter) is a parameter of the metric's configuration.
"""

import fnmatch
import json
import re
import unicodedata
from ast import parse
from collections import Counter
from collections.abc import Mapping
from typing import Any

from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.metrics.template_rendering import (
    build_template_context,
    render_default_output_text_candidate_or_raise,
    render_template_or_raise,
    template_metric_repr,
)
from nhx_evals_sdk.values.metrics import (
    AnswerEquals,
    AnswerExtract,
    AnswerMatch,
    AnswerNormalize,
    AnswerNumber,
    AnswerParses,
    AnswerSelect,
    AnswerSpanSet,
    AnswerTokenF1,
)

__all__ = ["AnswerMatchMetric", "SelectUnmatchedError", "answer_match_output_spec", "score_answer"]

_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


class SelectUnmatchedError(ValueError):
    """A select with ``unmatched: error`` found no case for the row's label value."""

    def __init__(self, select: str, label: str, value: str, cases: list[str]) -> None:
        self.select, self.label, self.value, self.cases = select, label, value, cases
        super().__init__(f"select {select!r}: the label {label}={value!r} matches no case ({', '.join(cases)})")


def normalize(value: object, spec: AnswerNormalize) -> str:
    """Apply the normalization steps in order: Unicode form, case folding, whitespace, end characters."""
    text = str(value)
    if spec.unicode:
        text = unicodedata.normalize(spec.unicode, text)
    if spec.casefold:
        text = text.casefold()
    if spec.collapse_whitespace:
        text = re.sub(r"\s+", " ", text.strip())
    if spec.strip:
        text = text.strip(spec.strip)
    return text


def parse_number(value: object, comparison: AnswerNumber, spec: AnswerNormalize) -> tuple[float | None, str]:
    """Parse a whole normalized text as a number with an optional scale word: ``(number, scale name)``."""
    text = normalize(value, spec)
    for char in comparison.ignore:
        text = text.replace(char, "")
    words = "|".join(re.escape(word) for word in sorted(comparison.scales, key=len, reverse=True))
    match = re.fullmatch(_NUMBER + (rf"\s*({words})?" if words else ""), text)
    if not match:
        return None, ""
    scale = comparison.scales.get(match.group(2) or "", "") if words else ""
    return float(match.group(1)), scale


def number_match(answer: str, reference: str, comparison: AnswerNumber, spec: AnswerNormalize) -> bool:
    """Numbers within the tolerance, with the scale rule; otherwise ``equals`` or false."""
    (left, left_scale), (right, right_scale) = (
        parse_number(answer, comparison, spec),
        parse_number(reference, comparison, spec),
    )
    if left is not None and right is not None:
        if comparison.require_reference_scale and right_scale and left_scale != right_scale:
            return False
        tolerance = max(comparison.absolute_tolerance, abs(right) * comparison.relative_tolerance)
        return abs(left - right) <= tolerance
    if comparison.otherwise == "equals":
        return normalize(answer, spec) == normalize(reference, spec)
    return False


def spans(value: object, separators: list[str], spec: AnswerNormalize) -> list[str]:
    """Split the text on the separators first, then normalize each span; sorted, without empty spans."""
    pattern = r"\s*(?:" + "|".join(re.escape(separator) for separator in separators) + r")\s*"
    return sorted(span for span in (normalize(part, spec) for part in re.split(pattern, str(value))) if span)


def token_f1(answer: str, reference: str, pattern: str, spec: AnswerNormalize) -> float:
    """F1 between the token counts of the normalized texts. Two empty texts score 1."""
    predicted = Counter(re.findall(pattern, normalize(answer, spec)))
    expected = Counter(re.findall(pattern, normalize(reference, spec)))
    if not predicted and not expected:
        return 1.0
    overlap = sum((predicted & expected).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def parses(text: str, language: str) -> bool:
    """Whether the text parses. Python is parsed, never run; input beyond the parser's limits doesn't parse."""
    try:
        if language == "python":
            parse(text)
        else:
            json.loads(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def first_message(response: object) -> Mapping[str, Any] | None:
    """The first choice's message of a chat completion response, or None."""
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, Mapping) else None


def raw_completion(message: Mapping[str, Any], open_tag: str, close_tag: str) -> str:
    """Rebuild what the model wrote before a reasoning parser split it: ``{open}\\n{reasoning}\\n{close}\\n{content}``.

    The reasoning is ``reasoning_content``, or ``reasoning`` when that is empty. Without reasoning, it is the
    content alone. Tags already around the reasoning (left by a run config's ``reasoning.end_token``) are dropped,
    so the trace is wrapped once.
    """
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning.strip().endswith(close_tag):
        reasoning = reasoning.strip()[: -len(close_tag)].removeprefix(open_tag)
    raw = f"{open_tag}\n{reasoning}\n{close_tag}\n{content}" if reasoning else str(content)
    return raw.strip()


def extract_answer(text: str, extract: AnswerExtract | None) -> tuple[str, str, bool]:
    """Return ``(answer, trace, found)``. Without a delimiter or match, the answer is the whole text."""
    text = text.strip()
    if extract is None:
        return text, "", False
    last = extract.occurrence == "last"
    if extract.after is not None:
        if extract.after not in text:
            return text, "", False
        trace, answer = text.rsplit(extract.after, 1) if last else text.split(extract.after, 1)
    else:
        matches = list(re.finditer(str(extract.regex), text))
        if not matches:
            return text, "", False
        match = matches[-1] if last else matches[0]
        answer, trace = match.group(extract.group) or "", text[: match.start()]
    if extract.trace_starts_after and extract.trace_starts_after in trace:
        trace = trace.split(extract.trace_starts_after, 1)[1]
    return answer.strip(), trace.strip(), True


def _as_text(value: object, template_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    raise TypeError(f"The {template_name} must be a string or a number.")


def _case(select: AnswerSelect, value: str) -> str | None:
    for pattern, comparison in select.cases.items():
        if value == pattern or fnmatch.fnmatchcase(value, pattern):
            return comparison
    return select.default


def _is_continuous(config: AnswerMatch, names: list[str]) -> bool:
    return any(isinstance(config.comparisons[name], AnswerTokenF1) for name in names)


def _select_targets(select: AnswerSelect) -> list[str]:
    return [*select.cases.values(), *([select.default] if select.default else [])]


def answer_match_output_spec(config: AnswerMatch) -> list[MetricOutputSpec]:
    """The outputs, in order: reported comparisons, ``extract.found_as``, selects, label outputs."""
    specs: list[MetricOutputSpec] = []
    for name, comparison in config.comparisons.items():
        if comparison.report:
            continuous = isinstance(comparison, AnswerTokenF1)
            specs.append(MetricOutputSpec.continuous_score(name) if continuous else MetricOutputSpec.boolean(name))
    if config.extract is not None and config.extract.found_as:
        specs.append(MetricOutputSpec.boolean(config.extract.found_as))
    for name, select in config.select.items():
        required = select.default is not None or select.unmatched == "error"
        if _is_continuous(config, _select_targets(select)):
            specs.append(MetricOutputSpec.continuous_score(name, required=required))
        else:
            specs.append(MetricOutputSpec.boolean(name, required=required))
    specs.extend(MetricOutputSpec.label(name) for name in config.label_outputs)
    return specs


def _render(name: str, template: str, input: MetricInput, context: dict[str, Any], metric_repr: str) -> str:
    value = render_template_or_raise(
        template_name=name,
        template=template,
        context=context,
        item=input.row.data,
        sample=input.candidate,
        metric_repr=metric_repr,
    )
    return _as_text(value, name)


def _candidate_text(
    config: AnswerMatch, input: MetricInput, context: dict[str, Any], metric_repr: str, metric_name: str
) -> str:
    if config.candidate:
        return _render("candidate", config.candidate, input, context, metric_repr)
    extract = config.extract
    if extract is not None and extract.source == "raw":
        message = first_message(input.candidate.response)
        if message is not None:
            return raw_completion(message, extract.trace_starts_after or "", str(extract.after))
    output_text = render_default_output_text_candidate_or_raise(sample=input.candidate, metric_name=metric_name)
    return _as_text(output_text, "candidate")


def score_answer(config: AnswerMatch, input: MetricInput, *, metric_repr: str, metric_name: str) -> list[MetricOutput]:
    """Score one row with an answer-match configuration.

    The order follows the configuration's dependencies, and fails early: the reference and the label outputs
    are rendered, a select with ``unmatched: error`` is checked, and only then is the candidate read.
    """
    context = build_template_context(input.row.data, input.candidate)
    reference = _render("reference", config.reference, input, context, metric_repr)
    labels = {
        name: _render(name, template, input, context, metric_repr) for name, template in config.label_outputs.items()
    }
    chosen: dict[str, str | None] = {}
    for name, select in config.select.items():
        chosen[name] = _case(select, labels[select.on])
        if chosen[name] is None and select.unmatched == "error":
            raise SelectUnmatchedError(name, select.on, labels[select.on], list(select.cases))

    text = _candidate_text(config, input, context, metric_repr, metric_name)
    answer, trace, found = extract_answer(text, config.extract)

    spec = config.normalize
    values: dict[str, bool | float] = {}
    for name, comparison in config.comparisons.items():
        if isinstance(comparison, AnswerEquals):
            values[name] = normalize(answer, spec) == normalize(reference, spec)
        elif isinstance(comparison, AnswerNumber):
            values[name] = number_match(answer, reference, comparison, spec)
        elif isinstance(comparison, AnswerSpanSet):
            values[name] = spans(answer, comparison.separators, spec) == spans(reference, comparison.separators, spec)
        elif isinstance(comparison, AnswerTokenF1):
            values[name] = token_f1(answer, reference, comparison.pattern, spec)
        elif isinstance(comparison, AnswerParses):
            values[name] = parses(trace if comparison.input == "trace" else answer, comparison.language)

    outputs = [
        MetricOutput(name=name, value=values[name])
        for name, comparison in config.comparisons.items()
        if comparison.report
    ]
    if config.extract is not None and config.extract.found_as:
        outputs.append(MetricOutput(name=config.extract.found_as, value=found))
    for name, select in config.select.items():
        target = chosen[name]
        if target is None:
            continue
        value = values[target]
        continuous = _is_continuous(config, _select_targets(select))
        outputs.append(MetricOutput(name=name, value=float(value) if continuous else value))
    outputs.extend(MetricOutput(name=name, value=value) for name, value in labels.items())
    return outputs


class AnswerMatchMetric(AnswerMatch):
    """Extract the answer from the candidate, normalize it, and compare it with the reference.

    Scores per row:

    - each comparison with ``report`` set, under its name: ``equals``, ``number`` and ``span-set`` and ``parses``
      are booleans, and ``token-f1`` is a score from 0 to 1;
    - ``extract.found_as``: whether the delimiter or a match was found;
    - each ``select``: the comparison its label's value picks for the row. It is optional unless it has a default
      or ``unmatched: error``;
    - each of ``label_outputs``, as a label for grouping rows.
    """

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return outputs emitted by this metric."""
        return answer_match_output_spec(self)

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Compute the scores for the metric."""
        outputs = score_answer(self, input, metric_repr=template_metric_repr(self), metric_name=self.__class__.__name__)
        return MetricResult(outputs=outputs)
