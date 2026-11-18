<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Evals SDK

## Quickstart

```python
from nhx_evals_sdk import evaluate

result = evaluate(
    dataset="data/eval.jsonl",
    metric="exact_match",
    reference="{{ expected }}",
    candidate="{{ prediction }}",
)

result.print_summary()
rows_df = result.to_pandas()
aggregate_df = result.to_pandas(view="aggregate")
```

`evaluate(...)` is the recommended product-level API for scripts and notebooks. It accepts:
- inline rows
- `DatasetRows`
- `pyarrow.Table`
- a local file path
- a local directory path plus an optional `pattern`

The returned `OfflineEvaluationResult` supports:
- `print_summary()`
- `format_summary()`
- `to_records()`
- `to_table()`
- `to_pandas()`

Pandas conversion is optional. Install `nhx-evals-sdk[pandas]` to use `to_pandas()`.

Offline evaluation uses each dataset row as both `item` and `sample`. In practice,
that means templates should usually read directly from the row, for example
`{{item.answer}}` and `{{item.model_output}}`.

## Advanced Usage

If you want direct control over metric construction or async execution, the low-level APIs remain available:

```python
from nhx_evals_sdk import ExactMatchMetric, evaluate_offline_sync

metric = ExactMatchMetric.from_template(
    reference="{{item.expected}}",
    candidate="{{item.prediction}}",
)

result = evaluate_offline_sync(metric=metric, dataset=[{"expected": "4", "prediction": "4"}])
```

## Gym captured usage in OTLP

Captured Gym model calls preserve input, output, reported total, cache read,
cache write, and reasoning token counts when converted to OTLP traces for Intake.
Gym's normalized usage takes precedence over provider-reported usage; older
captures without normalized counters continue to use the provider's counts.
Explicit zeros are preserved, unknown counts remain absent, and missing totals
are not inferred.

When a rollout includes valid embedded model-call captures, those captures are
used instead of separately supplied calls. Empty or malformed embedded captures
leave separately supplied calls in effect.

## Docker Compose sandbox provider

`compose.py` is the stable public facade for the production Docker Compose
provider. Focused private modules separate contracts, lifecycle state, CLI and
subprocess behavior, inspection and readiness, file transfer and ownership,
lifecycle primitives, and provider orchestration.

## Docker Compose sandbox tests

The hermetic tests for `DockerComposeSandboxProvider` are organized by the
provider component they exercise: contracts, CLI execution, inspection,
lifecycle, and file transfer. Shared Compose fakes live in the agent-eval test
kit, while the vendored-import test remains a standalone compatibility check.

## Skill evaluation metrics (beta)

The SDK provides three artifact metrics and three trial-scoring capabilities
inspired by SkillEvaluator. Import them from `nhx_evals_sdk` or
`nhx_evals_sdk.metrics.skills`.

```python
from nhx_evals_sdk import (
    SkillStructureMetric,
    SkillReferenceIntegrityMetric,
    SkillContextEfficiencyMetric,
    SkillActivationMetric,
    TokenEfficiencyMetric,
    expected_behavior_metric,
)

artifact_metrics = [
    SkillStructureMetric(),
    SkillReferenceIntegrityMetric(),
    SkillContextEfficiencyMetric(token_budget=5000),
]
trial_metrics = [
    SkillActivationMetric(skill_name="my-skill", should_activate=True),
    TokenEfficiencyMetric(half_life=200_000),
]
# judge, views = expected_behavior_metric(judge_model, ["Reports the recovery action."])
```

Artifact metrics read `reference.skill_snapshot`, validated as
`nhx_evals_sdk.SkillSnapshot`. The experimental snapshot contains the original
`SKILL.md` text, the directory name, a complete package-relative file inventory,
and a package digest supplied by an artifact adapter. Reference integrity covers
links in the entry document. It checks file existence against the inventory,
including image paths and reference-style Markdown links.

Activation reads the target skill's boolean observation from
`trial.metadata.skill_activation`. An absent or null observation leaves scores
unmeasured. A false observation requires enough harness evidence to establish
non-activation. Token efficiency reads the canonical trial measurements and
requires an explicit cache-read count, including zero.

`expected_behavior_metric(model, assertions)` returns a configured
`LLMJudgeMetric` and the `expected_behavior` and `all_assertions_met` semantic
views. It evaluates the final answer against shared outcome assertions. All
assertions must have finite verdicts for either view to be measured.

The metric classes participate in the built-in metric union, so the Evals plugin
can package and hydrate their configurations inline. The expected-behavior
factory uses the existing inline judge configuration. The example
[`skill_eval/run_metric_preview.py`](examples/skill_eval/run_metric_preview.py)
supplies mock artifact, activation, and judge evidence for a v1/v2 comparison.
