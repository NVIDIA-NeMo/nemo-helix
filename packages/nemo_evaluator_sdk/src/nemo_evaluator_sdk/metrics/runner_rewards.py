# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metrics that surface a runner's own reward.

These live here rather than beside their runners because a built-in metric subclasses
``MetricBase``, which drags the dataset-schema stack (jinja2, jsonschema) with it. The Harbor
runtime is on the optimizer's light import path — see
``test_agent_eval_import_does_not_pull_the_execution_stack`` — so defining them there would make
every consumer of that module pay for machinery these metrics do not use. The runner modules
re-export them lazily, so ``from ...harbor_runtime import HarborRewardMetric`` still works.
"""

from collections.abc import Mapping
from typing import Literal

from nemo_evaluator_sdk.agent_eval.reward_keys import (
    REWARD_DETAILS_KEY,
    HarborRewardValueRejection,
    ParsedHarborRewards,
    finite_reward,
    validate_reward_key,
)
from nemo_evaluator_sdk.enums import MetricType
from nemo_evaluator_sdk.metrics.protocol import (
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nemo_evaluator_sdk.values.metrics import MetricBase
from nemo_evaluator_sdk.values.protocol import OUTPUT_DETAIL, REASON_DETAIL, ContinuousScore
from pydantic import Field, field_validator

__all__ = ["GymRewardMetric", "HarborRewardMetric"]


class GymRewardMetric(MetricBase):
    """Score the Gym verifier reward stamped onto trial metadata.

    The Gym analogue of :class:`HarborRewardMetric`: reads the per-trial ``reward`` off the
    candidate metadata (populated by ``GymAgentTaskRunner``); a trial with no reward is left
    **unscored** (``None`` → ``nan``), excluded from the mean and surfaced as ``nan_count`` rather
    than counted as a spurious ``0.0``. Gym owns the scoring — this metric only surfaces it.
    """

    type: Literal[MetricType.GYM_REWARD] = MetricType.GYM_REWARD
    output_name: str = Field(
        default="reward", description="Name of the emitted score, read from the trial's `reward` metadata."
    )

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score(self.output_name)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        reward = input.candidate.metadata.get("reward")
        value = float(reward) if reward is not None else None
        return MetricResult(outputs=[MetricOutput(name=self.output_name, value=value)])


class HarborRewardMetric(MetricBase):
    """Convert one task's Harbor verifier reward mapping into SDK scores.

    The primary reward is always emitted, preserving Harbor's accepted zero
    fallback. Finalized secondary rewards are optional and are omitted when the
    verifier did not provide a usable finite number.
    """

    type: Literal[MetricType.HARBOR_REWARD] = MetricType.HARBOR_REWARD
    output_name: str = Field(
        default="reward", description="Name of the emitted score, read from the trial's `reward` metadata."
    )
    reward_keys: tuple[str, ...] = Field(
        default=(), description="Finalized task-local Harbor reward keys, including the primary output."
    )

    @field_validator("output_name", "reward_keys")
    @classmethod
    def _names_are_safe(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        # Every name here becomes a declared output. A reserved ``<name>.pass@k`` or malformed name
        # would collide with the SDK's aggregate names; fail at construction, not after the run.
        for name in (value,) if isinstance(value, str) else value:
            validate_reward_key(name)
        return value

    def _ordered_keys(self) -> tuple[str, ...]:
        secondary = sorted(set(self.reward_keys) - {self.output_name})
        return (self.output_name, *secondary)

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec(
                name=key,
                value_schema=ContinuousScore,
                required=(key == self.output_name),
            )
            for key in self._ordered_keys()
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        metadata = input.candidate.metadata
        # One defensive parse of the envelope the adapter wrote: `values` are finite floats and the
        # reasons are known literals, whatever a hand-built metadata dict happened to contain.
        rewards = ParsedHarborRewards.from_metadata(metadata)
        # The raw mapping, kept only to tell a reward that was never emitted from one that was and
        # is unusable -- a distinction the parse deliberately collapses.
        raw_details = metadata.get(REWARD_DETAILS_KEY)
        raw_details = raw_details if isinstance(raw_details, Mapping) else {}

        outputs: list[MetricOutput] = []
        diagnostics: list[MetricDiagnostic] = []
        for key in self._ordered_keys():
            is_primary = key == self.output_name
            # The primary is read from `reward`, not `reward_details`: `output_name` is the
            # runner's `reward_key`, which a metric built independently need not share.
            raw = metadata.get("reward") if is_primary else raw_details.get(key)
            value = finite_reward(raw) if is_primary else rewards.values.get(key)
            if value is not None:
                outputs.append(MetricOutput(name=key, value=value))
                continue
            diagnostics.append(_missing_reward_diagnostic(key, raw, rewards.rejected_by_key))
            if is_primary:
                # Required output: an unusable primary still scores, preserving Harbor's zero.
                outputs.append(MetricOutput(name=key, value=0.0))

        diagnostics.extend(
            MetricDiagnostic(
                message="Harbor reward entry was rejected",
                details={OUTPUT_DETAIL: None, REASON_DETAIL: reason},
            )
            for reason in rewards.rejected_entries
        )
        return MetricResult(outputs=outputs, diagnostics=diagnostics)


def _missing_reward_diagnostic(
    key: str,
    raw: object,
    rejected_by_key: Mapping[str, HarborRewardValueRejection],
) -> MetricDiagnostic:
    """Say why one declared reward produced no score.

    The adapter's own classification when it recorded one; otherwise ``absent`` (nothing was
    emitted under this name) or ``unusable`` (something was, but it is not a finite number) --
    the only reading left for metadata a caller built by hand.
    """
    reason: str = rejected_by_key.get(key) or ("absent" if raw is None else "unusable")
    return MetricDiagnostic(
        message=f"Harbor reward {key!r} was not measured",
        details={OUTPUT_DETAIL: key, REASON_DETAIL: reason},
    )
