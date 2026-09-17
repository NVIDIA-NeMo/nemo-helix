# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Harbor verifier-reward vocabulary shared by the Harbor trial adapter, the Harbor runtime,
and :class:`~nemo_evaluator_sdk.metrics.runner_rewards.HarborRewardMetric`.

Owns three things those modules would otherwise each restate: which reward *names* are usable,
what a rejected reward *value* is called, and how a parsed reward mapping travels on trial
metadata. Kept dependency-light -- stdlib plus
:func:`~nemo_evaluator_sdk.metrics.utils.as_finite_float`, which the ``agent_eval`` import path
already pays for -- so both the light ``agent_eval`` path and the metric stack can depend on it
without pulling each other in.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from nemo_evaluator_sdk.metrics.utils import as_finite_float

RewardKeyRejection = Literal["invalid_key", "reserved_key"]

#: Why a Harbor reward *value* was dropped: ``boolean`` (True/False, not read as 1/0),
#: ``non_numeric`` (unparseable), or ``non_finite`` (NaN/Inf/overflow).
HarborRewardValueRejection = Literal["boolean", "non_numeric", "non_finite"]

#: Trial-metadata keys carrying one parsed Harbor reward mapping. Named here rather than spelled
#: independently at each writer and reader so the durable vocabulary stays centralized.
REWARD_DETAILS_KEY = "reward_details"
REWARD_REJECTIONS_KEY = "reward_rejections"
REWARD_ENTRY_REJECTIONS_KEY = "reward_entry_rejections"

# ``<output>.pass@<k>`` is how the SDK names pass@k aggregates of a metric output; a reward name
# carrying that suffix would collide with them in the summary.
_RESERVED_REWARD_KEY_RE = re.compile(r"\.pass@\d+$")


def reward_key_rejection(key: object) -> RewardKeyRejection | None:
    """Classify unusable or reserved verifier reward names; ``None`` means the name is safe."""
    if (
        type(key) is not str
        or not key
        or len(key) > 255
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
    ):
        return "invalid_key"
    if _RESERVED_REWARD_KEY_RE.search(key):
        return "reserved_key"
    return None


def validate_reward_key(key: str) -> None:
    """Require a reward name to be safe and non-reserved.

    Callers' tests match the message on ``"reward_key"``; keep it stable.
    """
    rejection = reward_key_rejection(key)
    if rejection is not None:
        raise ValueError(f"invalid Harbor reward_key: {rejection}")


def finite_reward(value: object) -> float | None:
    """The value as a reward, or ``None`` when it is not one.

    Stricter than :func:`~nemo_evaluator_sdk.metrics.utils.as_finite_float` in one way, on
    purpose: ``type(...) in`` rather than ``isinstance``, so an ``int``/``float`` *subclass*
    handed over by a verifier is not read as a measurement.
    """
    return as_finite_float(value) if type(value) in (int, float) else None


@dataclass(frozen=True)
class ParsedHarborRewards:
    """One trial's Harbor verifier rewards, classified.

    ``values`` are finite floats by construction. Everything the verifier emitted that could not
    become one is accounted for rather than dropped silently: ``rejected_by_key`` names the reward
    and why its value was unusable, and ``rejected_entries`` counts entries whose *name* was
    unusable -- those have no key to report under, which is the whole reason they were rejected.
    """

    values: dict[str, float] = field(default_factory=dict)
    rejected_by_key: dict[str, HarborRewardValueRejection] = field(default_factory=dict)
    rejected_entries: tuple[RewardKeyRejection, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        """This mapping as the trial-metadata fragment its readers expect."""
        return {
            REWARD_DETAILS_KEY: dict(self.values),
            REWARD_REJECTIONS_KEY: dict(self.rejected_by_key),
            REWARD_ENTRY_REJECTIONS_KEY: list(self.rejected_entries),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> ParsedHarborRewards:
        """Read back what :meth:`to_metadata` wrote, defensively.

        Trial metadata is a free-form dict that may have been hand-built or imported, so nothing
        here trusts its shape: anything that is not a usable reward or a known rejection reason is
        left out. This is the one place that parse happens, so readers work in floats and literals
        rather than re-deriving both.
        """
        details = metadata.get(REWARD_DETAILS_KEY)
        rejections = metadata.get(REWARD_REJECTIONS_KEY)
        entries = metadata.get(REWARD_ENTRY_REJECTIONS_KEY)
        values: dict[str, float] = {}
        if isinstance(details, Mapping):
            values = {
                key: number
                for key, value in details.items()
                if isinstance(key, str) and (number := finite_reward(value)) is not None
            }
        rejected_by_key: dict[str, HarborRewardValueRejection] = {}
        if isinstance(rejections, Mapping):
            value_reasons = get_args(HarborRewardValueRejection)
            rejected_by_key = {
                key: reason for key, reason in rejections.items() if isinstance(key, str) and reason in value_reasons
            }
        rejected_entries: tuple[RewardKeyRejection, ...] = ()
        if isinstance(entries, (list, tuple)):
            key_reasons = get_args(RewardKeyRejection)
            rejected_entries = tuple(entry for entry in entries if entry in key_reasons)
        return cls(values=values, rejected_by_key=rejected_by_key, rejected_entries=rejected_entries)
