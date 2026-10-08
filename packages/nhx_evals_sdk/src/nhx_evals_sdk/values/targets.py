# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The union of candidate producers accepted as an evaluation ``target``.

This union spans three sibling modules and belongs to none of them, so it lives here rather than
in one member. It cannot live in the ``values`` barrel either: that surface is lazy, exposing
``Model``/``Agent`` only through ``_LAZY_ATTRS``, while a ``|`` union needs the real classes at
import time.
"""

from __future__ import annotations

from typing import TypeAlias

from nhx_evals_sdk.values.agents import Agent
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.retrieval import Retrieval

__all__ = ["EvalTarget"]

EvalTarget: TypeAlias = Model | Agent | Retrieval | None
