# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity definitions owned by the customization router."""

from __future__ import annotations

from typing import Any

from nemo_platform_plugin.entity import NemoEntity


class CustomizationJobTemplate(NemoEntity, entity_type="customization_job_template"):
    """A named, re-runnable customization job input.

    ``config`` holds the same job input a submit would take, stored as given: the
    submitter-facing schema belongs to whichever backend plugin is installed, and the
    router does not import them. Submit is what checks it. ``backend`` is lifted out so
    callers know which customization backend/form should replay the saved config.
    """

    backend: str
    config: dict[str, Any]
    description: str = ""
