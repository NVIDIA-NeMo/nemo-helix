# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notif event entity, stored in the NeMo Helix entity store."""

from __future__ import annotations

from typing import Any

from nemo_helix_plugin.entity import NemoEntity
from nemo_notif_plugin.types import Severity


class NotifEventEntity(NemoEntity, entity_type="notif_event"):
    """A published event. ``name`` doubles as the public event id.

    Read state is per principal: ``read_by`` lists the principal ids that have read the event.
    """

    type: str
    title: str
    message: str | None = None
    severity: Severity = "info"
    link: str | None = None
    data: dict[str, Any] = {}
    read_by: list[str] = []
