# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the notif plugin (the ``notif:`` section of the platform config)."""

from __future__ import annotations

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from nemo_notif_plugin.types import Severity
from pydantic import BaseModel, Field

DEFAULT_SEVERITIES: list[Severity] = ["warning", "error"]


class EmailConfig(BaseModel):
    """Email delivery of published events. Off by default, and a dry run until told otherwise."""

    enabled: bool = Field(default=False, description="Turn published events into emails.")
    dry_run: bool = Field(
        default=True,
        description="Build and log each email without opening an SMTP connection.",
    )
    smtp_host: str = Field(default="localhost", description="SMTP relay host. No authentication is attempted.")
    smtp_port: int = Field(default=25)
    starttls: bool = Field(default=False, description="Upgrade the SMTP connection with STARTTLS before sending.")
    timeout_seconds: float = Field(default=10.0)
    sender: str = Field(default="nemo-notif@localhost", description="From address.")
    recipients: list[str] = Field(default_factory=list, description="Addresses that receive every matching event.")
    allowed_recipient_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Domains a publisher may address through an event's `data.email_to`. "
            "Empty means publisher-supplied recipients are ignored."
        ),
    )
    severities: list[Severity] = Field(
        default=DEFAULT_SEVERITIES,
        description="Only events with one of these severities are emailed.",
    )
    type_prefixes: list[str] = Field(
        default_factory=list,
        description="Only events whose type starts with one of these are emailed. Empty means every type.",
    )
    studio_base_url: str = Field(
        default="",
        description="Absolute URL Studio is served at, e.g. https://host/studio. Used to link back to the event.",
    )


class NotifConfig(NemoConfig):
    plugin_name: ClassVar[str] = "notif"
    plugin_description: ClassVar[str] = "Notif plugin configuration"

    email: EmailConfig = Field(default_factory=EmailConfig)
