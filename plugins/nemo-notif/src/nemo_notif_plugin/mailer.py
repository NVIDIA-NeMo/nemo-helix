# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Email delivery of published notif events.

Prototype scope: one plain-text email per matching event, sent to a configured list plus any
publisher-supplied addresses in an allowed domain. There is no per-user preference, batching or
retry; a failed send is logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from collections.abc import Callable
from email.message import EmailMessage

from nemo_notif_plugin.config import EmailConfig
from nemo_notif_plugin.types import NotifEvent

logger = logging.getLogger(__name__)

# Key in an event's `data` a publisher sets to address specific people, e.g. a job's creator.
EMAIL_TO_KEY = "email_to"

Send = Callable[[EmailConfig, EmailMessage], None]


def recipients_for(event: NotifEvent, config: EmailConfig) -> list[str]:
    """Configured recipients, then the event's own in an allowed domain, without duplicates."""
    requested = event.data.get(EMAIL_TO_KEY)
    allowed = {domain.lower() for domain in config.allowed_recipient_domains}
    from_event = [
        address
        for address in (requested if isinstance(requested, list) else [])
        if isinstance(address, str) and address.rpartition("@")[2].lower() in allowed
    ]
    return list(dict.fromkeys([*config.recipients, *from_event]))


def build_message(event: NotifEvent, config: EmailConfig) -> EmailMessage | None:
    """The email for an event, or None when the event should not be emailed."""
    if not config.enabled or event.severity not in config.severities:
        return None
    if config.type_prefixes and not event.type.startswith(tuple(config.type_prefixes)):
        return None
    recipients = recipients_for(event, config)
    if not recipients:
        return None

    lines = [event.title, ""]
    if event.message:
        lines += [event.message, ""]
    lines += [f"Workspace: {event.workspace}", f"Event: {event.type}"]
    if event.link and config.studio_base_url:
        lines += ["", f"Open in Studio: {config.studio_base_url.rstrip('/')}{event.link}"]

    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = ", ".join(recipients)
    # Header values may not contain line breaks.
    message["Subject"] = f"[NeMo {event.workspace}] {' '.join(event.title.split())}"
    message.set_content("\n".join(lines) + "\n")
    return message


def send_smtp(config: EmailConfig, message: EmailMessage) -> None:
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=config.timeout_seconds) as smtp:
        if config.starttls:
            smtp.starttls()
        smtp.send_message(message)


class Mailer:
    """Emails events in the background so publishing never waits on SMTP."""

    def __init__(self, config: EmailConfig, send: Send = send_smtp) -> None:
        self._config = config
        self._send = send
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, event: NotifEvent) -> bool:
        """Queue the event's email. Returns whether one was queued."""
        message = build_message(event, self._config)
        if message is None:
            return False
        task = asyncio.get_running_loop().create_task(self._deliver(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self) -> None:
        """Wait for queued emails to finish."""
        await asyncio.gather(*self._tasks)

    async def _deliver(self, message: EmailMessage) -> None:
        if self._config.dry_run:
            logger.info(
                "Email dry run: would send %r to %s\n%s", message["Subject"], message["To"], message.get_content()
            )
            return
        try:
            await asyncio.to_thread(self._send, self._config, message)
        except (OSError, smtplib.SMTPException) as exc:
            logger.warning("Failed to email %r to %s: %s", message["Subject"], message["To"], exc)
            return
        logger.info("Emailed %r to %s", message["Subject"], message["To"])
