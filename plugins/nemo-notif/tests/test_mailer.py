# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for emailing notif events. Nothing here talks to a real mail server."""

from __future__ import annotations

import logging
import socketserver
import threading
from collections.abc import Iterator
from email.message import EmailMessage

import httpx
import pytest
from fake_entities import FakeEntitiesClient
from fastapi import FastAPI
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_notif_plugin.broker import NotifBroker
from nemo_notif_plugin.config import EmailConfig
from nemo_notif_plugin.mailer import Mailer, build_message, recipients_for, send_smtp
from nemo_notif_plugin.service import build_events_router
from nemo_notif_plugin.types import NotifEvent


def _config(**overrides: object) -> EmailConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "dry_run": False,
        "sender": "nemo@example.com",
        "recipients": ["ops@example.com"],
        "studio_base_url": "https://studio.example.com/studio/",
    }
    return EmailConfig.model_validate({**settings, **overrides})


def _event(**overrides: object) -> NotifEvent:
    fields: dict[str, object] = {
        "id": "evt-1",
        "workspace": "team",
        "type": "openshell.policy.draft.pending",
        "title": "Network access requested: example.com:443",
        "message": "python3.13 in job train was blocked from example.com:443.",
        "severity": "warning",
        "link": "/workspaces/team/jobs/train",
        "data": {},
    }
    return NotifEvent.model_validate({**fields, **overrides})


def test_message_carries_the_event_and_a_studio_link() -> None:
    message = build_message(_event(), _config())

    assert message is not None
    assert message["From"] == "nemo@example.com"
    assert message["To"] == "ops@example.com"
    assert message["Subject"] == "[NeMo team] Network access requested: example.com:443"
    assert message.get_content() == (
        "Network access requested: example.com:443\n"
        "\n"
        "python3.13 in job train was blocked from example.com:443.\n"
        "\n"
        "Workspace: team\n"
        "Event: openshell.policy.draft.pending\n"
        "\n"
        "Open in Studio: https://studio.example.com/studio/workspaces/team/jobs/train\n"
    )


def test_subject_drops_line_breaks_from_the_title() -> None:
    message = build_message(_event(title="Blocked\r\nBcc: someone@example.com"), _config())

    assert message is not None
    assert message["Subject"] == "[NeMo team] Blocked Bcc: someone@example.com"
    assert message["Bcc"] is None


@pytest.mark.parametrize(
    ("event", "config"),
    [
        (_event(), _config(enabled=False)),
        (_event(severity="success"), _config()),
        (_event(type="jobs.job.failed"), _config(type_prefixes=["openshell."])),
        (_event(), _config(recipients=[])),
    ],
    ids=["disabled", "severity", "type", "no-recipients"],
)
def test_events_that_are_not_emailed(event: NotifEvent, config: EmailConfig) -> None:
    assert build_message(event, config) is None


def test_publisher_recipients_need_an_allowed_domain() -> None:
    event = _event(data={"email_to": ["creator@example.com", "ops@example.com", "x@elsewhere.test", 7]})

    assert recipients_for(event, _config()) == ["ops@example.com"]
    assert recipients_for(event, _config(allowed_recipient_domains=["Example.com"])) == [
        "ops@example.com",
        "creator@example.com",
    ]
    assert recipients_for(_event(data={"email_to": "creator@example.com"}), _config(recipients=[])) == []


async def test_mailer_sends_matching_events_and_skips_the_rest() -> None:
    sent: list[EmailMessage] = []
    mailer = Mailer(_config(), send=lambda _config, message: sent.append(message))

    assert mailer.submit(_event()) is True
    assert mailer.submit(_event(severity="info")) is False
    await mailer.drain()

    assert [message["Subject"] for message in sent] == ["[NeMo team] Network access requested: example.com:443"]


async def test_dry_run_logs_the_email_and_sends_nothing(caplog: pytest.LogCaptureFixture) -> None:
    sent: list[EmailMessage] = []
    mailer = Mailer(_config(dry_run=True), send=lambda _config, message: sent.append(message))

    with caplog.at_level(logging.INFO, logger="nemo_notif_plugin.mailer"):
        assert mailer.submit(_event()) is True
        await mailer.drain()

    assert sent == []
    assert "Email dry run: would send" in caplog.text
    assert "Open in Studio: https://studio.example.com/studio/workspaces/team/jobs/train" in caplog.text


async def test_failed_send_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    def refuse(_config: EmailConfig, _message: EmailMessage) -> None:
        raise ConnectionRefusedError("relay down")

    mailer = Mailer(_config(), send=refuse)

    with caplog.at_level(logging.WARNING, logger="nemo_notif_plugin.mailer"):
        mailer.submit(_event())
        await mailer.drain()

    assert "relay down" in caplog.text


async def test_publish_route_emails_the_stored_event() -> None:
    sent: list[EmailMessage] = []
    mailer = Mailer(_config(), send=lambda _config, message: sent.append(message))
    app = FastAPI()
    app.include_router(build_events_router(NotifBroker(), mailer), prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: FakeEntitiesClient()
    app.dependency_overrides[get_effective_principal_id] = lambda: "alice"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v2/workspaces/team/events",
            json={"type": "jobs.job.failed", "title": "Fine-tune failed", "severity": "error"},
        )
        await mailer.drain()

    assert response.status_code == 201
    assert [message["Subject"] for message in sent] == ["[NeMo team] Fine-tune failed"]


class _FakeSmtpHandler(socketserver.StreamRequestHandler):
    """Just enough SMTP to accept one message and record it on the server."""

    def handle(self) -> None:
        self.wfile.write(b"220 fake ready\r\n")
        in_data = False
        body: list[str] = []
        for raw in self.rfile:
            line = raw.decode().rstrip("\r\n")
            if in_data:
                if line == ".":
                    in_data = False
                    self.server.messages.append("\n".join(body))  # ty: ignore[unresolved-attribute]
                    self.wfile.write(b"250 queued\r\n")
                else:
                    body.append(line)
                continue
            verb = line.split(" ", 1)[0].split(":", 1)[0].upper()
            if verb == "EHLO":
                self.wfile.write(b"250 fake\r\n")
            elif verb == "DATA":
                in_data = True
                self.wfile.write(b"354 go ahead\r\n")
            elif verb == "QUIT":
                self.wfile.write(b"221 bye\r\n")
                return
            else:
                self.server.envelope.append(line)  # ty: ignore[unresolved-attribute]
                self.wfile.write(b"250 ok\r\n")


@pytest.fixture
def fake_smtp() -> Iterator[socketserver.TCPServer]:
    server = socketserver.TCPServer(("127.0.0.1", 0), _FakeSmtpHandler)
    server.messages = []  # ty: ignore[unresolved-attribute]
    server.envelope = []  # ty: ignore[unresolved-attribute]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_send_smtp_delivers_to_the_configured_relay(fake_smtp: socketserver.TCPServer) -> None:
    host, port = fake_smtp.server_address[:2]
    config = _config(smtp_host=host, smtp_port=port, recipients=["ops@example.com", "sec@example.com"])
    message = build_message(_event(), config)
    assert message is not None

    send_smtp(config, message)

    assert [line.lower() for line in fake_smtp.envelope] == [  # ty: ignore[unresolved-attribute]
        "mail from:<nemo@example.com>",
        "rcpt to:<ops@example.com>",
        "rcpt to:<sec@example.com>",
    ]
    [delivered] = fake_smtp.messages  # ty: ignore[unresolved-attribute]
    assert "Subject: [NeMo team] Network access requested: example.com:443" in delivered
    assert "Open in Studio: https://studio.example.com/studio/workspaces/team/jobs/train" in delivered
