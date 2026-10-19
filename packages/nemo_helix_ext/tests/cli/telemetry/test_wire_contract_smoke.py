# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-safe wire-contract smoke test for telemetry events.

This module verifies that each telemetry event serializes to the wire contract
without making network calls. Each test constructs a valid event, wraps it in
QueuedEvent, calls build_payload, and asserts JSON serialization and envelope
structure.

Live UAT validation (POST against the telemetry endpoint) is a manual runbook:
set NEMO_TELEMETRY_ENDPOINT to the UAT URL, NEMO_TELEMETRY_ENABLED=true, and
run one real command; live validation is pending nemoSource schema registration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from nemo_helix_ext.cli.telemetry.events import (
    CommandInvokedEvent,
    JobRunEvent,
    OnboardingStepEvent,
    TaskStatusEnum,
)
from nemo_helix_ext.cli.telemetry.handler import QueuedEvent, build_payload

EXPECTED_ENVELOPE_KEYS = {
    "browserType",
    "clientId",
    "clientType",
    "clientVariant",
    "clientVer",
    "cpuArchitecture",
    "deviceGdprBehOptIn",
    "deviceGdprFuncOptIn",
    "deviceGdprTechOptIn",
    "deviceId",
    "deviceMake",
    "deviceModel",
    "deviceOS",
    "deviceOSVersion",
    "deviceType",
    "eventProtocol",
    "eventSchemaVer",
    "eventSysVer",
    "externalUserId",
    "gdprBehOptIn",
    "gdprFuncOptIn",
    "gdprTechOptIn",
    "idpId",
    "integrationId",
    "productName",
    "productVersion",
    "sentTs",
    "sessionId",
    "userId",
    "events",
}


class TestWireContractSmoke:
    """Verify each event type serializes to the wire contract envelope."""

    def test_onboarding_step_event_contract(self):
        """OnboardingStepEvent must serialize with camelCase aliases and correct envelope."""
        event = OnboardingStepEvent(
            task_status=TaskStatusEnum.COMPLETED,
            step="provider_discovery",
            provider_type="huggingface",
            models_discovered_bucket="10-100",
            skills_target="audit_content",
            agent_deployed=True,
        )
        queued = QueuedEvent(event=event, timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        payload = build_payload([queued], source_client_version="1.0.0", session_id="test")

        # Envelope keys must match exactly.
        assert set(payload.keys()) == EXPECTED_ENVELOPE_KEYS, (
            f"Envelope keys mismatch: {set(payload.keys()) ^ EXPECTED_ENVELOPE_KEYS}"
        )

        # Schema version must be 1.10.
        assert payload["eventSchemaVer"] == "1.10"

        # Must be JSON serializable.
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)

        # Event parameters must include camelCase aliases.
        event_entry = payload["events"][0]
        params = event_entry["parameters"]
        assert params["nemoSource"] == "platform"
        assert params["taskStatus"] == "completed"
        assert params["deploymentType"] == "cli"
        assert params["isCi"] is False
        assert params["step"] == "provider_discovery"
        assert params["providerType"] == "huggingface"
        assert params["modelsDiscoveredBucket"] == "10-100"
        assert params["skillsTarget"] == "audit_content"
        assert params["agentDeployed"] is True
        assert event_entry["name"] == "onboarding_step"

    def test_command_invoked_event_contract(self):
        """CommandInvokedEvent must serialize with camelCase aliases and correct envelope."""
        event = CommandInvokedEvent(
            task_status=TaskStatusEnum.COMPLETED,
            command="nemo helix apply",
            duration_sec=2.5,
            agent_mode=True,
        )
        queued = QueuedEvent(event=event, timestamp=datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc))
        payload = build_payload([queued], source_client_version="1.2.3", session_id="cmd-test")

        # Envelope keys must match exactly.
        assert set(payload.keys()) == EXPECTED_ENVELOPE_KEYS, (
            f"Envelope keys mismatch: {set(payload.keys()) ^ EXPECTED_ENVELOPE_KEYS}"
        )

        # Schema version must be 1.10.
        assert payload["eventSchemaVer"] == "1.10"

        # Must be JSON serializable.
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)

        # Event parameters must include camelCase aliases.
        event_entry = payload["events"][0]
        params = event_entry["parameters"]
        assert params["nemoSource"] == "platform"
        assert params["taskStatus"] == "completed"
        assert params["deploymentType"] == "cli"
        assert params["isCi"] is False
        assert params["command"] == "nemo helix apply"
        assert params["durationSec"] == 2.5
        assert params["agentMode"] is True
        assert event_entry["name"] == "command_invoked"

    def test_job_run_event_contract(self):
        """JobRunEvent must serialize with camelCase aliases and correct envelope."""
        event = JobRunEvent(
            task_status=TaskStatusEnum.COMPLETED,
            job_type="evaluate",
            duration_sec=15.75,
            plugins=["garak", "llm_judge"],
            model="defined",
            input_tokens=2048,
            output_tokens=512,
        )
        queued = QueuedEvent(event=event, timestamp=datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc))
        payload = build_payload([queued], source_client_version="2.0.0", session_id="job-test")

        # Envelope keys must match exactly.
        assert set(payload.keys()) == EXPECTED_ENVELOPE_KEYS, (
            f"Envelope keys mismatch: {set(payload.keys()) ^ EXPECTED_ENVELOPE_KEYS}"
        )

        # Schema version must be 1.10.
        assert payload["eventSchemaVer"] == "1.10"

        # Must be JSON serializable.
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)

        # Event parameters must include camelCase aliases.
        event_entry = payload["events"][0]
        params = event_entry["parameters"]
        assert params["nemoSource"] == "platform"
        assert params["taskStatus"] == "completed"
        assert params["deploymentType"] == "cli"
        assert params["isCi"] is False
        assert params["jobType"] == "evaluate"
        assert params["durationSec"] == 15.75
        assert params["plugins"] == ["garak", "llm_judge"]
        assert params["model"] == "defined"
        assert params["inputTokens"] == 2048
        assert params["outputTokens"] == 512
        assert event_entry["name"] == "job_run"
