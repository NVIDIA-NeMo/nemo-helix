# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent entity definitions and API schema models.

Entity tests live here alongside schema tests because both cover pure Pydantic
model behaviour — no network, no entity store required.

Entity classes: ``Agent``, ``AgentDeployment``, ``AgentSession``
                                               → ``nemo_agents_plugin.entities``
Request schemas: ``CreateAgentRequest``, ``CreateDeploymentRequest``
                                               → ``nemo_agents_plugin.schema``
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from nemo_agents_plugin.entities import (
    NAT_WORKFLOW_CONFIG_FORMAT,
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentInline,
    AgentEnvironmentSpec,
    AgentSandboxSpec,
    AgentSession,
    ComputeResources,
    ComputeSpecInline,
    EnvironmentSpecInline,
    McpFulfillment,
    SandboxSpecInline,
    SessionStatus,
    agent_config_file_ref,
    ethos_file_ref,
    ethos_fileset_name,
    ethos_local_path,
)
from nemo_agents_plugin.schema import (
    CreateAgentRequest,
    CreateComputeSpecRequest,
    CreateDeploymentRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentSpecRequest,
)
from nemo_platform_plugin.auth import AuthContext
from pydantic import ValidationError

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Entity: Agent
# ---------------------------------------------------------------------------


class TestAgentEntity:
    def test_entity_type(self) -> None:
        assert Agent.__entity_type__ == "agent"

    def test_defaults(self) -> None:
        a = Agent(name="calc", workspace="default")
        assert a.name == "calc"
        assert a.workspace == "default"
        assert a.description == ""
        assert a.config == {}
        assert a.config_format == NAT_WORKFLOW_CONFIG_FORMAT

    def test_config_stored(self) -> None:
        config = {"llms": {"my_llm": {"_type": "nim", "model_name": "llama"}}}
        a = Agent(name="calc", workspace="default", config=config)
        assert a.config["llms"]["my_llm"]["_type"] == "nim"

    def test_data_fields_include_domain_fields(self) -> None:
        a = Agent(
            name="calc",
            workspace="default",
            description="A calculator",
            config={"key": "value"},
            config_format=NAT_WORKFLOW_CONFIG_FORMAT,
        )
        data = a._get_data_fields()
        assert "description" in data
        assert "config" in data
        assert "config_format" in data

    def test_data_fields_exclude_base_fields(self) -> None:
        a = Agent(name="calc", workspace="default")
        data = a._get_data_fields()
        assert "name" not in data
        assert "workspace" not in data

    def test_description_optional(self) -> None:
        a = Agent(name="x", workspace="w", description="hello")
        assert a.description == "hello"

    def test_id_and_created_at_accessible_after_persistence(self) -> None:
        """Entity computed fields include id and created_at for API serialisation."""
        a = Agent(name="calc", workspace="default")
        a._id = "agent-id-123"
        a._created_at = NOW
        assert a.id == "agent-id-123"
        assert a.created_at == NOW

    def test_entity_serialises_with_computed_fields(self) -> None:
        """model_dump() includes id, created_at — these appear in API responses."""
        a = Agent(name="calc", workspace="default", config={"k": "v"})
        a._id = "abc"
        a._created_at = NOW
        data = a.model_dump()
        assert data["id"] == "abc"
        assert data["name"] == "calc"
        assert data["config"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Entity: AgentDeployment
# ---------------------------------------------------------------------------


class TestAgentDeploymentEntity:
    def test_entity_type(self) -> None:
        assert AgentDeployment.__entity_type__ == "agent_deployment"

    def test_defaults(self) -> None:
        d = AgentDeployment(name="dep", workspace="default")
        assert d.agent == ""
        assert d.status == "pending"
        assert d.endpoint == ""
        assert d.port == 0
        assert d.pid == 0
        assert d.error == ""

    def test_status_transitions(self) -> None:
        d = AgentDeployment(name="dep", workspace="default", agent="calc", status="pending")
        d.status = "starting"
        assert d.status == "starting"
        d.status = "running"
        assert d.status == "running"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentDeployment.model_validate({"name": "dep", "workspace": "default", "status": "unknown"})

    def test_data_fields_include_deployment_fields(self) -> None:
        d = AgentDeployment(
            name="dep",
            workspace="default",
            agent="calc",
            status="running",
            endpoint="http://localhost:9001",
            port=9001,
            pid=12345,
        )
        data = d._get_data_fields()
        assert "agent" in data
        assert "status" in data
        assert "endpoint" in data
        assert "port" in data
        assert "pid" in data

    def test_data_fields_exclude_base_fields(self) -> None:
        d = AgentDeployment(name="dep", workspace="default")
        data = d._get_data_fields()
        assert "name" not in data
        assert "workspace" not in data

    def test_entity_serialises_as_api_response(self) -> None:
        """model_dump() produces the full API response shape including base fields."""
        d = AgentDeployment(
            name="dep",
            workspace="default",
            agent="calc",
            status="running",
        )
        d._id = "dep-id"
        d._created_at = NOW
        data = d.model_dump()
        assert data["id"] == "dep-id"
        assert data["name"] == "dep"
        assert data["agent"] == "calc"
        assert data["status"] == "running"

    def test_data_fields_include_private_auth_context(self) -> None:
        auth_context = AuthContext(
            principal_id="user:alice",
            principal_email="alice@example.com",
            principal_groups=["research"],
        )
        d = AgentDeployment(name="dep", workspace="default", agent="calc").with_auth_context(auth_context)

        data = d._get_data_fields()

        assert d.auth_context == auth_context
        assert data["_auth_context"]["principal_id"] == "user:alice"
        assert data["_auth_context"]["principal_email"] == "alice@example.com"

    def test_auth_context_schema_is_nullable(self) -> None:
        schema = AgentDeployment.model_json_schema(mode="serialization")

        assert schema["properties"]["auth_context"]["nullable"] is True


# ---------------------------------------------------------------------------
# Entity: AgentSession
# ---------------------------------------------------------------------------


class TestAgentSessionEntity:
    def test_entity_type(self) -> None:
        assert AgentSession.__entity_type__ == "agent_session"

    def test_deployment_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentSession.model_validate({"name": "session", "workspace": "default"})

    def test_deployment_id_must_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            AgentSession(name="session", workspace="default", deployment_id="")

    def test_deployment_id_is_stored_as_domain_data(self) -> None:
        session = AgentSession(name="session", workspace="default", deployment_id="deployment-id")

        assert session.deployment_id == "deployment-id"
        assert session._get_data_fields()["deployment_id"] == "deployment-id"

    def test_lifecycle_defaults(self) -> None:
        session = AgentSession(name="session", workspace="default", deployment_id="deployment-id")

        assert session.status is SessionStatus.ACTIVE
        assert session.first_active_at is None
        assert session.last_active_at is None
        assert session.expires_at is None

    @pytest.mark.parametrize("status", list(SessionStatus))
    def test_lifecycle_status(self, status: SessionStatus) -> None:
        session = AgentSession(
            name="session",
            workspace="default",
            deployment_id="deployment-id",
            status=status,
        )

        assert session.status is status

    @pytest.mark.parametrize(
        ("current_status", "new_status"),
        [
            (SessionStatus.ACTIVE, SessionStatus.EXPIRED),
            (SessionStatus.ACTIVE, SessionStatus.LOST),
            (SessionStatus.ACTIVE, SessionStatus.CLOSED),
            (SessionStatus.EXPIRED, SessionStatus.CLOSED),
            (SessionStatus.LOST, SessionStatus.CLOSED),
        ],
    )
    def test_allowed_status_transitions(
        self,
        current_status: SessionStatus,
        new_status: SessionStatus,
    ) -> None:
        assert current_status.can_transition_to(new_status)

    @pytest.mark.parametrize("status", list(SessionStatus))
    def test_same_status_transition_is_idempotent(self, status: SessionStatus) -> None:
        assert status.can_transition_to(status)

    @pytest.mark.parametrize(
        ("current_status", "new_status"),
        [
            (SessionStatus.EXPIRED, SessionStatus.ACTIVE),
            (SessionStatus.EXPIRED, SessionStatus.LOST),
            (SessionStatus.LOST, SessionStatus.ACTIVE),
            (SessionStatus.LOST, SessionStatus.EXPIRED),
            (SessionStatus.CLOSED, SessionStatus.ACTIVE),
            (SessionStatus.CLOSED, SessionStatus.EXPIRED),
            (SessionStatus.CLOSED, SessionStatus.LOST),
        ],
    )
    def test_disallowed_status_transitions(
        self,
        current_status: SessionStatus,
        new_status: SessionStatus,
    ) -> None:
        assert not current_status.can_transition_to(new_status)

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentSession.model_validate(
                {
                    "name": "session",
                    "workspace": "default",
                    "deployment_id": "deployment-id",
                    "status": "failed",
                }
            )

    def test_lifecycle_timestamps_round_trip(self) -> None:
        session = AgentSession(
            name="session",
            workspace="default",
            deployment_id="deployment-id",
            first_active_at=NOW,
            last_active_at=NOW,
            expires_at=NOW,
        )

        assert session.first_active_at == NOW
        assert session.last_active_at == NOW
        assert session.expires_at == NOW

    def test_entity_serialises_as_api_response(self) -> None:
        session = AgentSession(name="session", workspace="default", deployment_id="deployment-id")
        session._id = "session-id"
        session._created_at = NOW

        data = session.model_dump()
        assert data["id"] == "session-id"
        assert data["name"] == "session"
        assert data["workspace"] == "default"
        assert data["deployment_id"] == "deployment-id"


# ---------------------------------------------------------------------------
# API schema: CreateAgentRequest
# ---------------------------------------------------------------------------


class TestCreateAgentRequest:
    def test_required_fields(self) -> None:
        req = CreateAgentRequest(name="calc", config={"llms": {}})
        assert req.name == "calc"
        assert req.config == {"llms": {}}
        assert req.description == ""
        assert req.config_format == NAT_WORKFLOW_CONFIG_FORMAT

    def test_missing_config_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreateAgentRequest.model_validate({"name": "calc"})

    def test_custom_format(self) -> None:
        req = CreateAgentRequest(name="x", config={}, config_format="custom-v2")
        assert req.config_format == "custom-v2"


# ---------------------------------------------------------------------------
# Canonical Ethos-location helpers
# ---------------------------------------------------------------------------


class TestEthosLocationConvention:
    def test_ethos_location_convention(self) -> None:
        assert ethos_fileset_name("checkout-bot") == "checkout-bot-ethos"
        ref = ethos_file_ref("default", "checkout-bot")
        assert str(ref) == "default/checkout-bot-ethos#ETHOS.md"
        assert ethos_local_path("checkout-bot").as_posix() == "agents/checkout-bot-ethos/ETHOS.md"

    def test_config_file_ref_uses_canonical_agent_yaml(self) -> None:
        ref = agent_config_file_ref("default", "checkout-bot")
        assert str(ref) == "default/checkout-bot-ethos#agent.yaml"


# ---------------------------------------------------------------------------
# API schema: CreateDeploymentRequest
# ---------------------------------------------------------------------------


class TestCreateDeploymentRequest:
    def test_required_agent(self) -> None:
        req = CreateDeploymentRequest(agent="calc")
        assert req.agent == "calc"
        assert req.name is None

    def test_optional_name(self) -> None:
        req = CreateDeploymentRequest(agent="calc", name="calc-abc1")
        assert req.name == "calc-abc1"

    def test_environment_defaults_none(self) -> None:
        req = CreateDeploymentRequest(agent="calc")
        assert req.environment is None

    def test_environment_ref_string(self) -> None:
        req = CreateDeploymentRequest(agent="calc", environment="default/env1")
        assert req.environment == "default/env1"

    def test_environment_inline(self) -> None:
        req = CreateDeploymentRequest(
            agent="calc",
            environment=AgentEnvironmentInline.model_validate(
                {"compute_spec": {"resources": {"limits": {"cpu": "2"}}}}
            ),
        )
        assert isinstance(req.environment, AgentEnvironmentInline)
        assert isinstance(req.environment.compute_spec, ComputeSpecInline)


# ---------------------------------------------------------------------------
# Entities: AgentEnvironment / AgentEnvironmentSpec / AgentComputeSpec
# ---------------------------------------------------------------------------


class TestEnvironmentEntities:
    def test_entity_types(self) -> None:
        assert AgentEnvironment.__entity_type__ == "agent_environment"
        assert AgentEnvironmentSpec.__entity_type__ == "agent_environment_spec"
        assert AgentComputeSpec.__entity_type__ == "agent_compute_spec"
        assert AgentSandboxSpec.__entity_type__ == "agent_sandbox_spec"

    def test_compute_spec_resources(self) -> None:
        cs = AgentComputeSpec(
            name="c1",
            workspace="default",
            resources=ComputeResources.model_validate(
                {"limits": {"cpu": "2", "nvidia.com/gpu": "1"}, "requests": {"cpu": "1"}}
            ),
        )
        assert cs.resources.limits == {"cpu": "2", "nvidia.com/gpu": "1"}
        assert cs.resources.requests == {"cpu": "1"}

    def test_environment_spec_fields(self) -> None:
        es = AgentEnvironmentSpec(
            name="e1",
            workspace="default",
            env={"FOO": "bar"},
            secrets={"TOKEN": "default/token"},
            mcp={"search": McpFulfillment(url="http://x", secrets={"KEY": "default/key"})},
        )
        assert es.env == {"FOO": "bar"}
        assert es.secrets == {"TOKEN": "default/token"}
        assert es.mcp["search"].url == "http://x"
        assert es.mcp["search"].secrets == {"KEY": "default/key"}
        assert es.provider == "local"

    def test_environment_ref_and_inline_unions(self) -> None:
        by_ref = AgentEnvironment(name="env1", workspace="default", environment_spec="default/e1")
        assert by_ref.environment_spec == "default/e1"
        assert by_ref.compute_spec is None
        assert by_ref.sandbox_spec is None

        inline = AgentEnvironment(
            name="env2",
            workspace="default",
            environment_spec=EnvironmentSpecInline.model_validate({"env": {"A": "1"}}),
            compute_spec=ComputeSpecInline.model_validate({"resources": {"limits": {"cpu": "1"}}}),
        )
        assert isinstance(inline.environment_spec, EnvironmentSpecInline)
        assert isinstance(inline.compute_spec, ComputeSpecInline)

    def test_sandbox_spec_fields(self) -> None:
        ss = AgentSandboxSpec(
            name="s1",
            workspace="default",
            provider="openshell",
            provider_config={"policy_ref": "default/my-policy"},
        )
        assert ss.provider == "openshell"
        assert ss.provider_config == {"policy_ref": "default/my-policy"}

    def test_environment_with_sandbox_spec(self) -> None:
        by_ref = AgentEnvironment(
            name="env3",
            workspace="default",
            sandbox_spec="default/s1",
        )
        assert by_ref.sandbox_spec == "default/s1"

        inline = AgentEnvironment(
            name="env4",
            workspace="default",
            sandbox_spec=SandboxSpecInline(provider="opensandbox", provider_config={"runtime_class": "kata"}),
        )
        assert isinstance(inline.sandbox_spec, SandboxSpecInline)
        assert inline.sandbox_spec.provider == "opensandbox"

        from_wire = AgentEnvironment.model_validate(
            {
                "name": "env5",
                "workspace": "default",
                "sandbox_spec": {"provider": "opensandbox", "provider_config": {"runtime_class": "kata"}},
            }
        )
        assert isinstance(from_wire.sandbox_spec, SandboxSpecInline)
        assert from_wire.sandbox_spec.provider_config == {"runtime_class": "kata"}

    def test_data_fields_include_domain_fields(self) -> None:
        es = AgentEnvironmentSpec(name="e1", workspace="default", env={"FOO": "bar"})
        data = es._get_data_fields()
        assert "env" in data
        assert "provider" in data
        assert "name" not in data


class TestAgentDeploymentEnvironmentSnapshot:
    def test_environment_and_compute_default_none(self) -> None:
        d = AgentDeployment(name="dep", workspace="default")
        assert d.environment is None
        assert d.compute is None

    def test_environment_ref_and_compute_snapshot(self) -> None:
        d = AgentDeployment(
            name="dep",
            workspace="default",
            agent="calc",
            environment="default/env1",
            compute=ComputeSpecInline(resources=ComputeResources.model_validate({"limits": {"cpu": "2"}})),
        )
        assert d.environment == "default/env1"
        assert isinstance(d.compute, ComputeSpecInline)
        assert d.compute.resources.limits == {"cpu": "2"}


class TestCreateEnvironmentRequests:
    def test_create_environment_request(self) -> None:
        req = CreateEnvironmentRequest(name="env1", environment_spec="default/e1")
        assert req.name == "env1"
        assert req.environment_spec == "default/e1"

    def test_create_environment_spec_request(self) -> None:
        req = CreateEnvironmentSpecRequest(name="e1", env={"FOO": "bar"})
        assert req.name == "e1"
        assert req.env == {"FOO": "bar"}

    def test_create_compute_spec_request(self) -> None:
        req = CreateComputeSpecRequest(
            name="c1",
            resources=ComputeResources.model_validate({"limits": {"cpu": "2"}}),
        )
        assert req.name == "c1"
        assert req.resources.limits == {"cpu": "2"}
