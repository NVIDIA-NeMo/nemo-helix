# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resource class for the Agents plugin.

Registered under the ``nemo.sdk`` entry-point group. The platform lazily
instantiates this plugin's sync or async SDK resource as ``client.agents``.

Usage (once the SDK hub is wired up)::

    from nemoplatform import NeMo

    nemo = NeMo(base_url="http://localhost:8000")

    # Agent CRUD
    agent = nemo.agents.create(name="calculator", config={...})
    agents = nemo.agents.list()
    agent = nemo.agents.get("calculator")
    nemo.agents.delete("calculator")

    # Deployment lifecycle
    dep = nemo.agents.deployments.create(agent="calculator")  # subprocess
    dep = nemo.agents.deployments.create(
        agent="calculator", deployment_mode="docker", image="calculator:local"
    )
    deps = nemo.agents.deployments.list()
    dep = nemo.agents.deployments.get("calculator-a1b2")
    nemo.agents.deployments.delete("calculator-a1b2")

    # Environments / specs (the request/fulfill split)
    spec = nemo.agents.environment_specs.create(
        name="ben",
        spec=EnvironmentSpecInline(
            env={"LOG_LEVEL": "debug"},
            secrets={"GITHUB_PERSONAL_ACCESS_TOKEN": "default/ben-pat"},
        ),
    )
    env = nemo.agents.environments.create(name="repo-research-ben", environment_spec="default/ben")
    cs = nemo.agents.compute_specs.create(
        name="big", spec=ComputeSpecInline(resources=ComputeResources(limits={"cpu": "2"})),
    )
    dep = nemo.agents.deployments.create(agent="calculator", environment="default/repo-research-ben")

    # Invocation (routes through the agents gateway)
    result = nemo.agents.invoke(agent="calculator", input="What is 2+2?")
    result = nemo.agents.invoke(
        deployment="calculator-a1b2",
        session_id="session-entity-id",
        input="Continue",
    )

    # agents.execute jobs
    job = nemo.agents.jobs.execute.create(spec={"agent": "calculator", "input": "What is 2+2?"})
    job = nemo.agents.jobs.execute.get(job["name"])
    results = nemo.agents.jobs.execute.list_results(job["name"])

An async namespace is mounted as ``client.agents`` on ``AsyncNeMoPlatform``.
It currently exposes ``jobs`` only — agent CRUD, deployments, and ``invoke``
remain sync-only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeVar

from nemo_agents_plugin.entities import (
    AgentEnvironmentInline,
    ComputeSpecInline,
    EnvironmentSpecInline,
)
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.agents.client import AgentsClient, AsyncAgentsClient
from nemo_platform_plugin.agents.types import (
    AgentJobRequest,
    CreateAgentRequest,
    CreateComputeSpecRequest,
    CreateDeploymentRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentSpecRequest,
    InvokeAgentRequest,
    JsonMap,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.response import NemoPaginatedResponse, NemoResponse
from nemo_platform_plugin.sdk import NemoPluginSDKResources
from pydantic import BaseModel, TypeAdapter

_DEFAULT_MODEL_PLACEHOLDER = re.compile(r"\$(?:\{NEMO_DEFAULT_MODEL\}|NEMO_DEFAULT_MODEL(?![A-Za-z0-9_]))")
_DEFAULT_WORKSPACE = "default"
_JSON_MAP_ADAPTER = TypeAdapter(JsonMap)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _json_map(value: object) -> JsonMap:
    return _JSON_MAP_ADAPTER.validate_python(value)


def _json_map_from_response(response: NemoResponse[_ModelT]) -> JsonMap:
    response.data()
    body: object = response.http_response.json()
    return _json_map(body)


def _json_map_from_page(response: NemoPaginatedResponse[_ModelT]) -> JsonMap:
    response.page()
    body: object = response.http_response.json()
    return _json_map(body)


def _spec_to_dict(spec: BaseModel | Mapping[str, object] | None) -> JsonMap:
    """Normalize a typed ``*Inline`` model or a loose mapping to a request body."""
    if spec is None:
        return {}
    if isinstance(spec, BaseModel):
        return _json_map(spec.model_dump(exclude_unset=True, mode="json"))
    return _json_map(dict(spec))


def _contains_default_model_placeholder(value: object) -> bool:
    """Return True when *value* still contains an unresolved default-model placeholder."""
    if isinstance(value, str):
        protected = value.replace("$$", "\0DOLLAR\0")
        return _DEFAULT_MODEL_PLACEHOLDER.search(protected) is not None
    if isinstance(value, dict):
        return any(_contains_default_model_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_default_model_placeholder(item) for item in value)
    return False


def _agents_client_from_platform(platform: NeMoPlatform) -> AgentsClient:
    client = client_from_platform(platform, AgentsClient)
    if client.workspace is None:
        return client.with_workspace(_DEFAULT_WORKSPACE)
    return client


def _async_agents_client_from_platform(platform: AsyncNeMoPlatform) -> AsyncAgentsClient:
    async_client = client_from_platform(platform, AsyncAgentsClient)
    if async_client.workspace is None:
        return async_client.with_workspace(_DEFAULT_WORKSPACE)
    return async_client


class AgentsResource:
    """SDK namespace for ``nemo.agents.*``."""

    def __init__(self, platform: NeMoPlatform) -> None:
        """
        Args:
            platform: The generated ``NeMoPlatform`` client. The Agents resource
                adapts it to the typed ``AgentsClient`` while sharing the same
                base URL, default workspace, auth headers, timeout, retry policy,
                and underlying HTTP transport.
        """
        self._platform = platform
        self._client = _agents_client_from_platform(platform)
        self._deployments: _DeploymentResource | None = None
        self._environments: _EnvironmentResource | None = None
        self._environment_specs: _EnvironmentSpecResource | None = None
        self._compute_specs: _ComputeSpecResource | None = None
        self._jobs: _JobsResource | None = None

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        name: str,
        config: Mapping[str, object],
        description: str = "",
        config_format: str = "nat-workflow-v1",
        workspace: str | None = None,
    ) -> JsonMap:
        """Create a new agent.

        Args:
            name: Unique agent name within the workspace.
            config: NAT workflow config dict.
            description: Optional human-readable description.
            config_format: Config format identifier (default: ``"nat-workflow-v1"``).
            workspace: Target workspace. Defaults to the workspace configured
                on the platform client, or ``"default"`` when the client has no
                workspace.

        Returns:
            The created agent as a dict.
        """
        from nemo_agents_plugin.utils import inject_default_model

        resolved_config = _json_map(inject_default_model(dict(config)))
        if _contains_default_model_placeholder(resolved_config):
            raise ValueError(
                "Agent config references ${NEMO_DEFAULT_MODEL}, but no default model is selected. "
                "Run `nemo setup`, set NEMO_DEFAULT_MODEL, or replace the placeholder with an explicit "
                "VirtualModel name."
            )
        response = self._client.create_agent(
            workspace=workspace,
            body=CreateAgentRequest(
                name=name,
                config=resolved_config,
                description=description,
                config_format=config_format,
            ),
        )
        return _json_map_from_response(response)

    def list(self, workspace: str | None = None) -> JsonMap:
        """List agents in *workspace*."""
        return _json_map_from_page(self._client.list_agents(workspace=workspace))

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get an agent by name."""
        return _json_map_from_response(self._client.get_agent(workspace=workspace, name=name))

    def delete(self, name: str, workspace: str | None = None) -> None:
        """Delete an agent by name."""
        self._client.delete_agent(workspace=workspace, name=name).data()

    # ------------------------------------------------------------------
    # Deployment sub-resource
    # ------------------------------------------------------------------

    @property
    def deployments(self) -> "_DeploymentResource":
        """Sub-resource for deployment lifecycle operations."""
        if self._deployments is None:
            self._deployments = _DeploymentResource(self._client)
        return self._deployments

    @property
    def environments(self) -> "_EnvironmentResource":
        """Sub-resource for AgentEnvironment CRUD (``nemo.agents.environments``)."""
        if self._environments is None:
            self._environments = _EnvironmentResource(self._client)
        return self._environments

    @property
    def environment_specs(self) -> "_EnvironmentSpecResource":
        """Sub-resource for AgentEnvironmentSpec CRUD (``nemo.agents.environment_specs``)."""
        if self._environment_specs is None:
            self._environment_specs = _EnvironmentSpecResource(self._client)
        return self._environment_specs

    @property
    def compute_specs(self) -> "_ComputeSpecResource":
        """Sub-resource for AgentComputeSpec CRUD (``nemo.agents.compute_specs``)."""
        if self._compute_specs is None:
            self._compute_specs = _ComputeSpecResource(self._client)
        return self._compute_specs

    @property
    def jobs(self) -> "_JobsResource":
        """Sub-resource for agents job collections."""
        if self._jobs is None:
            self._jobs = _JobsResource(self._client)
        return self._jobs

    # ------------------------------------------------------------------
    # Invocation and evaluation
    # ------------------------------------------------------------------

    def invoke(
        self,
        *,
        input: str,
        agent: str | None = None,
        deployment: str | None = None,
        session_id: str | None = None,
        workspace: str | None = None,
        timeout: int = 300,
    ) -> JsonMap:
        """Send a single request to an agent via the gateway.

        Args:
            input: The user message / query string.
            agent: Agent name (gateway resolves the active deployment).
            deployment: Deployment name (direct targeting).
            session_id: Optional persisted Platform session entity ID. Sent in
                the ``X-Nemo-Session-Id`` request header.
            workspace: Workspace.
            timeout: Request timeout in seconds.

        Returns:
            The agent's response as a dict.
        """
        if agent is not None and deployment is not None:
            raise ValueError("Provide either agent= or deployment=, not both.")
        if session_id == "":
            raise ValueError("session_id must not be empty.")

        body = InvokeAgentRequest(messages=[{"role": "user", "content": input}], stream=False)
        client = self._client.with_options(
            headers={SESSION_ID_HEADER: session_id} if session_id is not None else None,
            timeout=timeout,
        )
        if agent is not None:
            return client.invoke_agent(workspace=workspace, name=agent, body=body).data()
        if deployment is not None:
            return client.invoke_deployment(workspace=workspace, name=deployment, body=body).data()
        raise ValueError("Provide either agent= or deployment=.")

    def evaluate(
        self,
        *,
        eval_config: str,
        agent: str | None = None,
        endpoint: str | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Trigger an evaluation run.

        .. note::
            Platform-managed evaluation is not yet implemented.
            Use the CLI instead::

                nemo agents evaluate --eval-config <path>
                nemo agents evaluate --eval-config <path> --agent <name>

        Raises:
            NotImplementedError: Always — platform-managed evaluation is not
                yet available. Use ``nemo agents evaluate`` CLI instead.
        """
        raise NotImplementedError(
            "Platform-managed evaluation is not yet implemented. "
            "Use the CLI: nemo agents evaluate --eval-config <path> [--agent <name>]"
        )


class _DeploymentResource:
    """Deployment lifecycle operations under ``nemo.agents.deployments``."""

    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        agent: str,
        name: str | None = None,
        deployment_mode: str = "subprocess",
        image: str | None = None,
        use_image_entrypoint: bool = False,
        environment: str | Mapping[str, object] | AgentEnvironmentInline | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Create a deployment for *agent*.

        Args:
            agent: Name of the agent to deploy.
            name: Deployment name (auto-generated if omitted).
            deployment_mode: Runtime backend — ``"subprocess"`` (default),
                ``"docker"``, or ``"k8s"``. Container modes run the agent as a
                durable container through the deployments plugin and require a
                configured executor.
            image: Container image for ``docker``/``k8s`` modes. Falls back to
                ``agents.deployments.default_image`` when omitted. Rejected in
                ``subprocess`` mode.
            use_image_entrypoint: For ``docker``/``k8s`` modes, preserve the
                image ENTRYPOINT/CMD instead of injecting the platform-owned
                agent server command.
            environment: Optional AgentEnvironment to deploy under — a
                ``"workspace/name"`` ref to a stored AgentEnvironment, or an
                inline environment dict. Its EnvironmentSpec is merged into the
                agent config and its ComputeSpec/secret refs are snapshotted onto
                the deployment at creation time.
            workspace: Target workspace.

        Returns:
            The created deployment as a dict.
        """
        if image and deployment_mode == "subprocess":
            raise ValueError("image requires deployment_mode='docker' or 'k8s'.")
        if use_image_entrypoint and deployment_mode == "subprocess":
            raise ValueError("use_image_entrypoint requires deployment_mode='docker' or 'k8s'.")
        if isinstance(environment, Mapping):
            environment_body: str | AgentEnvironmentInline | None = AgentEnvironmentInline.model_validate(environment)
        else:
            environment_body = environment
        payload: dict[str, object] = {"agent": agent, "deployment_mode": deployment_mode}
        if name is not None:
            payload["name"] = name
        if image is not None:
            payload["image"] = image
        if use_image_entrypoint:
            payload["use_image_entrypoint"] = True
        if environment_body is not None:
            payload["environment"] = environment_body
        response = self._client.create_deployment(
            workspace=workspace,
            body=CreateDeploymentRequest.model_validate(payload),
        )
        return _json_map_from_response(response)

    def list(self, workspace: str | None = None) -> JsonMap:
        """List all deployments in *workspace*."""
        return _json_map_from_page(self._client.list_deployments(workspace=workspace))

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get a deployment by name."""
        return _json_map_from_response(self._client.get_deployment(workspace=workspace, name=name))

    def delete(self, name: str, workspace: str | None = None) -> None:
        """Mark a deployment for deletion."""
        self._client.delete_deployment(workspace=workspace, name=name).data()


class _EnvironmentSpecResource:
    """AgentEnvironmentSpec CRUD under ``nemo.agents.environment_specs``.

    An EnvironmentSpec is the fulfillment half of the request/fulfill split: the
    Agent declares the dependencies it needs; the spec provides concrete
    endpoints, plaintext env, and secret refs. It is merged into the agent config
    at deployment/job create time.
    """

    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        spec: EnvironmentSpecInline | Mapping[str, object] | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Create an environment spec.

        Args:
            name: Unique environment-spec name within the workspace.
            spec: The environment spec, as a shared :class:`EnvironmentSpecInline`
                model (the typed, discoverable path) or a plain dict. Only the
                fields explicitly set on the model are sent.
            workspace: Target workspace.

        Returns:
            The created AgentEnvironmentSpec as a dict.
        """
        payload = {**_spec_to_dict(spec), "name": name}
        response = self._client.create_environment_spec(
            workspace=workspace,
            body=CreateEnvironmentSpecRequest.model_validate(payload),
        )
        return _json_map_from_response(response)

    def list(self, workspace: str | None = None) -> JsonMap:
        """List environment specs in *workspace*."""
        return _json_map_from_page(self._client.list_environment_specs(workspace=workspace))

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get an environment spec by name."""
        return _json_map_from_response(self._client.get_environment_spec(workspace=workspace, name=name))

    def delete(self, name: str, workspace: str | None = None) -> None:
        """Delete an environment spec by name."""
        self._client.delete_environment_spec(workspace=workspace, name=name).data()


class _EnvironmentResource:
    """AgentEnvironment CRUD under ``nemo.agents.environments``.

    An AgentEnvironment composes an ``environment_spec`` and a ``compute_spec``
    (each a ``"workspace/name"`` ref or an inline object). It is the single thing
    a deployment or execute job references via its ``environment`` field.
    """

    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        spec: AgentEnvironmentInline | Mapping[str, object] | None = None,
        environment_spec: str | Mapping[str, object] | EnvironmentSpecInline | None = None,
        compute_spec: str | Mapping[str, object] | ComputeSpecInline | None = None,
        description: str | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Create an AgentEnvironment.

        Args:
            name: Unique environment name within the workspace.
            spec: The full environment composition as a shared
                :class:`AgentEnvironmentInline` model (the typed, discoverable
                path) or a plain dict. Only the fields explicitly set are sent.
                The ``environment_spec`` / ``compute_spec`` / ``description``
                arguments below override the matching keys from ``spec`` when
                given — handy for the common ref case.
            environment_spec: A ``"workspace/name"`` ref to a stored
                AgentEnvironmentSpec, an inline spec dict, or ``None``.
            compute_spec: A ``"workspace/name"`` ref to a stored AgentComputeSpec,
                an inline spec dict, or ``None``.
            description: Optional human-readable description. Overrides ``spec``'s
                description when passed (including ``""`` to clear it); left unset,
                ``spec``'s value — or the server default — stands.
            workspace: Target workspace.

        Returns:
            The created AgentEnvironment as a dict.
        """
        payload = {**_spec_to_dict(spec), "name": name}
        if description is not None:
            payload["description"] = description
        if environment_spec is not None:
            payload["environment_spec"] = _environment_spec_body(environment_spec)
        if compute_spec is not None:
            payload["compute_spec"] = _compute_spec_body(compute_spec)
        response = self._client.create_environment(
            workspace=workspace,
            body=CreateEnvironmentRequest.model_validate(payload),
        )
        return _json_map_from_response(response)

    def list(self, workspace: str | None = None) -> JsonMap:
        """List environments in *workspace*."""
        return _json_map_from_page(self._client.list_environments(workspace=workspace))

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get an environment by name."""
        return _json_map_from_response(self._client.get_environment(workspace=workspace, name=name))

    def delete(self, name: str, workspace: str | None = None) -> None:
        """Delete an environment by name."""
        self._client.delete_environment(workspace=workspace, name=name).data()


class _ComputeSpecResource:
    """AgentComputeSpec CRUD under ``nemo.agents.compute_specs``.

    A ComputeSpec is a reusable set of k8s-style resource requests/limits an
    invocation runs with.
    """

    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        spec: ComputeSpecInline | Mapping[str, object] | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Create a compute spec.

        Args:
            name: Unique compute-spec name within the workspace.
            spec: The compute spec, as a shared :class:`ComputeSpecInline` model
                (the typed, discoverable path) or a plain dict. Only the fields
                explicitly set on the model are sent.
            workspace: Target workspace.

        Returns:
            The created AgentComputeSpec as a dict.
        """
        payload = {**_spec_to_dict(spec), "name": name}
        response = self._client.create_compute_spec(
            workspace=workspace,
            body=CreateComputeSpecRequest.model_validate(payload),
        )
        return _json_map_from_response(response)

    def list(self, workspace: str | None = None) -> JsonMap:
        """List compute specs in *workspace*."""
        return _json_map_from_page(self._client.list_compute_specs(workspace=workspace))

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get a compute spec by name."""
        return _json_map_from_response(self._client.get_compute_spec(workspace=workspace, name=name))

    def delete(self, name: str, workspace: str | None = None) -> None:
        """Delete a compute spec by name."""
        self._client.delete_compute_spec(workspace=workspace, name=name).data()


def _environment_spec_body(value: str | Mapping[str, object] | EnvironmentSpecInline) -> str | EnvironmentSpecInline:
    if isinstance(value, str):
        return value
    if isinstance(value, EnvironmentSpecInline):
        return value
    return EnvironmentSpecInline.model_validate(value)


def _compute_spec_body(value: str | Mapping[str, object] | ComputeSpecInline) -> str | ComputeSpecInline:
    if isinstance(value, str):
        return value
    if isinstance(value, ComputeSpecInline):
        return value
    return ComputeSpecInline.model_validate(value)


def _execute_job_body(
    spec: Mapping[str, object],
    name: str | None,
    description: str | None,
) -> AgentJobRequest:
    """Build the create-job request body.

    ``name`` is omitted when not supplied so the Jobs service generates a
    unique one; sending a fixed name makes the second submission collide.
    """
    payload: dict[str, object] = {"spec": _json_map(dict(spec))}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    return AgentJobRequest.model_validate(payload)


class _ExecuteJobsResource:
    """Sync ``client.agents.jobs.execute`` — the ``agents.execute`` job collection."""

    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        spec: Mapping[str, object],
        name: str | None = None,
        description: str | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Submit an execute-agent job. *spec* is an ``ExecuteAgentJobConfig``."""
        response = self._client.create_agent_job(
            workspace=workspace,
            collection="execute",
            body=_execute_job_body(spec, name, description),
        )
        return _json_map_from_response(response)

    def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get one execute-agent job by name."""
        response = self._client.get_agent_job(workspace=workspace, collection="execute", name=name)
        return _json_map_from_response(response)

    def list_results(self, name: str, workspace: str | None = None) -> JsonMap:
        """List the named results a finished execute-agent job saved."""
        response = self._client.list_agent_job_results(workspace=workspace, collection="execute", name=name)
        return _json_map_from_response(response)


class _AsyncExecuteJobsResource:
    """Async ``client.agents.jobs.execute``."""

    def __init__(self, client: AsyncAgentsClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        spec: Mapping[str, object],
        name: str | None = None,
        description: str | None = None,
        workspace: str | None = None,
    ) -> JsonMap:
        """Submit an execute-agent job. *spec* is an ``ExecuteAgentJobConfig``."""
        response = await self._client.create_agent_job(
            workspace=workspace,
            collection="execute",
            body=_execute_job_body(spec, name, description),
        )
        return _json_map_from_response(response)

    async def get(self, name: str, workspace: str | None = None) -> JsonMap:
        """Get one execute-agent job by name."""
        response = await self._client.get_agent_job(workspace=workspace, collection="execute", name=name)
        return _json_map_from_response(response)

    async def list_results(self, name: str, workspace: str | None = None) -> JsonMap:
        """List the named results a finished execute-agent job saved."""
        response = await self._client.list_agent_job_results(workspace=workspace, collection="execute", name=name)
        return _json_map_from_response(response)


class _JobsResource:
    """Sync ``client.agents.jobs`` namespace."""

    def __init__(self, client: AgentsClient) -> None:
        self._client = client
        self._execute: _ExecuteJobsResource | None = None

    @property
    def execute(self) -> _ExecuteJobsResource:
        if self._execute is None:
            self._execute = _ExecuteJobsResource(self._client)
        return self._execute


class _AsyncJobsResource:
    """Async ``client.agents.jobs`` namespace."""

    def __init__(self, client: AsyncAgentsClient) -> None:
        self._client = client
        self._execute: _AsyncExecuteJobsResource | None = None

    @property
    def execute(self) -> _AsyncExecuteJobsResource:
        if self._execute is None:
            self._execute = _AsyncExecuteJobsResource(self._client)
        return self._execute


class AsyncAgentsResource:
    """Async SDK namespace for ``nemo.agents.*``.

    Only ``jobs`` is implemented. Agent CRUD, deployments, and ``invoke``
    remain sync-only on :class:`AgentsResource`.
    """

    def __init__(self, platform: AsyncNeMoPlatform) -> None:
        self._platform = platform
        self._client = _async_agents_client_from_platform(platform)
        self._jobs: _AsyncJobsResource | None = None

    @property
    def jobs(self) -> _AsyncJobsResource:
        """Sub-resource for agents job collections."""
        if self._jobs is None:
            self._jobs = _AsyncJobsResource(self._client)
        return self._jobs


agents_sdk_resources = NemoPluginSDKResources(
    sync_resource=AgentsResource,
    async_resource=AsyncAgentsResource,
)
