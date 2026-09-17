# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give each evaluated task its own Platform session on a deployed agent.

A chat-completions request that carries no session header makes Fabric mint a session id
of its own, so the trajectory it exports to Intake lands under an identity nothing else in
the run knows. The scores then have to be published against a *second*, adapter-minted
trajectory, and the two views of the same task never meet.

Creating the session here inverts that: Platform names the conversation, the agents gateway
forwards the name to the deployment's Fabric server, and the trace Fabric exports is the one
the scores attach to.

One session per task, not per run: tasks are independent conversations, and a shared session
would serialize them through a single runtime and interleave their transcripts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar
from urllib.parse import unquote, urlparse

import httpx
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_evaluator_sdk.agent_inference import (
    AgentInferenceContext,
    AgentInferenceFn,
    AgentInferenceFnFactory,
    AgentInvocationResult,
    make_agent_inference_fn,
)
from nemo_evaluator_sdk.values import Agent
from nemo_platform_plugin.agents.client import AgentsClient, AsyncAgentsClient
from nemo_platform_plugin.agents.types import AgentDeployment, AgentSession, CreateSessionRequest
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.response import NemoResponse

logger = logging.getLogger(__name__)

#: Mirrors ``nemo_agents_plugin.session_protocol.SESSION_ID_HEADER``. Restated rather than
#: imported because an evaluator job does not depend on the agents plugin.
SESSION_ID_HEADER = "X-Nemo-Session-Id"

T = TypeVar("T")

_DEPLOYMENT_PATH = re.compile(r"^/apis/agents/v\d+/workspaces/(?P<workspace>[^/]+)/deployments/(?P<deployment>[^/]+)/")


def deployment_ref_from_url(url: str) -> tuple[str, str] | None:
    """``(workspace, deployment name)`` for an agents-gateway deployment URL, else ``None``.

    A target pointed at anything else — a NAT server, a third-party agent, an agents route
    that is not a deployment — has no session to create, which is the ``None`` case.
    """
    match = _DEPLOYMENT_PATH.match(urlparse(url).path)
    if match is None:
        return None
    return unquote(match.group("workspace")), unquote(match.group("deployment"))


class AgentSessionBroker:
    """Own the Platform sessions an agent-target evaluation runs its tasks through.

    Sessions are opened up front rather than on first use, because the task-to-session map
    is what publication keys trajectories on: built lazily, a task that failed before its
    first request would be indistinguishable from one that never had a session.
    """

    def __init__(
        self,
        *,
        open_session: Callable[[], AgentSession],
        close_session: Callable[[str], None],
    ) -> None:
        self._open_session = open_session
        self._close_session = close_session
        self._sessions: dict[str, str] = {}
        self._names: dict[str, str] = {}

    @property
    def sessions(self) -> dict[str, str]:
        """Task id to session id, for every task that got one."""
        return dict(self._sessions)

    def session_for(self, task_id: str) -> str | None:
        return self._sessions.get(task_id)

    def open(self, task_ids: Sequence[str]) -> None:
        """Open one session per task id, skipping any the gateway refuses.

        A refusal costs that task its session identity, not its evaluation: the request still
        runs, exactly as it did before sessions existed.
        """
        for task_id in task_ids:
            try:
                session = self._open_session()
            except Exception:
                logger.warning(
                    "Could not open an agent session for task %r; it runs unsessioned.", task_id, exc_info=True
                )
                continue
            if not session.id:
                logger.warning("Agent session for task %r came back without an id; it runs unsessioned.", task_id)
                continue
            self._sessions[task_id] = session.id
            self._names[task_id] = session.name

    def close(self) -> None:
        """Close every open session, freeing its Fabric runtime now instead of at the idle sweep."""
        for task_id, name in self._names.items():
            if not name:
                continue
            try:
                self._close_session(name)
            except Exception:
                logger.warning("Could not close the agent session for task %r; the idle sweep reclaims it.", task_id)

    def inference_fn_factory(self) -> AgentInferenceFnFactory:
        """An agent-inference factory that stamps each task's session onto its request."""

        def factory(context: AgentInferenceContext) -> AgentInferenceFn:
            base = make_agent_inference_fn(context)
            task_id = context.metadata.get("task_id")
            session_id = self.session_for(task_id) if isinstance(task_id, str) else None
            if session_id is None:
                return base

            def invoke(
                agent: Agent,
                request: dict,
                *,
                client: httpx.AsyncClient | None = None,
                max_retries: int | None = None,
                api_key: str | None = None,
                default_headers: dict[str, str] | None = None,
                timeout: float | None = None,
            ) -> Awaitable[dict | AgentInvocationResult]:
                return base(
                    agent,
                    request,
                    client=client,
                    max_retries=max_retries,
                    api_key=api_key,
                    default_headers={**(default_headers or {}), SESSION_ID_HEADER: session_id},
                    timeout=timeout,
                )

            return invoke

        return factory


def broker_for_target(client: NemoClient | AsyncNemoClient, *, url: str) -> AgentSessionBroker | None:
    """A broker for a platform-routed deployment target, or ``None`` when sessions do not apply.

    Resolving the deployment is what turns the URL's *name* into the immutable id a session is
    created against, so a target naming a deployment that no longer exists yields no broker
    rather than a run that fails once per task at session creation.
    """
    if not client.is_platform_url(url):
        return None
    ref = deployment_ref_from_url(url)
    if ref is None:
        return None
    workspace, deployment_name = ref

    try:
        deployment = _get_deployment(client, workspace=workspace, name=deployment_name)
    except Exception:
        logger.warning(
            "Could not resolve deployment %r in workspace %r; the evaluation runs unsessioned.",
            deployment_name,
            workspace,
            exc_info=True,
        )
        return None

    deployment_id = deployment.id or deployment.entity_id
    if not deployment_id:
        return None

    return AgentSessionBroker(
        open_session=lambda: _create_session(client, workspace=workspace, deployment_id=deployment_id),
        close_session=lambda name: _close_session(client, workspace=workspace, name=name),
    )


def _get_deployment(client: NemoClient | AsyncNemoClient, *, workspace: str, name: str) -> AgentDeployment:
    if isinstance(client, AsyncNemoClient):
        return run_with_isolated_async_client(
            client,
            lambda isolated: _awaited_data(
                AsyncAgentsClient.from_client(isolated).get_deployment(workspace=workspace, name=name)
            ),
        )
    return AgentsClient.from_client(client).get_deployment(workspace=workspace, name=name).data()


def _create_session(client: NemoClient | AsyncNemoClient, *, workspace: str, deployment_id: str) -> AgentSession:
    body = CreateSessionRequest(deployment_id=deployment_id)
    if isinstance(client, AsyncNemoClient):
        return run_with_isolated_async_client(
            client,
            lambda isolated: _awaited_data(
                AsyncAgentsClient.from_client(isolated).create_session(workspace=workspace, body=body)
            ),
        )
    return AgentsClient.from_client(client).create_session(workspace=workspace, body=body).data()


def _close_session(client: NemoClient | AsyncNemoClient, *, workspace: str, name: str) -> None:
    if isinstance(client, AsyncNemoClient):
        run_with_isolated_async_client(
            client,
            lambda isolated: _awaited_data(
                AsyncAgentsClient.from_client(isolated).close_session(workspace=workspace, name=name)
            ),
        )
        return
    AgentsClient.from_client(client).close_session(workspace=workspace, name=name)


async def _awaited_data(pending: Awaitable[NemoResponse[T]]) -> T:
    """Unwrap an async client call to the body the sync path returns directly."""
    return (await pending).data()
