# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local assistant bridge for Studio."""

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from nemo_helix_plugin.authz import CallerKind, path_rule
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityStoreError
from nhx.common.entities.constants import NAME_PATTERN
from nhx.common.service.dependencies import get_entity_client
from nhx.studio import studio_links
from nhx.studio.assistant_artifacts import (
    ChatArtifactsResponse,
    InputSelectionTool,
    answer_selection_pairs,
    record_answer_selections,
    record_assistant_model,
    record_spec_text_artifacts,
    record_tool_artifacts,
    record_tool_name,
    record_user_tool_result_artifacts,
    record_workspace_artifact,
    string_value,
)
from nhx.studio.assistant_mcp_tools import (
    APPROVAL_TOOL_NAME,
    CLAUDE_MCP_SERVER_NAME,
    JOB_PROGRESS_TOOL_NAME,
    MCP_TOOLS,
    SELECT_AGENT_TOOL_NAME,
    SELECT_DATASET_FILE_TOOL_NAME,
    SELECT_EVAL_CONFIG_TOOL_NAME,
    SELECT_MODEL_TOOL_NAME,
    STUDIO_ASSISTANT_CONTEXT,
    STUDIO_LINK_TOOL_NAME,
    allowed_mcp_tools,
    permission_prompt_tool,
)
from nhx.studio.assistant_skills import ClaudeSkillResponse, DuplicateSkillError, list_claude_skill_responses
from nhx.studio.authz import scope
from nhx.studio.entities import AssistantConversation, AssistantMessage, LegacyAssistantConversation
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import NoMatchFound

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/assistant")

MCP_ROUTE_NAME = "studio_assistant_mcp"
PUBLIC_MCP_ROUTE_NAME = "studio_assistant_public_mcp"
PUBLIC_MCP_UNSUPPORTED_METHOD_ROUTE_NAME = "studio_assistant_public_mcp_unsupported_method"
PUBLIC_MCP_PATH = "/studio/api/assistant/mcp/{session_id}"
CLAUDE_MCP_TOOL_TIMEOUT_MS = 2_147_483_647
MCP_KEEPALIVE_INTERVAL_SECONDS = 15
DEFAULT_STUDIO_ASSISTANT_NAME = "nemo-studio-assistant"
DEFAULT_STUDIO_ASSISTANT_BASE_URL = "http://127.0.0.1:8080"
STUDIO_ASSISTANT_TIMEOUT_SECONDS = 600.0
MAX_RETAINED_SESSIONS = 100
MAX_MODEL_CONTEXT_TURNS_PER_SESSION = 8

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
SERVER_CWD = Path(os.getcwd()).resolve()
STUDIO_CONTEXT_START = "<nemo_studio_context>"
STUDIO_CONTEXT_END = "</nemo_studio_context>"
STUDIO_CONTEXT_USER_REQUEST_PREFIX = "User request:"
STUDIO_CONTEXT_PROMPT_HEADER = "You are NeMo Assistant, running inside NeMo Studio."
STUDIO_MESSAGE_SUMMARY_START = "<<<NEMO_STUDIO_MESSAGE_SUMMARY_V1>>>"
STUDIO_MESSAGE_SUMMARY_END = "<<<END_NEMO_STUDIO_MESSAGE_SUMMARY_V1>>>"
WORKSPACE_NAME_RE = re.compile(NAME_PATTERN)


class NewSessionResponse(BaseModel):
    """Response returned when Studio starts a new assistant session."""

    session_id: str


class MessageRequest(BaseModel):
    """A user message to send to the local assistant."""

    message: str = Field(min_length=1)
    studio_base_url: str | None = Field(default=None, min_length=1)
    studio_pathname: str | None = Field(default=None, min_length=1)
    workspace: str | None = Field(default=None, min_length=1, pattern=NAME_PATTERN)

    model_config = ConfigDict(regex_engine="python-re")


class PermissionDecision(BaseModel):
    """Studio's decision for a pending local-agent tool permission request."""

    approved: bool
    reason: str | None = None
    updated_input: dict[str, Any] | None = None


class AgentInputDecision(BaseModel):
    """Studio's value for a pending local-agent UI input request."""

    skipped: bool = False
    value: dict[str, Any] | None = None


class HistorySessionResponse(BaseModel):
    """Summary of a persisted Assistant or legacy Claude session."""

    session_id: str
    mtime: float
    title: str | None = None
    first_prompt: str
    message_count: int
    token_count: int
    tool_call_count: int
    tool_calls: list[str]
    chat_artifacts: ChatArtifactsResponse


class SessionHistoryResponse(BaseModel):
    """Assistant session history normalized for Studio chat replay."""

    session_id: str
    items: list[dict[str, Any]]
    chat_artifacts: ChatArtifactsResponse


_initialized_sessions: set[str] = set()
_session_streams: dict[str, asyncio.Queue[tuple[str, Any]]] = {}
_pending_permissions: dict[str, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
_pending_agent_inputs: dict[str, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
# Cache of Entity-Store-confirmed workspace names, keyed by session then by
# (caller fingerprint, requested workspace), so the membership lookup runs once per
# session/caller/workspace rather than on every message. Session ids are not bound to
# a caller, so the caller's credential participates in the key: a cached authorization
# decision must never be reused for a different caller.
_session_workspace_cache: dict[str, dict[tuple[str, str], str]] = {}
_AGENT_INPUT_RESPONSE_RESERVED_KEYS = frozenset({"message", "status"})


def _recent_conversation_messages(conversation: list[AssistantMessage]) -> list[AssistantMessage]:
    """Bound and normalize model context without truncating persisted history."""
    max_messages = MAX_MODEL_CONTEXT_TURNS_PER_SESSION * 2
    return [
        message.model_copy(
            update={
                "content": (
                    _strip_studio_context_from_prompt(message.content) if message.role == "user" else message.content
                )
            }
        )
        for message in conversation[-max_messages:]
    ]


def _append_conversation_turn(
    conversation: AssistantConversation,
    user_message: str,
    assistant_message: str,
    model: str,
) -> None:
    conversation.messages.extend(
        [
            AssistantMessage(role="user", content=user_message),
            AssistantMessage(role="assistant", content=assistant_message),
        ]
    )
    record_assistant_model(conversation.chat_artifacts, model)


@dataclass
class HistorySummary:
    """Aggregated metadata from a Claude session history file."""

    title: str | None = None
    first_prompt: str | None = None
    message_count: int = 0
    token_count: int = 0
    tool_call_count: int = 0
    tool_calls: list[str] = dataclass_field(default_factory=list)
    chat_artifacts: ChatArtifactsResponse = dataclass_field(default_factory=ChatArtifactsResponse)


@dataclass(frozen=True)
class HistoryToolUse:
    """Tool metadata needed to restore user interactions during history replay."""

    name: str
    input: dict[str, Any]


def _mcp_tools_for_destinations(
    destinations: Mapping[str, studio_links.StudioLinkDestination],
) -> list[dict[str, Any]]:
    return [*MCP_TOOLS, studio_links.tool_for_destinations(destinations)]


def mount_public_mcp_route(app: FastAPI) -> None:
    """Mount the MCP callback under /studio so the local Claude CLI can call it."""
    app.add_api_route(
        PUBLIC_MCP_PATH,
        mcp_endpoint,
        methods=["POST"],
        name=PUBLIC_MCP_ROUTE_NAME,
        include_in_schema=False,
    )
    app.add_api_route(
        PUBLIC_MCP_PATH,
        mcp_unsupported_method,
        methods=["GET", "DELETE"],
        name=PUBLIC_MCP_UNSUPPORTED_METHOD_ROUTE_NAME,
        include_in_schema=False,
    )


def _validate_session_id(session_id: str) -> str:
    try:
        return str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="session_id must be a UUID") from exc


def _conversation_name(session_id: str) -> str:
    """Return the Entity Store name for a Studio session UUID."""
    return f"assistant-{session_id}"


def _legacy_conversation_name(session_id: str) -> str:
    """Return the Entity Store name used before the Assistant rename."""
    return f"copilot-{session_id}"


async def _get_conversation(
    entity_store: EntityClient,
    *,
    session_id: str,
    workspace: str,
) -> AssistantConversation:
    """Load a current conversation, falling back to its pre-rename identity."""
    try:
        return await entity_store.get(
            AssistantConversation,
            _conversation_name(session_id),
            workspace=workspace,
        )
    except EntityNotFoundError:
        return await entity_store.get(
            LegacyAssistantConversation,
            _legacy_conversation_name(session_id),
            workspace=workspace,
        )


def _request_principal_id(request: Request) -> str:
    """Return the end-user principal, including service-on-behalf-of requests."""
    return (
        request.headers.get("x-nhx-principal-on-behalf-of") or request.headers.get("x-nhx-principal-id") or "local-user"
    )


async def _get_owned_conversation(
    entity_store: EntityClient,
    *,
    session_id: str,
    workspace: str,
    owner_id: str,
) -> AssistantConversation:
    """Load a conversation and enforce per-user ownership within a workspace."""
    try:
        conversation = await _get_conversation(
            entity_store,
            session_id=session_id,
            workspace=workspace,
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail="no such session history") from exc
    if conversation.owner_id != owner_id:
        # Do not reveal whether another user's conversation exists.
        raise HTTPException(status_code=404, detail="no such session history")
    return conversation


def _trimmed_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalize_studio_base_url(value: str | None) -> str | None:
    base_url = _trimmed_string(value)
    if not base_url:
        return None

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    return base_url.rstrip("/")


def _studio_base_url_from_referer(value: str | None) -> str | None:
    referer = _trimmed_string(value)
    if not referer:
        return None

    parsed = urlparse(referer)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    base_path = ""
    for marker in ("/workspaces/", "/models"):
        marker_index = parsed.path.find(marker)
        if marker_index >= 0:
            base_path = parsed.path[:marker_index]
            break

    if not base_path and (parsed.path == "/studio" or parsed.path.startswith("/studio/")):
        base_path = "/studio"

    return f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")


def _studio_pathname_from_referer(value: str | None) -> str | None:
    referer = _trimmed_string(value)
    if not referer:
        return None

    parsed = urlparse(referer)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    return parsed.path or None


def _studio_base_url_from_request(body: MessageRequest, request: Request) -> str | None:
    return (
        _studio_base_url_from_referer(request.headers.get("referer"))
        or _normalize_studio_base_url(body.studio_base_url)
        or _normalize_studio_base_url(request.headers.get("origin"))
    )


def _studio_pathname_from_request(body: MessageRequest, request: Request) -> str | None:
    return _trimmed_string(body.studio_pathname) or _studio_pathname_from_referer(request.headers.get("referer"))


def _build_studio_url(studio_base_url: str | None, path: str) -> str | None:
    base_url = _normalize_studio_base_url(studio_base_url)
    if not base_url:
        return None
    return f"{base_url}/{path.lstrip('/')}"


def _strip_studio_context_from_prompt(content: str) -> str:
    wrapper_prefix = f"{STUDIO_CONTEXT_START}\n{STUDIO_CONTEXT_PROMPT_HEADER}\n"
    request_separator = f"\n{STUDIO_CONTEXT_END}\n\n{STUDIO_CONTEXT_USER_REQUEST_PREFIX}\n"
    if not content.startswith(wrapper_prefix):
        return content

    _, separator, request = content.partition(request_separator)
    if not separator:
        return content
    return request.strip() or content


def _build_claude_prompt(
    message: str,
    workspace: str | None,
    studio_base_url: str | None,
    studio_pathname: str | None,
    enabled_destinations: Mapping[str, studio_links.StudioLinkDestination] | None = None,
) -> str:
    return "\n".join(
        [
            STUDIO_CONTEXT_START,
            _build_studio_system_prompt(workspace, studio_base_url, studio_pathname, enabled_destinations),
            STUDIO_CONTEXT_END,
            "",
            STUDIO_CONTEXT_USER_REQUEST_PREFIX,
            message,
        ]
    )


def _build_studio_system_prompt(
    workspace: str | None,
    studio_base_url: str | None,
    studio_pathname: str | None,
    enabled_destinations: Mapping[str, studio_links.StudioLinkDestination] | None = None,
) -> str:
    normalized_base_url = _normalize_studio_base_url(studio_base_url)
    current_studio_route = _trimmed_string(studio_pathname) or "unknown"
    destinations = studio_links.STUDIO_LINK_DESTINATIONS if enabled_destinations is None else enabled_destinations
    lines = [
        STUDIO_CONTEXT_PROMPT_HEADER,
        f"Current Studio workspace: {workspace or 'unknown'}",
        f"Studio UI base URL: {normalized_base_url or 'unknown'}",
        f"Current Studio route path: {current_studio_route}",
        "Enabled Studio link destinations for this Studio instance: "
        f"{studio_links.destination_description(destinations)}.",
        "Only call studio_link with one of the enabled destinations above.",
        "If a Studio page is disabled by feature flag, choose the closest enabled parent/list page instead of linking to the disabled route.",
        "When the user asks for a Studio page link, do not ask them for the base URL.",
        "Always use the current Studio workspace for Studio UI links unless the user explicitly names another workspace.",
        "Do not infer the Studio workspace from the local username, account name, API response defaults, or filesystem paths.",
        "The MCP server URL is an internal callback for tools, not the Studio UI base URL.",
        "Do not invent Studio route paths manually when studio_link can provide the link.",
        "If studio_link is unavailable and you must construct a Studio UI link manually, use only a known enabled Studio route and prefer a relative Markdown link that starts with /workspaces/ or /models/.",
        "Evaluation pages use /workspaces/{workspace}/evaluation/... with singular evaluation; never nest evaluation links under /dashboard/evaluations/.",
        "Interactive Studio choice behavior:",
        "Studio ships dedicated visual picker tools. When a picker fits, you MUST use it instead of plain text.",
        "Whenever you need the user to name, pick, confirm, or disambiguate an agent (including choosing among deployed agents), you MUST call mcp__nemo_studio__select_agent to render the agent dropdown. Never ask for an agent in plain text.",
        "Whenever you need the user to choose a model, you MUST call mcp__nemo_studio__select_model. Never ask for a model in plain text.",
        "Whenever you need a fileset, fileset reference, dataset, or input/source data file (including an anonymizer or evaluation input, or a CSV/Parquet file), you MUST call mcp__nemo_studio__select_dataset_file instead of asking for a fileset reference or '<workspace>/<fileset>#<file>' path in plain text; for an evaluation config file, you MUST call mcp__nemo_studio__select_eval_config.",
        "Treat 'which agent', 'pick an agent', 'choose a model', 'which fileset', and 'what is your fileset reference' as mandatory tool-use requests for the matching select_* tool, exactly like Studio link requests are mandatory studio_link requests.",
        "Set the picker title and description to match the current workflow, for example title='Select agent to audit'.",
        "Only skip a picker when the user already gave the value, the value is already unambiguous from the conversation, or the user explicitly skipped a previous picker.",
        "A timeout, disconnect, or other interactive-tool error is not permission to continue or repeat the question in plain text. Leave the input unresolved and tell the user the interactive request must be retried.",
        "A message that needs user input is not complete until you call the matching Studio input tool. Never end a message with only a plain-text question when an interactive tool applies.",
        "In particular, if you need an agent, model, dataset file, or evaluation config, call the matching select_* tool before completing the message; mentioning the needed selection in prose is not a substitute for the tool call.",
        "For any finite set of choices without a dedicated select_* picker — including yes/no, multiple-choice, 'pick one of these' (for example deployments, jobs, or next actions), or whenever you would offer the user options to choose from — you MUST call AskUserQuestion to render a selectable options picker instead of listing the choices in plain text. Only ask a concise plain-text question for genuinely open-ended, free-form input that has no discrete options.",
        "Conditional message-summary behavior:",
        "Use a Studio summary block only after substantive work that benefits from collapsing details.",
        "A summary block is required when you called one or more tools, ran commands, changed files or platform state, performed a multi-step investigation, or produced a long detailed response.",
        "For a short informational answer, definition, confirmation, greeting, or simple clarification completed without tool calls, omit the summary block entirely and return the normal answer so Studio shows it in full.",
        "Do not emit the summary markers with a shortened duplicate of an already concise answer.",
        f"Start the summary block on its own line with {STUDIO_MESSAGE_SUMMARY_START} and end it on its own line with {STUDIO_MESSAGE_SUMMARY_END}.",
        "Inside the summary block, use exactly these fields on separate lines:",
        f"{STUDIO_MESSAGE_SUMMARY_START}",
        "title: <meaningful 3-7 word title naming the user's overall task; keep it stable across messages unless the topic clearly changes>",
        "worked_for: <elapsed time if you know it, otherwise unknown>",
        "summary: <concise Markdown, at most 60 words, describing the user-visible result and current state>",
        "details_label: worked for <same elapsed time or unknown>",
        f"{STUDIO_MESSAGE_SUMMARY_END}",
        "Keep detailed explanation, command output, tool logs, diffs, and step-by-step work outside the summary block and before it.",
        "Do not put raw command output, code diffs, tool logs, or long reasoning inside the summary block.",
        "Format the summary as valid Markdown. When it contains multiple suggestions, results, or steps, put a blank line after 'summary:' and use a numbered or bulleted list rather than flattening the items into prose.",
        "If the detailed message body contains any Markdown links, repeat those links at the bottom of the summary so they remain visible when the details are collapsed.",
        "Put repeated links on separate lines without a heading and without formatting them as a bulleted or numbered list.",
        "Studio will use this block to collapse everything before it behind a 'worked for <time>' accordion and show only the short summary by default.",
        "When you are interrupted by a permission request or input request after substantive work, include the summary block describing what happened so far and what you need next.",
        "When you only need to ask one simple clarification and have not performed substantive work or called a tool, ask it normally without a summary block.",
        "When user input is still required, the summary's final sentence MUST state the exact unresolved selection or action, for example 'Select an agent to continue.' Never show only the investigation result while hiding the request for input in the collapsed details.",
        "Prefer NeMo Studio MCP tools and Studio views over CLI commands for user-facing follow-up actions, navigation, inspection, and status/result review.",
        "Do not tell the user to run nemo CLI commands, shell commands, curl commands, or status commands to inspect agents, jobs, evaluations, filesets, models, traces, logs, or results when a Studio view, Studio link, or Studio progress card is available for the same purpose.",
        "Use CLI commands only to perform work that has no Studio UI equivalent, when the user explicitly asks for CLI/debugging, or when you must gather data that Studio tools cannot provide.",
        "Required Studio-link behavior:",
        "Default to trying to include a Studio link in Studio-related responses.",
        "When your answer mentions or depends on a Studio resource, page, workflow, or result, first choose the nearest studio_link destination and include that link unless no relevant Studio page exists.",
        "When you are unsure which detail page applies, link to the closest list page for the current workspace instead of omitting a link.",
        "Direct Studio link requests are mandatory tool-use requests.",
        "When the user asks for a link, URL, clickable link, href, where to open, where to find, how to view, or how to chat with a Studio resource or page, call mcp__nemo_studio__studio_link before responding.",
        "Never answer a Studio link request by saying you cannot generate URLs, do not know the port, do not know the base URL, or need the user to provide the Studio URL.",
        "After any successful Studio action, you must include a Studio link in the response even if the user did not ask for one.",
        "Before your final response for any successful create, start, deploy, evaluate, inspect, or modify action, call mcp__nemo_studio__studio_link and include the returned markdown exactly.",
        "Never finish a successful Studio action without a visible Markdown link to the most relevant Studio page.",
        "Required job-progress behavior:",
        "Whenever you start, submit, or kick off any platform job and you know its job name, you MUST call mcp__nemo_studio__job_progress with that job name before your final response, once for every job you launch.",
        "Do not replace the job_progress card with a plain-text job summary or by telling the user to run a status command; call job_progress in addition to any Studio link.",
        "Use the returned markdown from studio_link exactly; do not replace it with localhost, the API host, or the MCP server host.",
        "If the user asks for an agent link and an agent name is known from the conversation, use destination='agent' with that name; otherwise use destination='agents'.",
        "If the user asks for an agent chat or playground link and an agent name is known from the conversation, use destination='agent_chat' with that name; otherwise use destination='agents'.",
        "If the user asks for a deployment, deployment chat, or deployment playground link and the agent name is known from the conversation, use destination='agent_chat' with the agent name; otherwise use destination='agents'.",
        "For a newly started job, use destination='job' and the job name when available; otherwise use destination='jobs'.",
        "For generated filesets, custom models, deployments, evaluations, guardrails, secrets, Data Designer, Safe Synthesizer, settings, members, or intake work, choose the matching studio_link destination.",
        "For created datasets or filesets use destination='fileset_panel' with the fileset name when available; otherwise use destination='filesets'.",
        "For started evaluations use destination='evaluation_result' with the result or job name when available; otherwise use destination='evaluation_results' or destination='evaluation_metrics'.",
        "For the evaluation results list specifically, use destination='evaluation_results'; it resolves to /workspaces/{workspace}/evaluation/results.",
        "For Data Designer jobs use destination='data_designer_job' with the job name when available; otherwise use destination='data_designer'.",
        "For Safe Synthesizer jobs use destination='safe_synthesizer_job' or destination='safe_synthesizer_report' with the job name when available; otherwise use destination='safe_synthesizer'.",
        "For Base Models or available base models use destination='base_models'.",
        "For Custom Models or customization jobs use destination='customizations'; never use customizations for Base Models.",
        "For Agents use destination='agents'.",
    ]
    if "agent_chat" in destinations:
        lines.extend(
            [
                "For a newly created agent, use studio_link with destination='agent_chat' and the agent name when available; otherwise use destination='agents'.",
                "For a newly deployed agent, use destination='agent_chat' and the agent name when available; otherwise use destination='agents'.",
            ]
        )
    if "model_chat" in destinations:
        lines.extend(
            [
                "When the user wants to chat with, try, compare, validate, or test a model, call studio_link with destination='model_chat' and point them to the Studio Chat page.",
                "Do not list agents or ask the user to choose an agent for model-chat intent unless the user explicitly asks to chat with an agent.",
                "For model chat, model comparison, or trying an available model, use destination='model_chat'.",
            ]
        )
    else:
        lines.append(
            "The model_chat destination is not enabled in this Studio instance; do not link to the Studio Chat page."
        )
    return "\n".join(lines)


def _build_assistant_system_prompt(
    session_id: str,
    workspace: str | None,
    studio_base_url: str | None,
    studio_pathname: str | None,
    enabled_destinations: Mapping[str, studio_links.StudioLinkDestination] | None = None,
) -> str:
    """Adapt the existing Studio assistant rules to the deployed NeMo Assistant tools."""
    context = STUDIO_ASSISTANT_CONTEXT
    prompt = _build_studio_system_prompt(
        workspace,
        studio_base_url,
        studio_pathname,
        enabled_destinations,
    )
    tool_names = (
        APPROVAL_TOOL_NAME,
        SELECT_AGENT_TOOL_NAME,
        SELECT_EVAL_CONFIG_TOOL_NAME,
        SELECT_DATASET_FILE_TOOL_NAME,
        SELECT_MODEL_TOOL_NAME,
        JOB_PROGRESS_TOOL_NAME,
        STUDIO_LINK_TOOL_NAME,
    )
    for tool_name in tool_names:
        context = context.replace(f"mcp__{CLAUDE_MCP_SERVER_NAME}__{tool_name}", tool_name)
        prompt = prompt.replace(f"mcp__{CLAUDE_MCP_SERVER_NAME}__{tool_name}", tool_name)
    # Claude Code's native options picker is AskUserQuestion; the deployed NeMo
    # agent exposes it as the ask_user_question tool. Map the name so the
    # "use the options picker" directives resolve to the deployed tool.
    context = context.replace("AskUserQuestion", "ask_user_question")
    prompt = prompt.replace("AskUserQuestion", "ask_user_question")
    return "\n".join(
        [
            context,
            prompt,
            "Your identity in this interface is NeMo Assistant.",
            "Do not describe yourself as a different assistant or provider.",
            "Deployed NeMo Assistant callback behavior:",
            (
                f"For every select_agent, select_model, select_dataset_file, select_eval_config, "
                f"job_progress, studio_link, and ask_user_question call, pass studio_session_id='{session_id}'."
            ),
            "Do not reveal the Studio session id to the user.",
            (
                "The approval context for nemo_api is injected by Studio and is not a tool argument. "
                "Read-only calls run immediately; mutating calls will automatically pause for the user's approval."
            ),
        ]
    )


def _project_history_dir() -> Path:
    encoded = str(SERVER_CWD).replace("/", "-")
    return CLAUDE_PROJECTS_DIR / encoded


_TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _int_metric(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _append_unique_string(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _usage_token_count(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    return sum(_int_metric(usage.get(field)) for field in _TOKEN_USAGE_FIELDS)


def _tool_result_token_count(tool_result: Any) -> int:
    if not isinstance(tool_result, dict):
        return 0
    total_tokens = _int_metric(tool_result.get("totalTokens"))
    if total_tokens:
        return total_tokens
    return _usage_token_count(tool_result.get("usage"))


def _usage_identity(entry: dict[str, Any], message: dict[str, Any]) -> tuple[str, str] | None:
    request_id = entry.get("requestId")
    message_id = message.get("id")
    if not isinstance(request_id, str) and not isinstance(message_id, str):
        return None
    return (request_id if isinstance(request_id, str) else "", message_id if isinstance(message_id, str) else "")


def _append_tool_call(summary: HistorySummary, tool_name: str) -> None:
    summary.tool_call_count += 1
    _append_unique_string(summary.tool_calls, tool_name)
    record_tool_name(summary.chat_artifacts, tool_name)


def _studio_message_title(text: str) -> str | None:
    summary_start = text.find(STUDIO_MESSAGE_SUMMARY_START)
    if summary_start < 0:
        return None

    summary_end = text.find(STUDIO_MESSAGE_SUMMARY_END, summary_start)
    block = text[summary_start + len(STUDIO_MESSAGE_SUMMARY_START) : summary_end if summary_end >= 0 else None]
    match = re.search(
        r"(?:^|\s)title:\s*(.+?)(?=\s+(?:worked_for|summary|details_label):\s*|$)",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    return _trimmed_string(" ".join(match.group(1).split()))


def _record_assistant_tool_calls(
    summary: HistorySummary,
    message: dict[str, Any],
    seen_tool_use_ids: set[str],
    question_labels_by_tool_use_id: dict[str, dict[str, str]],
    input_selection_tools_by_tool_use_id: dict[str, InputSelectionTool],
) -> None:
    for part in message.get("content") or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = string_value(part.get("text"))
            if text:
                summary.title = _studio_message_title(text) or summary.title
                record_spec_text_artifacts(summary.chat_artifacts, text)
            continue
        if part.get("type") != "tool_use":
            continue
        tool_use_id = part.get("id")
        if isinstance(tool_use_id, str):
            if tool_use_id in seen_tool_use_ids:
                continue
            seen_tool_use_ids.add(tool_use_id)
        tool_name = part.get("name")
        tool_name = tool_name if isinstance(tool_name, str) and tool_name else "tool"
        _append_tool_call(summary, tool_name)
        record_tool_artifacts(
            summary.chat_artifacts,
            tool_name,
            part.get("input") or {},
            tool_use_id if isinstance(tool_use_id, str) else None,
            question_labels_by_tool_use_id,
            input_selection_tools_by_tool_use_id,
        )


def _summarize_history_session(path: Path) -> HistorySummary:
    summary = HistorySummary()
    seen_usage_events: set[tuple[str, str]] = set()
    seen_tool_use_ids: set[str] = set()
    question_labels_by_tool_use_id: dict[str, dict[str, str]] = {}
    input_selection_tools_by_tool_use_id: dict[str, InputSelectionTool] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isSidechain"):
                    continue
                if not isinstance(entry, dict):
                    continue

                message = entry.get("message")
                if isinstance(message, dict):
                    record_assistant_model(summary.chat_artifacts, string_value(message.get("model")))
                    usage_identity = _usage_identity(entry, message)
                    if usage_identity is None or usage_identity not in seen_usage_events:
                        summary.token_count += _usage_token_count(message.get("usage"))
                        if usage_identity is not None:
                            seen_usage_events.add(usage_identity)

                summary.token_count += _tool_result_token_count(entry.get("toolUseResult"))

                entry_type = entry.get("type")
                if entry_type == "assistant" and isinstance(message, dict):
                    _record_assistant_tool_calls(
                        summary,
                        message,
                        seen_tool_use_ids,
                        question_labels_by_tool_use_id,
                        input_selection_tools_by_tool_use_id,
                    )
                elif entry_type == "user" and isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        record_workspace_artifact(summary.chat_artifacts, content)
                        record_answer_selections(summary.chat_artifacts, content)
                        content = _strip_studio_context_from_prompt(content)
                        summary.message_count += 1
                        if summary.first_prompt is None:
                            summary.first_prompt = content
                    else:
                        record_user_tool_result_artifacts(
                            summary.chat_artifacts,
                            content,
                            question_labels_by_tool_use_id,
                            input_selection_tools_by_tool_use_id,
                        )
    except OSError:
        return HistorySummary()
    return summary


def _extract_assistant_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif part_type == "thinking":
            thinking = part.get("thinking")
            if isinstance(thinking, str) and thinking:
                parts.append({"type": "thinking", "thinking": thinking})
        elif part_type == "tool_use":
            tool_use = {
                "type": "tool_use",
                "name": part.get("name") or "tool",
                "input": part.get("input") or {},
            }
            tool_use_id = part.get("id")
            if isinstance(tool_use_id, str) and tool_use_id:
                tool_use["id"] = tool_use_id
            parts.append(tool_use)
    return parts


@router.post("/sessions", response_model=NewSessionResponse)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def create_session(
    request: Request,
    workspace: str = "default",
    entity_store: EntityClient = Depends(get_entity_client),
) -> NewSessionResponse:
    """Create a durable, user-owned Assistant session."""
    workspace = _validated_workspace_or_default(workspace)
    session_id = str(uuid.uuid4())
    await entity_store.create(
        AssistantConversation(
            name=_conversation_name(session_id),
            workspace=workspace,
            session_id=session_id,
            owner_id=_request_principal_id(request),
        )
    )
    return NewSessionResponse(session_id=session_id)


@router.get("/history/sessions", response_model=list[HistorySessionResponse])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def list_history_sessions(
    request: Request,
    workspace: str = "default",
    entity_store: EntityClient = Depends(get_entity_client),
) -> list[HistorySessionResponse]:
    """List the current user's durable NeMo Assistant sessions."""
    workspace = _validated_workspace_or_default(workspace)
    owner_id = _request_principal_id(request)
    results = [
        await entity_store.list(
            entity_type,
            workspace=workspace,
            filter_obj={"owner_id": owner_id},
            sort="-updated_at",
            page_size=MAX_RETAINED_SESSIONS,
        )
        for entity_type in (AssistantConversation, LegacyAssistantConversation)
    ]
    sessions: list[HistorySessionResponse] = []
    seen_session_ids: set[str] = set()
    for conversation in (conversation for result in results for conversation in result.data):
        if conversation.session_id in seen_session_ids:
            continue
        seen_session_ids.add(conversation.session_id)
        user_messages = [message.content for message in conversation.messages if message.role == "user"]
        if not user_messages:
            continue
        first_prompt = user_messages[0]
        modified_at = conversation.updated_at or conversation.created_at
        sessions.append(
            HistorySessionResponse(
                session_id=conversation.session_id,
                mtime=modified_at.timestamp() if modified_at else 0,
                title=first_prompt.splitlines()[0][:80],
                first_prompt=first_prompt,
                message_count=len(user_messages),
                token_count=0,
                tool_call_count=0,
                tool_calls=[],
                chat_artifacts=conversation.chat_artifacts,
            )
        )
    sessions.sort(key=lambda session: session.mtime, reverse=True)
    return sessions


def _history_tool_result_text(content: Any) -> str | None:
    if isinstance(content, str):
        return _trimmed_string(content)
    if not isinstance(content, list):
        return None
    text_parts = [
        text for part in content if isinstance(part, dict) and (text := _trimmed_string(part.get("text"))) is not None
    ]
    return "\n".join(text_parts) or None


def _history_interaction_text(tool_use: HistoryToolUse, result_text: str) -> str | None:
    short_name = tool_use.name.rsplit("__", maxsplit=1)[-1]
    if tool_use.name == "AskUserQuestion" or short_name == "ask_user_question":
        try:
            result_value = json.loads(result_text)
        except json.JSONDecodeError:
            result_value = None
        if isinstance(result_value, dict):
            result_text = _trimmed_string(result_value.get("response")) or result_text
        pairs = answer_selection_pairs(result_text)
        return "\n\n".join(f"{question}\n{answer}" for question, answer in pairs) or None

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict) or result.get("status") != "submitted":
        return None

    if short_name == SELECT_AGENT_TOOL_NAME:
        agent = _trimmed_string(result.get("agent"))
        return f"Selected agent: {agent}" if agent else None
    if short_name == SELECT_MODEL_TOOL_NAME:
        output_key = _trimmed_string(tool_use.input.get("output_key")) or "model"
        model = _trimmed_string(result.get(output_key))
        display_label = _trimmed_string(tool_use.input.get("display_label")) or "Selected model"
        return f"{display_label}: {model}" if model else None
    if short_name == SELECT_DATASET_FILE_TOOL_NAME:
        fileset = _trimmed_string(result.get("dataset_fileset"))
        path = _trimmed_string(result.get("dataset_path"))
        return f"Selected dataset: {fileset}/{path}" if fileset and path else None
    if short_name == SELECT_EVAL_CONFIG_TOOL_NAME:
        if result.get("needs_eval_config") is True:
            return "I don't have an evaluation config yet"
        if result.get("use_sample_eval_config") is True:
            return "Use sample evaluation config"
        fileset = _trimmed_string(result.get("eval_config_fileset"))
        path = _trimmed_string(result.get("eval_config"))
        return f"Selected eval config: {fileset}/{path}" if fileset and path else None
    return None


def _history_user_interaction_texts(
    content: Any,
    tool_uses: dict[str, HistoryToolUse],
) -> list[str]:
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "tool_result":
            continue
        tool_use_id = _trimmed_string(part.get("tool_use_id"))
        tool_use = tool_uses.get(tool_use_id) if tool_use_id else None
        result_text = _history_tool_result_text(part.get("content"))
        if tool_use is None or result_text is None:
            continue
        display_text = _history_interaction_text(tool_use, result_text)
        if display_text:
            texts.append(display_text)
    return texts


@router.get("/history/sessions/{session_id}", response_model=SessionHistoryResponse)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def get_session_history(
    session_id: str,
    request: Request,
    workspace: str = "default",
    entity_store: EntityClient = Depends(get_entity_client),
) -> SessionHistoryResponse:
    """Load a NeMo Assistant session or legacy Claude history for replay."""
    sid = _validate_session_id(session_id)
    workspace = _validated_workspace_or_default(workspace)
    try:
        conversation = await _get_conversation(entity_store, session_id=sid, workspace=workspace)
    except EntityNotFoundError:
        conversation = None
    if conversation is not None:
        if conversation.owner_id != _request_principal_id(request):
            raise HTTPException(status_code=404, detail="no such session history")
        items: list[dict[str, Any]] = []
        for message in conversation.messages:
            if message.role == "user":
                items.append({"kind": "user", "text": message.content})
            elif message.role == "assistant":
                items.append(
                    {
                        "kind": "assistant",
                        "parts": [{"type": "text", "text": message.content}],
                    }
                )
        return SessionHistoryResponse(
            session_id=sid,
            items=items,
            chat_artifacts=conversation.chat_artifacts,
        )

    path = _project_history_dir() / f"{sid}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such session history")

    items = []
    summary = _summarize_history_session(path)
    tool_uses: dict[str, HistoryToolUse] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isSidechain"):
                    continue

                entry_type = entry.get("type")
                message = entry.get("message")
                if entry_type == "user" and isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        content = _strip_studio_context_from_prompt(content)
                        items.append({"kind": "user", "text": content})
                    else:
                        items.extend(
                            {"kind": "user", "text": text}
                            for text in _history_user_interaction_texts(content, tool_uses)
                        )
                elif entry_type == "assistant" and isinstance(message, dict):
                    parts = _extract_assistant_parts(message.get("content"))
                    if parts:
                        items.append({"kind": "assistant", "parts": parts})
                    for part in parts:
                        tool_use_id = _trimmed_string(part.get("id"))
                        tool_name = _trimmed_string(part.get("name"))
                        tool_input = part.get("input")
                        if tool_use_id and tool_name:
                            tool_uses[tool_use_id] = HistoryToolUse(
                                name=tool_name,
                                input=tool_input if isinstance(tool_input, dict) else {},
                            )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _initialized_sessions.add(sid)
    return SessionHistoryResponse(session_id=sid, items=items, chat_artifacts=summary.chat_artifacts)


@router.delete("/history/sessions/{session_id}", status_code=204)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def delete_session_history(
    session_id: str,
    request: Request,
    workspace: str = "default",
    entity_store: EntityClient = Depends(get_entity_client),
) -> Response:
    """Delete the current user's persisted Assistant conversation."""
    sid = _validate_session_id(session_id)
    workspace = _validated_workspace_or_default(workspace)
    if sid in _session_streams:
        raise HTTPException(status_code=409, detail="cannot delete a session while it is running")
    conversation = await _get_owned_conversation(
        entity_store,
        session_id=sid,
        workspace=workspace,
        owner_id=_request_principal_id(request),
    )
    try:
        await entity_store.delete(
            type(conversation),
            conversation.name,
            workspace=workspace,
            expected_db_version=conversation.db_version,
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail="no such session history") from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=409, detail="session changed; refresh history and try again") from exc
    _session_workspace_cache.pop(sid, None)
    return Response(status_code=204)


@router.get("/skills", response_model=list[ClaudeSkillResponse])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
def list_claude_skills() -> list[ClaudeSkillResponse]:
    """List NeMo skills that the repo's Claude Code installer exposes."""
    try:
        return list_claude_skill_responses(SERVER_CWD)
    except DuplicateSkillError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _mcp_url(
    request: Request,
    session_id: str,
    workspace: str | None,
    studio_base_url: str | None,
) -> str:
    for route_name in (PUBLIC_MCP_ROUTE_NAME, MCP_ROUTE_NAME):
        try:
            url = str(request.url_for(route_name, session_id=session_id))
            query_params = {}
            if workspace:
                query_params["workspace"] = workspace
            if studio_base_url:
                query_params["studio_base_url"] = studio_base_url
            return f"{url}?{urlencode(query_params)}" if query_params else url
        except NoMatchFound:
            continue
    raise RuntimeError("Studio assistant MCP route is not mounted")


def _build_claude_argv(
    session_id: str,
    message: str,
    mcp_url: str,
    studio_system_prompt: str | None = None,
) -> list[str]:
    mcp_config = json.dumps(
        {
            "mcpServers": {
                CLAUDE_MCP_SERVER_NAME: {
                    "type": "http",
                    "url": mcp_url,
                    # Studio input tools intentionally stay open until the user responds. Use the
                    # largest safe JavaScript timer value so Claude Code does not apply its
                    # otherwise-observed 60-second per-server tool timeout.
                    "timeout": CLAUDE_MCP_TOOL_TIMEOUT_MS,
                }
            }
        }
    )
    session_flag = "-r" if session_id in _initialized_sessions else "--session-id"
    argv = [
        "claude",
        "-p",
        message,
        "--output-format",
        "stream-json",
        "--verbose",
        "--mcp-config",
        mcp_config,
        "--allowedTools",
        ",".join(allowed_mcp_tools(CLAUDE_MCP_SERVER_NAME)),
        "--append-system-prompt",
        STUDIO_ASSISTANT_CONTEXT,
        "--permission-prompt-tool",
        permission_prompt_tool(CLAUDE_MCP_SERVER_NAME),
    ]
    if studio_system_prompt:
        argv.extend(["--append-system-prompt", studio_system_prompt])
    argv.extend([session_flag, session_id])
    return argv


def _claude_env() -> dict[str, str]:
    """Build a clean environment so Claude Code uses its own local auth."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ANTHROPIC_") and key != "CLAUDECODE" and not key.startswith("CLAUDE_CODE_")
    }


def _sse(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def _mcp_tool_result_payload(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(result)}],
        },
    }


async def _stream_mcp_tool_result(
    request_id: Any,
    result: Awaitable[dict[str, Any]],
) -> AsyncIterator[str]:
    """Keep a blocking MCP HTTP call alive until Studio receives user input."""
    task = asyncio.ensure_future(result)
    try:
        # Flush the SSE response headers immediately, then send comments often enough to keep
        # reverse proxies from treating the user-driven request as an idle connection.
        yield ": keepalive\n\n"
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=MCP_KEEPALIVE_INTERVAL_SECONDS)
            if not done:
                yield ": keepalive\n\n"

        payload = _mcp_tool_result_payload(request_id, await task)
        yield _sse(json.dumps(payload, separators=(",", ":")), event="message")
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _blocking_mcp_tool_response(
    session_id: str,
    request_id: Any,
    result: Awaitable[dict[str, Any]],
) -> Response:
    if session_id not in _session_streams:
        return JSONResponse(_mcp_tool_result_payload(request_id, await result))

    return StreamingResponse(
        _stream_mcp_tool_result(request_id, result),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


async def _request_permission(session_id: str, args: dict[str, Any]) -> dict[str, Any]:
    queue = _session_streams.get(session_id)
    if queue is None:
        return {"behavior": "deny", "message": "no active Studio assistant session"}

    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    _pending_permissions[request_id] = (session_id, future)

    payload = json.dumps(
        {
            "request_id": request_id,
            "tool_name": args.get("tool_name"),
            "input": args.get("input") or {},
            "tool_use_id": args.get("tool_use_id"),
        }
    )
    await queue.put(("permission_request", payload))

    try:
        decision = await future
    finally:
        _pending_permissions.pop(request_id, None)

    if decision.get("approved"):
        updated = decision.get("updated_input")
        if updated is None:
            updated = args.get("input") or {}
        return {"behavior": "allow", "updatedInput": updated}
    return {"behavior": "deny", "message": decision.get("reason") or "denied by user"}


async def _request_agent_input(session_id: str, kind: str, args: dict[str, Any]) -> dict[str, Any]:
    queue = _session_streams.get(session_id)
    if queue is None:
        return {"status": "error", "message": "no active Studio assistant session"}

    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    _pending_agent_inputs[request_id] = (session_id, future)

    payload = json.dumps(
        {
            "request_id": request_id,
            "kind": kind,
            "input": args,
        }
    )
    await queue.put(("input_request", payload))

    try:
        decision = await future
    finally:
        _pending_agent_inputs.pop(request_id, None)

    if decision.get("skipped"):
        return {"status": "skipped"}

    value = decision.get("value")
    if isinstance(value, dict):
        reserved_keys = sorted(_AGENT_INPUT_RESPONSE_RESERVED_KEYS.intersection(value))
        if reserved_keys:
            return {
                "status": "error",
                "message": f"input value included reserved keys: {', '.join(reserved_keys)}",
            }
        return {"status": "submitted", **value}

    return {"status": "error", "message": "input request resolved without a value"}


async def _pump_stdout(
    proc: asyncio.subprocess.Process,
    queue: asyncio.Queue[tuple[str, Any]],
) -> None:
    if proc.stdout is None:
        await queue.put(("end", None))
        return

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        payload = line.decode(errors="replace").rstrip("\n")
        if payload:
            await queue.put(("claude", payload))
    await queue.put(("end", None))


async def _pump_stderr(proc: asyncio.subprocess.Process, stderr_chunks: list[str]) -> None:
    if proc.stderr is None:
        return

    while True:
        line = await proc.stderr.readline()
        if not line:
            break
        stderr_chunks.append(line.decode(errors="replace"))


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def _stream_claude(
    session_id: str,
    message: str,
    mcp_url: str,
    studio_system_prompt: str | None = None,
) -> AsyncIterator[str]:
    if shutil.which("claude") is None:
        yield _sse(
            json.dumps({"exit_code": None, "stderr": "Claude Code CLI not found on PATH"}),
            event="error",
        )
        return

    if session_id in _session_streams:
        yield _sse(
            json.dumps({"exit_code": None, "stderr": "session already has an active stream"}),
            event="error",
        )
        return

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    _session_streams[session_id] = queue
    argv = _build_claude_argv(session_id, message, mcp_url, studio_system_prompt)
    stderr_chunks: list[str] = []
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(SERVER_CWD),
            env=_claude_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        logger.exception("Failed to start Claude Code subprocess for session %s", session_id)
        _session_streams.pop(session_id, None)
        yield _sse(
            json.dumps({"exit_code": None, "stderr": "Failed to start Claude Code process"}),
            event="error",
        )
        return

    stdout_task = asyncio.create_task(_pump_stdout(proc, queue))
    stderr_task = asyncio.create_task(_pump_stderr(proc, stderr_chunks))

    try:
        while True:
            try:
                event_type, payload = await asyncio.wait_for(queue.get(), timeout=MCP_KEEPALIVE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                yield ":\n\n"
                continue
            if event_type == "end":
                break
            if event_type == "claude":
                yield _sse(payload)
            elif event_type == "permission_request":
                yield _sse(payload, event="permission_request")
            elif event_type == "input_request":
                yield _sse(payload, event="input_request")
            elif event_type == "permission_expired":
                yield _sse(payload, event="permission_expired")
            elif event_type == "input_expired":
                yield _sse(payload, event="input_expired")

        returncode = await proc.wait()
        if stderr_task is not None:
            await stderr_task

        if returncode == 0:
            _initialized_sessions.add(session_id)
            yield _sse("", event="done")
        else:
            yield _sse(
                json.dumps({"exit_code": returncode, "stderr": "".join(stderr_chunks)}),
                event="error",
            )
    except asyncio.CancelledError:
        await _terminate_process(proc)
        raise
    finally:
        _session_streams.pop(session_id, None)
        for task in (stdout_task, stderr_task):
            if task is not None and not task.done():
                task.cancel()


def _studio_assistant_name() -> str:
    return _trimmed_string(os.environ.get("STUDIO_ASSISTANT_NAME")) or DEFAULT_STUDIO_ASSISTANT_NAME


def _studio_assistant_base_url() -> str:
    base_url = (
        _trimmed_string(os.environ.get("STUDIO_ASSISTANT_BASE_URL"))
        or _trimmed_string(os.environ.get("NHX_BASE_URL"))
        or DEFAULT_STUDIO_ASSISTANT_BASE_URL
    )
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("The configured assistant base URL is invalid")
    return base_url.rstrip("/")


def _validated_workspace_or_default(value: str | None) -> str:
    workspace = _trimmed_string(value)
    if not workspace:
        return "default"
    if WORKSPACE_NAME_RE.fullmatch(workspace) is None:
        raise HTTPException(status_code=400, detail="workspace must match the expected entity-name pattern")
    return workspace


# Bounds for the workspace-membership lookup below. The entities list is scoped to
# the caller's own memberships (forwarded auth), so the page count is normally tiny.
_WORKSPACE_LOOKUP_PAGE_SIZE = 100
_WORKSPACE_LOOKUP_MAX_PAGES = 50


def _caller_fingerprint(headers: Mapping[str, str]) -> str:
    """Return a stable, non-reversible fingerprint of the caller's forwarded credentials.

    Used only to scope cached authorization decisions to the caller they were made
    for. The raw credential is never retained -- just the digest.
    """
    material = json.dumps({name.lower(): value for name, value in sorted(headers.items())}, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def _authorized_workspace(workspace: str, headers: Mapping[str, str], session_id: str) -> str:
    """Confirm the caller may target ``workspace`` and return the platform's own
    spelling of the name.

    The value is looked up in the Entity Store (scoped to the caller's auth) and the
    returned name is taken from the *response*, not from client input, so the value
    that later becomes part of the agent request URL cannot be attacker-controlled.
    The list request itself carries no user-provided value in its URL.

    The confirmed name is cached per (session, caller, requested workspace) so the
    lookup runs once per session/caller/workspace instead of on every message; only
    successful resolutions are cached. The caller fingerprint is part of the key so
    one caller's authorization decision is never reused for another.
    """
    # ``default`` is our own fallback constant, always addressable; skip the lookup.
    if workspace == "default":
        return "default"

    cache_key = (_caller_fingerprint(headers), workspace)
    cached = _session_workspace_cache.get(session_id, {}).get(cache_key)
    if cached is not None:
        return cached

    base_url = _studio_assistant_base_url()
    list_url = f"{base_url}/apis/entities/v2/workspaces"
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for page in range(1, _WORKSPACE_LOOKUP_MAX_PAGES + 1):
                response = await client.get(
                    list_url,
                    headers=dict(headers),
                    params={"page": page, "page_size": _WORKSPACE_LOOKUP_PAGE_SIZE},
                )
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                if not isinstance(data, list) or not data:
                    break
                for entry in data:
                    name = entry.get("name") if isinstance(entry, dict) else None
                    if isinstance(name, str) and name == workspace:
                        # ``name`` is sourced from the platform response, not the request.
                        _session_workspace_cache.setdefault(session_id, {})[cache_key] = name
                        return name
                pagination = body.get("pagination") if isinstance(body, dict) else None
                total_pages = pagination.get("total_pages") if isinstance(pagination, dict) else None
                if isinstance(total_pages, int) and page >= total_pages:
                    break
    except httpx.HTTPError as exc:
        logger.warning("Workspace authorization lookup failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Could not verify the requested workspace") from exc

    raise HTTPException(status_code=404, detail="workspace not found or not accessible")


def _workspace_path_segment(workspace: str) -> str:
    if WORKSPACE_NAME_RE.fullmatch(workspace) is None:
        raise ValueError("Invalid workspace name")
    return quote(workspace, safe="")


def _studio_assistant_url(workspace: str) -> str:
    # ``workspace`` is expected to already be an Entity-Store-confirmed name (see
    # _authorized_workspace); it is percent-encoded as a single path segment here.
    return (
        f"{_studio_assistant_base_url()}/apis/agents/v2/workspaces/{_workspace_path_segment(workspace)}"
        f"/agents/{quote(_studio_assistant_name(), safe='')}/-/v1/chat/completions"
    )


def _assistant_request_headers(request: Request, agent_url: str) -> dict[str, str]:
    """Forward end-user auth context only over an encrypted connection."""
    if urlparse(agent_url).scheme != "https":
        return {}

    forwarded = {}
    for name in ("authorization", "cookie"):
        value = request.headers.get(name)
        if value:
            forwarded[name] = value
    return forwarded


def _assistant_response(body: Any) -> tuple[str, str]:
    if not isinstance(body, dict):
        raise RuntimeError("NeMo Assistant returned a non-object response")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("NeMo Assistant response did not include a choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("NeMo Assistant response did not include assistant text")
    model = _trimmed_string(body.get("model")) or _studio_assistant_name()
    return message["content"], model


def _assistant_error_detail(exc: httpx.HTTPError | RuntimeError | ValueError) -> str:
    """Return a safe client-facing error without leaking exception or upstream response details."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"The deployed NeMo Assistant returned HTTP {exc.response.status_code}."
    if isinstance(exc, httpx.ReadTimeout):
        return (
            "NeMo Assistant timed out before completing. The operation may still finish; "
            "check History and Intake traces before retrying."
        )
    if isinstance(exc, httpx.HTTPError):
        return "The deployed NeMo Assistant could not be reached."
    return "The deployed NeMo Assistant returned an invalid response."


def _assistant_request_payload(
    messages: list[dict[str, str]],
    studio_session_id: str,
) -> dict[str, Any]:
    return {
        "messages": messages,
        # Stream so the agent runs its streaming path and emits NAT
        # ``intermediate_data:`` tool steps we relay to the chat as tool-use parts.
        "stream": True,
        "studio_session_id": studio_session_id,
    }


_TOOL_STEP_PREFIX = "Tool: "


def _parse_tool_step_input(payload: Any) -> dict[str, Any]:
    """Best-effort extract the tool input dict from a NAT step markdown payload.

    Payloads look like ``**Input:**\\n```json\\n{'resource': 'secrets'}...``; the
    dict is a Python repr (single quotes), so parse the first balanced ``{...}``
    with ``ast.literal_eval`` and fall back to an empty dict.
    """
    if not isinstance(payload, str):
        return {}
    start = payload.find("{")
    if start == -1:
        return {}
    depth = 0
    for index in range(start, len(payload)):
        if payload[index] == "{":
            depth += 1
        elif payload[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(payload[start : index + 1])
                except (ValueError, SyntaxError):
                    return {}
                return value if isinstance(value, dict) else {}
    return {}


# Framework-injected tool arguments that must never be surfaced in the browser
# tool-use event (they are internal plumbing, not user-facing input).
_TOOL_INPUT_INTERNAL_KEYS = frozenset({"studio_session_id"})


_REASONING_STEP_PREFIX = "Reasoning: "


def _parse_reasoning_step_output(payload: Any) -> str:
    """Pull the chain of thought out of a NAT step payload.

    NAT renders steps as markdown with an ``**Input:**`` block and, once the step
    completes, an ``**Output:**`` block holding the reasoning.
    """
    if not isinstance(payload, str):
        return ""
    marker = "**Output:**"
    index = payload.find(marker)
    if index == -1:
        return ""
    return payload[index + len(marker) :].strip()


def _reasoning_stream_event(reasoning: str) -> tuple[str, str]:
    return (
        "agent",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "id": f"nemo-assistant-reasoning-{uuid.uuid4()}",
                    "model": _studio_assistant_name(),
                    "content": [{"type": "reasoning", "text": reasoning}],
                },
            }
        ),
    )


def _parse_tool_step_input(payload: Any) -> dict[str, Any]:
    """Best-effort extract the tool input dict from a NAT step markdown payload.

    Payloads look like ``**Input:**\\n```json\\n{'resource': 'secrets'}...``; the
    dict is a Python repr (single quotes), so parse the first balanced ``{...}``
    with ``ast.literal_eval`` and fall back to an empty dict.
    """
    if not isinstance(payload, str):
        return {}
    start = payload.find("{")
    if start == -1:
        return {}
    depth = 0
    for index in range(start, len(payload)):
        if payload[index] == "{":
            depth += 1
        elif payload[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(payload[start : index + 1])
                except (ValueError, SyntaxError):
                    return {}
                return value if isinstance(value, dict) else {}
    return {}


# Framework-injected tool arguments that must never be surfaced in the browser
# tool-use event (they are internal plumbing, not user-facing input).
_TOOL_INPUT_INTERNAL_KEYS = frozenset({"studio_session_id"})


def _tool_use_stream_event(tool_name: str, tool_input: dict[str, Any]) -> tuple[str, str]:
    safe_input = {key: value for key, value in tool_input.items() if key not in _TOOL_INPUT_INTERNAL_KEYS}
    return (
        "agent",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "id": f"nemo-assistant-tool-{uuid.uuid4()}",
                    "model": _studio_assistant_name(),
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"tool-{uuid.uuid4()}",
                            "name": tool_name,
                            "input": safe_input,
                        }
                    ],
                },
            }
        ),
    )


async def _invoke_assistant(
    agent_url: str,
    headers: Mapping[str, str],
    messages: list[dict[str, str]],
    studio_session_id: str,
) -> tuple[str, str]:
    timeout = httpx.Timeout(
        connect=10.0,
        read=STUDIO_ASSISTANT_TIMEOUT_SECONDS,
        write=60.0,
        pool=10.0,
    )
    queue = _session_streams.get(studio_session_id)
    content_parts: list[str] = []
    model = _studio_assistant_name()
    seen_tool_ids: set[str] = set()
    seen_reasoning_ids: set[str] = set()
    async with httpx.AsyncClient(timeout=timeout) as client:
        # The origin and agent name are server-configured; the workspace path segment is
        # an Entity-Store-confirmed name resolved by _authorized_workspace before this call.
        async with client.stream(
            "POST",
            agent_url,
            headers=dict(headers),
            json=_assistant_request_payload(messages, studio_session_id),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("model"):
                        model = chunk["model"]
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if isinstance(piece, str) and piece:
                            content_parts.append(piece)
                elif line.startswith("intermediate_data:") and queue is not None:
                    data = line[len("intermediate_data:") :].strip()
                    try:
                        step = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    name = step.get("name") if isinstance(step, dict) else None
                    if not isinstance(name, str):
                        continue
                    if name.startswith(_REASONING_STEP_PREFIX):
                        # Published as a start/end pair sharing one id, so this cannot use
                        # the tool dedup below: the start would claim the id and the end,
                        # which carries the trace, would be dropped. Only the end has an
                        # Output block, so record the id only once one is parsed -- that
                        # skips a repeated end without ever marking the start as seen.
                        reasoning = _parse_reasoning_step_output(step.get("payload"))
                        if not reasoning:
                            continue
                        step_id = step.get("id")
                        if isinstance(step_id, str):
                            if step_id in seen_reasoning_ids:
                                continue
                            seen_reasoning_ids.add(step_id)
                        await queue.put(_reasoning_stream_event(reasoning))
                        continue
                    if not name.startswith(_TOOL_STEP_PREFIX):
                        continue
                    step_id = step.get("id")
                    if isinstance(step_id, str):
                        if step_id in seen_tool_ids:
                            continue
                        seen_tool_ids.add(step_id)
                    await queue.put(
                        _tool_use_stream_event(
                            name[len(_TOOL_STEP_PREFIX) :].strip(),
                            _parse_tool_step_input(step.get("payload")),
                        )
                    )
    return "".join(content_parts), model


def _render_session_event(event_type: str, payload: Any) -> str | None:
    """Render a queued session event as an SSE frame, or None if it is not relayable."""
    if event_type == "permission_request":
        return _sse(payload, event="permission_request")
    if event_type == "input_request":
        return _sse(payload, event="input_request")
    if event_type == "permission_expired":
        return _sse(payload, event="permission_expired")
    if event_type == "input_expired":
        return _sse(payload, event="input_expired")
    if event_type == "agent":
        return _sse(payload)
    return None


async def _stream_assistant(
    session_id: str,
    message: str,
    agent_url: str,
    headers: Mapping[str, str],
    studio_system_prompt: str,
    conversation: AssistantConversation,
    entity_store: EntityClient,
) -> AsyncIterator[str]:
    """Invoke the deployed NeMo Assistant while preserving Studio's blocking UI event protocol."""
    if session_id in _session_streams:
        yield _sse(
            json.dumps({"message": "session already has an active stream"}),
            event="error",
        )
        return

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    _session_streams[session_id] = queue
    deployment_run_id = str(uuid.uuid4())
    contextual_message = "\n\n".join(
        [
            "<nemo_studio_context>",
            studio_system_prompt,
            (
                f"For any deploy_guardrail call for this request, pass deployment_run_id='{deployment_run_id}'. "
                "Use it exactly once and do not invent or substitute another value."
            ),
            "</nemo_studio_context>",
            "User request:",
            message,
        ]
    )
    request_messages = [
        *(persisted_message.model_dump() for persisted_message in _recent_conversation_messages(conversation.messages)),
        {"role": "user", "content": contextual_message},
    ]
    invocation = asyncio.create_task(
        _invoke_assistant(
            agent_url,
            headers,
            request_messages,
            studio_session_id=session_id,
        )
    )
    queued_event = asyncio.create_task(queue.get())

    try:
        while not invocation.done():
            done, _ = await asyncio.wait(
                {invocation, queued_event},
                timeout=MCP_KEEPALIVE_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                yield ":\n\n"
                continue
            if queued_event in done:
                rendered = _render_session_event(*queued_event.result())
                if rendered is not None:
                    yield rendered
                queued_event = asyncio.create_task(queue.get())

        # The invocation finished, but tool-use / prompt events may have been
        # queued in the same event-loop turn as completion. Flush them before the
        # final assistant message so they are not dropped when the loop exits.
        pending: list[tuple[str, Any]] = []
        if queued_event.done() and not queued_event.cancelled():
            pending.append(queued_event.result())
        else:
            queued_event.cancel()
        while True:
            try:
                pending.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for event_type, payload in pending:
            rendered = _render_session_event(event_type, payload)
            if rendered is not None:
                yield rendered

        assistant_text, model = await invocation
        _append_conversation_turn(conversation, message, assistant_text, model)
        try:
            await entity_store.update(conversation)
        except EntityConflictError:
            logger.info("Reloading conflicted NeMo Assistant session %s before retrying", session_id)
            latest_conversation = await entity_store.get(
                AssistantConversation,
                conversation.name,
                workspace=conversation.workspace,
            )
            _append_conversation_turn(latest_conversation, message, assistant_text, model)
            await entity_store.update(latest_conversation)
        yield _sse(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "id": f"nemo-studio-assistant-{uuid.uuid4()}",
                        "model": model,
                        "content": [{"type": "text", "text": assistant_text}],
                    },
                }
            )
        )
        yield _sse("", event="done")
    except asyncio.CancelledError:
        invocation.cancel()
        raise
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        logger.exception("NeMo Assistant invocation failed for session %s", session_id)
        yield _sse(
            json.dumps({"message": _assistant_error_detail(exc)}),
            event="error",
        )
    except EntityStoreError:
        logger.exception("Failed to persist NeMo Assistant session %s", session_id)
        yield _sse(
            json.dumps({"message": "NeMo Assistant could not save this conversation."}),
            event="error",
        )
    finally:
        _session_streams.pop(session_id, None)
        if not invocation.done():
            invocation.cancel()
        if not queued_event.done():
            queued_event.cancel()


@router.post("/sessions/{session_id}/messages")
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def send_message(
    session_id: str,
    body: MessageRequest,
    request: Request,
    entity_store: EntityClient = Depends(get_entity_client),
) -> StreamingResponse:
    """Send a message to the deployed NeMo Assistant and stream Studio events."""
    sid = _validate_session_id(session_id)
    workspace = _validated_workspace_or_default(body.workspace)
    # Headers are forwarded only over HTTPS, and the scheme comes from the
    # server-configured base URL, so deriving them before the workspace is
    # confirmed is equivalent -- and the authorization lookup needs them.
    agent_headers = _assistant_request_headers(request, _studio_assistant_url(workspace))
    canonical_workspace = await _authorized_workspace(workspace, agent_headers, sid)
    conversation = await _get_owned_conversation(
        entity_store,
        session_id=sid,
        workspace=canonical_workspace,
        owner_id=_request_principal_id(request),
    )
    agent_url = _studio_assistant_url(canonical_workspace)
    studio_base_url = _studio_base_url_from_request(body, request)
    studio_pathname = _studio_pathname_from_request(body, request)
    enabled_destinations = studio_links.enabled_destinations_from_request(request)
    system_prompt = _build_assistant_system_prompt(
        sid,
        canonical_workspace,
        studio_base_url,
        studio_pathname,
        enabled_destinations,
    )
    return StreamingResponse(
        _stream_assistant(
            sid,
            body.message,
            agent_url,
            agent_headers,
            system_prompt,
            conversation,
            entity_store,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/sessions/{session_id}/permissions/{request_id}")
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def resolve_permission(session_id: str, request_id: str, body: PermissionDecision) -> dict[str, bool]:
    """Resolve a pending Claude tool permission request."""
    sid = _validate_session_id(session_id)
    pending = _pending_permissions.get(request_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="no such pending permission")
    pending_session_id, future = pending
    if pending_session_id != sid or future.done():
        raise HTTPException(status_code=404, detail="no such pending permission")
    future.set_result(body.model_dump())
    return {"ok": True}


@router.post("/sessions/{session_id}/inputs/{request_id}")
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def resolve_agent_input(session_id: str, request_id: str, body: AgentInputDecision) -> dict[str, bool]:
    """Resolve a pending Claude UI input request."""
    sid = _validate_session_id(session_id)
    pending = _pending_agent_inputs.get(request_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="no such pending input")
    pending_session_id, future = pending
    if pending_session_id != sid or future.done():
        raise HTTPException(status_code=404, detail="no such pending input")
    future.set_result(body.model_dump())
    return {"ok": True}


@router.post("/mcp/{session_id}", name=MCP_ROUTE_NAME, include_in_schema=False)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def mcp_endpoint(session_id: str, request: Request) -> Response:
    """Minimal MCP endpoint used by Claude's permission-prompt tool."""
    sid = _validate_session_id(session_id)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"detail": "JSON body must be an object"})

    request_id = body.get("id")

    if request_id is None:
        return Response(status_code=202)

    method = body.get("method")
    raw_params = body.get("params")
    if raw_params is not None and not isinstance(raw_params, dict):
        return JSONResponse(status_code=400, content={"detail": "JSON-RPC params must be an object"})
    params = body.get("params") or {}

    if method == "initialize":
        client_protocol = params.get("protocolVersion", "2025-06-18")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": client_protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "nemo-studio-permissions", "version": "0.1.0"},
                },
            }
        )

    if method == "tools/list":
        enabled_destinations = studio_links.enabled_destinations_from_request(request)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": _mcp_tools_for_destinations(enabled_destinations)},
            }
        )

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return JSONResponse(status_code=400, content={"detail": "tool arguments must be an object"})

        if name == SELECT_AGENT_TOOL_NAME:
            return await _blocking_mcp_tool_response(
                sid,
                request_id,
                _request_agent_input(sid, "agent", args),
            )

        if name == SELECT_EVAL_CONFIG_TOOL_NAME:
            return await _blocking_mcp_tool_response(
                sid,
                request_id,
                _request_agent_input(sid, "eval_config", args),
            )

        if name == SELECT_DATASET_FILE_TOOL_NAME:
            return await _blocking_mcp_tool_response(
                sid,
                request_id,
                _request_agent_input(sid, "dataset_file", args),
            )

        if name == SELECT_MODEL_TOOL_NAME:
            return await _blocking_mcp_tool_response(
                sid,
                request_id,
                _request_agent_input(sid, "model", args),
            )

        if name == JOB_PROGRESS_TOOL_NAME:
            queue = _session_streams.get(sid)
            if queue is not None:
                await queue.put(
                    (
                        "agent",
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "id": f"nemo-studio-assistant-tool-{uuid.uuid4()}",
                                    "model": _studio_assistant_name(),
                                    "content": [
                                        {
                                            "type": "tool_use",
                                            "id": f"tool-{uuid.uuid4()}",
                                            "name": (f"mcp__{CLAUDE_MCP_SERVER_NAME}__{JOB_PROGRESS_TOOL_NAME}"),
                                            "input": args,
                                        }
                                    ],
                                },
                            }
                        ),
                    )
                )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"status": "rendered"})}],
                    },
                }
            )

        if name == APPROVAL_TOOL_NAME:
            return await _blocking_mcp_tool_response(
                sid,
                request_id,
                _request_permission(sid, args),
            )

        if name == STUDIO_LINK_TOOL_NAME:
            workspace = _trimmed_string(request.query_params.get("workspace"))
            studio_base_url = _trimmed_string(request.query_params.get("studio_base_url"))
            enabled_destinations = studio_links.enabled_destinations_from_request(request)
            result = studio_links.build_studio_link_result(workspace, studio_base_url, args, enabled_destinations)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result)}],
                    },
                }
            )

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown tool: {name}"},
            }
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    )


@router.api_route("/mcp/{session_id}", methods=["GET", "DELETE"], include_in_schema=False)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def mcp_unsupported_method(session_id: str) -> Response:
    """Decline optional MCP SSE/session methods without falling through to Studio HTML."""
    _validate_session_id(session_id)
    return Response(status_code=405, headers={"Allow": "POST"})
