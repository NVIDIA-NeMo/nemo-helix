# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool catalog for Studio assistant sessions."""

from typing import Any

CLAUDE_MCP_SERVER_NAME = "nemo_studio"

APPROVAL_TOOL_NAME = "approval_prompt"
STUDIO_LINK_TOOL_NAME = "studio_link"
SELECT_AGENT_TOOL_NAME = "select_agent"
SELECT_EVAL_CONFIG_TOOL_NAME = "select_eval_config"
SELECT_DATASET_FILE_TOOL_NAME = "select_dataset_file"
SELECT_MODEL_TOOL_NAME = "select_model"
JOB_PROGRESS_TOOL_NAME = "job_progress"

APPROVAL_TOOL: dict[str, Any] = {
    "name": APPROVAL_TOOL_NAME,
    "description": "Ask the human operator whether a tool call should be allowed.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "input": {"type": "object"},
            "tool_use_id": {"type": "string"},
        },
        "required": ["tool_name", "input"],
    },
}

SELECT_AGENT_TOOL: dict[str, Any] = {
    "name": SELECT_AGENT_TOOL_NAME,
    "description": (
        "Ask the Studio user to choose an agent from a visual dropdown. "
        "This is the required way to get an agent name in Studio: call it whenever a workflow needs the "
        "user to name, pick, confirm, or disambiguate an agent (including choosing among deployed agents). "
        "Always prefer this over asking the user to type an agent name. "
        "Returns the chosen agent name, or status=skipped / status=error if the user dismisses the dropdown."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "default_agent": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

SELECT_EVAL_CONFIG_TOOL: dict[str, Any] = {
    "name": SELECT_EVAL_CONFIG_TOOL_NAME,
    "description": (
        "Ask the Studio user to choose an evaluation YAML file from a visual fileset file picker. "
        "Use this instead of plain text prompting when an agent evaluation workflow needs an eval config. "
        "Returns eval_config_fileset and eval_config. If the user does not have a config yet, "
        "Studio may return needs_eval_config=true so you can help create one. "
        "Studio may also return use_sample_eval_config=true with a sample eval_config and eval_config_fileset."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "agent": {"type": "string"},
            "default_agent": {"type": "string"},
            "accepted_file_types": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    },
}

SELECT_DATASET_FILE_TOOL: dict[str, Any] = {
    "name": SELECT_DATASET_FILE_TOOL_NAME,
    "description": (
        "Ask the Studio user to choose a dataset file from a visual fileset file picker. "
        "This is the required way to get a fileset, fileset reference, dataset, or input/source data file "
        "in Studio (for example an anonymizer or evaluation input, or a CSV/Parquet file). Call it whenever "
        "a workflow needs the user to pick a fileset or a file inside one, instead of asking for a fileset "
        "reference, path, or '<workspace>/<fileset>#<file>' string in plain text. "
        "Always prefer this over asking the user to type a file or fileset reference. "
        "Pass accepted_file_types (for example ['.csv', '.parquet']) to constrain the picker. "
        "Returns dataset_fileset and dataset_path, which you can combine as '<fileset>#<path>' when a tool "
        "needs a fileset reference. Returns status=skipped / status=error if the user dismisses the picker."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "accepted_file_types": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    },
}

SELECT_MODEL_TOOL: dict[str, Any] = {
    "name": SELECT_MODEL_TOOL_NAME,
    "description": (
        "Ask the Studio user to choose a model from a visual model picker. "
        "Use this instead of plain text prompting when a Studio workflow needs a concrete model name. "
        "Returns model by default. You may pass output_key to request a different response key."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "default_model": {"type": "string"},
            "output_key": {"type": "string"},
            "display_label": {"type": "string"},
            "field_label": {"type": "string"},
            "placeholder": {"type": "string"},
            "required_message": {"type": "string"},
            "submit_label": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

JOB_PROGRESS_TOOL: dict[str, Any] = {
    "name": JOB_PROGRESS_TOOL_NAME,
    "description": (
        "Show a compact Studio progress card for a long-running job. "
        "This is the required way to surface a launched job in Studio: after you start, submit, or kick off "
        "any platform job and know its job_name, you must call this before your final response, once for "
        "every job you launch. Do not substitute a plain-text job summary or a manual status command. "
        "Call it only after a real job has been started and you know its job_name. "
        "Pass job_type or source when known so Studio can link to the best detail page."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "job_name": {"type": "string"},
            "job_type": {
                "type": "string",
                "description": (
                    "Optional job category, such as customization, data_designer, "
                    "safe_synthesizer, evaluator, or agent_evaluation."
                ),
            },
            "source": {
                "type": "string",
                "description": "Optional platform job source, such as customization or data-designer.",
            },
            "title": {"type": "string"},
            "description": {"type": "string"},
            "workspace": {"type": "string"},
        },
        "required": ["job_name"],
        "additionalProperties": False,
    },
}

MCP_TOOLS: list[dict[str, Any]] = [
    APPROVAL_TOOL,
    SELECT_AGENT_TOOL,
    SELECT_EVAL_CONFIG_TOOL,
    SELECT_DATASET_FILE_TOOL,
    SELECT_MODEL_TOOL,
    JOB_PROGRESS_TOOL,
]

STUDIO_UI_TOOL_NAMES = (
    SELECT_AGENT_TOOL_NAME,
    SELECT_EVAL_CONFIG_TOOL_NAME,
    SELECT_DATASET_FILE_TOOL_NAME,
    SELECT_MODEL_TOOL_NAME,
    JOB_PROGRESS_TOOL_NAME,
    STUDIO_LINK_TOOL_NAME,
)

STUDIO_ASSISTANT_CONTEXT = "\n".join(
    [
        "You are NeMo Assistant, running inside NeMo Studio.",
        (
            "NeMo Studio and the NeMo Helix API are already installed, set up, and running for this "
            "workspace. Treat the platform as healthy and available."
        ),
        (
            "Do not run setup, bootstrap, install, start, or health-check steps for the platform: skip "
            "'nemo setup', 'make bootstrap', 'nemo services run', service restarts, port probes, and "
            "'health/ready' checks unless the user explicitly asks you to. Assume services are up and "
            "go straight to the actual task."
        ),
        (
            "Prefer NeMo Studio MCP tools and Studio views over CLI commands for user-facing "
            "follow-up actions, navigation, inspection, and status/result review."
        ),
        (
            "Do not tell the user to run nemo CLI commands, shell commands, curl commands, or status "
            "commands to inspect agents, jobs, evaluations, filesets, models, traces, logs, or results "
            "when a Studio view, Studio link, or Studio progress card is available for the same purpose."
        ),
        (
            "Use CLI commands only to perform work that has no Studio UI equivalent, when the user "
            "explicitly asks for CLI/debugging, or when you must gather data that Studio tools cannot "
            "provide. For user-facing follow-up, prefer mcp__nemo_studio__studio_link and "
            "mcp__nemo_studio__job_progress."
        ),
        (
            "When you need to prompt the user for input, use a Studio UI tool instead of writing a "
            "plain-text question whenever a suitable tool exists."
        ),
        (
            "These needs are mandatory tool calls, not plain-text questions: "
            "use mcp__nemo_studio__select_agent whenever you need the user to name, pick, confirm, or "
            "disambiguate an agent (including among deployed agents); "
            "mcp__nemo_studio__select_model for model names; "
            "mcp__nemo_studio__select_dataset_file whenever you need a fileset, fileset reference, dataset, "
            "or input/source data file (for example an anonymizer or evaluation input, or a CSV/Parquet "
            "file) instead of asking for a fileset reference or '<workspace>/<fileset>#<file>' path in text; "
            "and mcp__nemo_studio__select_eval_config for evaluation config files."
        ),
        (
            "Never use a plain-text question to choose an agent, model, fileset, "
            "dataset or input file, or eval config; those each have a dedicated select_* tool you must "
            "call instead."
        ),
        (
            "For any question with a finite set of choices that does NOT map to a select_* tool — including "
            "yes/no, multiple-choice, 'pick one of these', or whenever you would offer the user options to "
            "choose from — you MUST call AskUserQuestion to render a selectable options picker instead of "
            "listing the choices in plain text. Only ask a concise plain-text question for genuinely "
            "open-ended, free-form input that has no discrete options."
        ),
        (
            "Only fall back to plain chat questions when no suitable UI tool exists, the user already "
            "provided the value, or the user explicitly skips the UI tool. A timeout, disconnect, or "
            "other UI-tool error is not permission to continue or repeat the question in plain text; "
            "leave the input unresolved and tell the user the interactive request must be retried."
        ),
        (
            "Set UI tool titles, descriptions, display labels, and output_key values to match the "
            "current workflow while keeping the tools reusable."
        ),
        (
            "Whenever you start, submit, or kick off any platform job (for example an anonymizer run, "
            "customization, data designer, safe synthesizer, or evaluation job) and you know its job name, "
            "you MUST call mcp__nemo_studio__job_progress with that job name before your final response so "
            "Studio renders the progress card inline. Do this for every job you launch, not only the first, "
            "and pass job_type or source when known so the card links to the right detail page."
        ),
        (
            "Never replace the job_progress card with a plain-text job summary or by telling the user to "
            "run a status command; call job_progress in addition to any Studio link you include."
        ),
    ]
)


def qualified_mcp_tool_name(tool_name: str, server_name: str = CLAUDE_MCP_SERVER_NAME) -> str:
    """Return Claude Code's fully qualified MCP tool name for a Studio tool."""
    return f"mcp__{server_name}__{tool_name}"


def allowed_mcp_tools(server_name: str = CLAUDE_MCP_SERVER_NAME) -> list[str]:
    """Return the Studio UI tools Claude may call without an approval prompt."""
    return [qualified_mcp_tool_name(tool_name, server_name) for tool_name in STUDIO_UI_TOOL_NAMES]


def permission_prompt_tool(server_name: str = CLAUDE_MCP_SERVER_NAME) -> str:
    """Return the fully qualified approval prompt tool name."""
    return qualified_mcp_tool_name(APPROVAL_TOOL_NAME, server_name)
