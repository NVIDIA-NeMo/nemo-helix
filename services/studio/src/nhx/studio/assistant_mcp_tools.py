# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool catalog for Studio assistant sessions."""

from typing import Any

APPROVAL_TOOL_NAME = "approval_prompt"
STUDIO_LINK_TOOL_NAME = "studio_link"
SELECT_AGENT_TOOL_NAME = "select_agent"
SELECT_EVAL_CONFIG_TOOL_NAME = "select_eval_config"
SELECT_DATASET_FILE_TOOL_NAME = "select_dataset_file"
SELECT_MODEL_TOOL_NAME = "select_model"
JOB_PROGRESS_TOOL_NAME = "job_progress"
ASSISTANT_ACTIVITY_TOOL_NAME = "assistant_activity"

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

STUDIO_ASSISTANT_CONTEXT = "\n".join(
    [
        (
            "NeMo Studio and the NeMo Helix API are already set up and running for this workspace. "
            "Treat the platform as healthy and go straight to the user's question."
        ),
        (
            "Set UI tool titles, descriptions, display labels, and output_key values to match the "
            "current workflow while keeping the tools reusable."
        ),
    ]
)
