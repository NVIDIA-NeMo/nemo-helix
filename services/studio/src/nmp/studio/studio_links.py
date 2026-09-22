# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio link destinations and MCP tool helpers for the assistant bridge."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from fastapi import Request
from nmp.studio.config import StudioConfig
from nmp.studio.env_mappings import ENV_MAPPINGS


@dataclass(frozen=True)
class StudioLinkDestination:
    """Known Studio destination that Claude Code can link to."""

    label: str
    path_template: str
    aliases: tuple[str, ...] = ()
    requires_name: bool = False
    required_args: tuple[str, ...] = ()


STUDIO_LINK_DESTINATIONS: dict[str, StudioLinkDestination] = {
    "workspace": StudioLinkDestination(
        "Workspace",
        "/workspaces/{workspace}",
        aliases=("workspace_home", "workspace_index"),
    ),
    "dashboard": StudioLinkDestination("Workspace dashboard", "/workspaces/{workspace}/dashboard"),
    "assistant": StudioLinkDestination(
        "NeMo Assistant",
        "/workspaces/{workspace}/dashboard/assistant",
        aliases=("claude_code", "claude_code_chat", "assistant_chat"),
    ),
    "agents": StudioLinkDestination(
        "Agents",
        "/workspaces/{workspace}/agents",
        aliases=("agent_list", "agents_page"),
    ),
    "agent": StudioLinkDestination(
        "Agent {name}",
        "/workspaces/{workspace}/agents/{name}",
        aliases=("agent_detail",),
        requires_name=True,
    ),
    "agent_chat": StudioLinkDestination(
        "Chat with agent {name}",
        "/workspaces/{workspace}/agents/{name}?tab=chat-playground",
        aliases=("agent_playground", "agent_chat_playground", "chat_with_agent"),
        requires_name=True,
    ),
    "agent_deployments": StudioLinkDestination(
        "Agent deployments",
        "/workspaces/{workspace}/agents",
        aliases=("agent_deployment_list", "agent_deployments_page"),
    ),
    "agent_deployment": StudioLinkDestination(
        "Agent deployment {name}",
        "/workspaces/{workspace}/agents/{name}",
        aliases=("agent_deployment_detail",),
        requires_name=True,
    ),
    "agent_evaluations": StudioLinkDestination(
        "Agent evaluations",
        "/workspaces/{workspace}/agents/evaluations",
        aliases=("agent_evaluation_list",),
    ),
    "agent_evaluation": StudioLinkDestination(
        "Agent evaluation {name}",
        "/workspaces/{workspace}/agents/evaluations/{name}",
        aliases=("agent_evaluation_detail",),
        requires_name=True,
    ),
    "agent_optimizations": StudioLinkDestination(
        "Agent optimizations",
        "/workspaces/{workspace}/agents/suggestions",
        aliases=("agent_suggestions", "agent_suggestions_list"),
    ),
    "base_models": StudioLinkDestination(
        "Base Models",
        "/workspaces/{workspace}/base-models",
        aliases=("base_model_list", "base_models_page", "available_models", "available_base_models"),
    ),
    "base_model": StudioLinkDestination(
        "Base model {name}",
        "/workspaces/{workspace}/base-models/{name}",
        aliases=("base_model_detail", "available_base_model"),
        requires_name=True,
    ),
    "base_model_chat": StudioLinkDestination(
        "Chat with base model {name}",
        "/workspaces/{workspace}/base-models/{name}?tab=chat-playground",
        aliases=("base_model_playground", "base_model_chat_playground", "chat_with_base_model"),
        requires_name=True,
    ),
    "evaluation": StudioLinkDestination(
        "Evaluation",
        "/workspaces/{workspace}/evaluation",
        aliases=("evaluator", "evaluations"),
    ),
    "evaluation_metrics": StudioLinkDestination(
        "Evaluation metrics",
        "/workspaces/{workspace}/evaluation/metrics",
        aliases=("metrics", "evaluator_metrics"),
    ),
    "evaluation_metric_new": StudioLinkDestination(
        "Create evaluation metric",
        "/workspaces/{workspace}/evaluation/metrics/new",
        aliases=("new_evaluation_metric", "create_evaluation_metric"),
    ),
    "evaluation_run": StudioLinkDestination(
        "Run evaluation",
        "/workspaces/{workspace}/evaluation/metrics/run",
        aliases=("run_evaluation", "start_evaluation"),
    ),
    "evaluation_metric": StudioLinkDestination(
        "Evaluation metric {name}",
        "/workspaces/{workspace}/evaluation/metrics/{name}",
        aliases=("evaluation_metric_detail", "metric", "metric_detail"),
        requires_name=True,
    ),
    "evaluation_metric_run": StudioLinkDestination(
        "Run evaluation metric {name}",
        "/workspaces/{workspace}/evaluation/metrics/{name}/run",
        aliases=("run_evaluation_metric", "metric_run"),
        requires_name=True,
    ),
    "evaluation_benchmarks": StudioLinkDestination(
        "Evaluation benchmarks",
        "/workspaces/{workspace}/evaluation/benchmarks",
        aliases=("benchmarks", "evaluator_benchmarks"),
    ),
    "evaluation_benchmark": StudioLinkDestination(
        "Evaluation benchmark {name}",
        "/workspaces/{workspace}/evaluation/benchmarks/{name}",
        aliases=("benchmark", "benchmark_detail", "evaluation_benchmark_detail"),
        requires_name=True,
    ),
    "evaluation_results": StudioLinkDestination(
        "Evaluation results",
        "/workspaces/{workspace}/evaluation/results",
        aliases=("eval_results", "evaluator_results"),
    ),
    "evaluation_result": StudioLinkDestination(
        "Evaluation result {name}",
        "/workspaces/{workspace}/evaluation/results/{name}",
        aliases=("eval_result", "evaluation_result_detail", "evaluator_result"),
        requires_name=True,
    ),
    "customizations": StudioLinkDestination(
        "Custom Models",
        "/workspaces/{workspace}/customizations",
        aliases=("custom_models", "custom_models_page", "customization_jobs", "customizations_page"),
    ),
    "customization_new": StudioLinkDestination(
        "Create custom model",
        "/workspaces/{workspace}/customizations/fine-tuned/new",
        aliases=("new_customization", "create_custom_model", "fine_tune", "fine_tuned_new"),
    ),
    "customization": StudioLinkDestination(
        "Custom model {name}",
        "/workspaces/{workspace}/customizations/{name}",
        aliases=("custom_model", "customization_job", "customization_detail"),
        requires_name=True,
    ),
    "prompt_tuning": StudioLinkDestination(
        "Prompt tuning",
        "/workspaces/{workspace}/customizations/prompt-tuned/new",
        aliases=("prompt_tuning_new", "prompt_tuned_new", "prompt_tuned_customization"),
    ),
    "model_chat": StudioLinkDestination(
        "Chat with models",
        "/workspaces/{workspace}/model-compare",
        aliases=("chat", "model_compare", "model_chat_page", "model_playground"),
    ),
    "jobs": StudioLinkDestination("Jobs", "/workspaces/{workspace}/jobs", aliases=("job_list",)),
    "job": StudioLinkDestination(
        "Job {name}",
        "/workspaces/{workspace}/jobs/{name}",
        aliases=("job_detail",),
        requires_name=True,
    ),
    "filesets": StudioLinkDestination("Filesets", "/workspaces/{workspace}/filesets"),
    "fileset_new": StudioLinkDestination(
        "Create fileset",
        "/workspaces/{workspace}/filesets/new",
        aliases=("new_fileset", "create_fileset", "new_dataset", "create_dataset"),
    ),
    "fileset_panel": StudioLinkDestination(
        "Fileset {name}",
        "/workspaces/{workspace}/filesets/{name}",
        aliases=("fileset_side_panel", "dataset_panel"),
        requires_name=True,
    ),
    "fileset": StudioLinkDestination(
        "Fileset {name}",
        "/workspaces/{workspace}/filesets/{name}/detail",
        aliases=("fileset_detail", "fileset_detail_page", "dataset", "dataset_detail"),
        requires_name=True,
    ),
    "fileset_file": StudioLinkDestination(
        "File {file_path}",
        "/workspaces/{workspace}/filesets/{name}/file/{file_path}",
        aliases=("dataset_file", "fileset_file_detail"),
        required_args=("name", "file_path"),
    ),
    "deployments": StudioLinkDestination("Deployments", "/workspaces/{workspace}/deployments"),
    "deployment": StudioLinkDestination(
        "Deployment {name}",
        "/workspaces/{workspace}/deployments/{name}/details",
        aliases=("deployment_detail",),
        requires_name=True,
    ),
    "inference_providers": StudioLinkDestination(
        "Inference providers",
        "/workspaces/{workspace}/inference-providers",
        aliases=("model_providers", "providers"),
    ),
    "guardrails": StudioLinkDestination("Guardrails", "/workspaces/{workspace}/guardrails"),
    "virtual_models": StudioLinkDestination(
        "Virtual Models",
        "/workspaces/{workspace}/virtual-models",
        aliases=("virtual_model_list", "virtual_models_page"),
    ),
    "virtual_model_chat": StudioLinkDestination(
        "Chat with VirtualModel {name}",
        "/workspaces/{workspace}/virtual-models?virtualModel={name}&tab=chat",
        aliases=("chat_with_virtual_model", "virtual_model_playground"),
        requires_name=True,
    ),
    "secrets": StudioLinkDestination("Secrets", "/workspaces/{workspace}/secrets"),
    "intake": StudioLinkDestination("Intake", "/workspaces/{workspace}/intake"),
    "intake_traces": StudioLinkDestination(
        "Intake traces",
        "/workspaces/{workspace}/intake/traces",
        aliases=("traces", "trace_list", "intake_trace_list"),
    ),
    "intake_spans": StudioLinkDestination(
        "Intake spans",
        "/workspaces/{workspace}/intake/spans",
        aliases=("spans", "span_list", "intake_span_list"),
    ),
    "intake_session": StudioLinkDestination(
        "Session {session_id}",
        "/workspaces/{workspace}/intake/sessions/{session_id}",
        aliases=("session", "session_detail", "intake_session_detail"),
        required_args=("session_id",),
    ),
    "intake_trace": StudioLinkDestination(
        "Trace {trace_id}",
        "/workspaces/{workspace}/intake/sessions/{session_id}?traceId={trace_id}",
        aliases=("trace", "trace_detail"),
        required_args=("session_id", "trace_id"),
    ),
    "intake_span": StudioLinkDestination(
        "Span {span_id}",
        "/workspaces/{workspace}/intake/sessions/{session_id}?traceId={trace_id}&spanId={span_id}",
        aliases=("span", "span_detail"),
        required_args=("session_id", "trace_id", "span_id"),
    ),
    "data_designer": StudioLinkDestination(
        "Data Designer",
        "/workspaces/{workspace}/data-designer",
        aliases=("data_designer_jobs",),
    ),
    "data_designer_new": StudioLinkDestination(
        "Create Data Designer job",
        "/workspaces/{workspace}/data-designer/new",
        aliases=("new_data_designer_job", "create_data_designer_job"),
    ),
    "data_designer_job": StudioLinkDestination(
        "Data Designer job {name}",
        "/workspaces/{workspace}/data-designer/{name}",
        aliases=("data_designer_job_detail",),
        requires_name=True,
    ),
    "safe_synthesizer": StudioLinkDestination(
        "Safe Synthesizer",
        "/workspaces/{workspace}/safe-synthesizer",
        aliases=("safe_synthesizer_jobs",),
    ),
    "safe_synthesizer_new": StudioLinkDestination(
        "Create Safe Synthesizer job",
        "/workspaces/{workspace}/safe-synthesizer/new",
        aliases=("new_safe_synthesizer_job", "create_safe_synthesizer_job"),
    ),
    "safe_synthesizer_job": StudioLinkDestination(
        "Safe Synthesizer job {name}",
        "/workspaces/{workspace}/safe-synthesizer/job/{name}",
        aliases=("safe_synthesizer_job_detail",),
        requires_name=True,
    ),
    "safe_synthesizer_report": StudioLinkDestination(
        "Safe Synthesizer report {name}",
        "/workspaces/{workspace}/safe-synthesizer/job/{name}/report",
        aliases=("safe_synthesizer_job_report", "safe_synthesizer_report_detail"),
        requires_name=True,
    ),
    "settings": StudioLinkDestination(
        "Workspace settings",
        "/workspaces/{workspace}/settings",
        aliases=("workspace_settings",),
    ),
    "members": StudioLinkDestination(
        "Workspace members",
        "/workspaces/{workspace}/members",
        aliases=("workspace_members", "member_list"),
    ),
    "experiment": StudioLinkDestination("Experiment", "/workspaces/{workspace}/experiment"),
    "experiment_group": StudioLinkDestination(
        "Experiment group {name}",
        "/workspaces/{workspace}/experiment/{name}",
        aliases=("experiment_group_detail",),
        requires_name=True,
    ),
    "experiment_detail": StudioLinkDestination(
        "Experiment {experiment_name}",
        "/workspaces/{workspace}/experiment/{name}/{experiment_name}",
        aliases=("experiment_run",),
        required_args=("name", "experiment_name"),
    ),
    "plugin": StudioLinkDestination(
        "Plugin {name}",
        "/workspaces/{workspace}/plugin/{name}",
        aliases=("plugin_page", "plugin_ui"),
        requires_name=True,
    ),
}

_STUDIO_LINK_DESTINATION_ALIASES = {
    alias: destination for destination, config in STUDIO_LINK_DESTINATIONS.items() for alias in config.aliases
}

_STUDIO_LINK_ARGUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "name": (
        "resource_name",
        "resourceName",
        "id",
        "job_name",
        "jobName",
        "agent_name",
        "agentName",
        "model_name",
        "modelName",
        "fileset_id",
        "filesetId",
        "fileset_name",
        "filesetName",
        "deployment_name",
        "deploymentName",
        "trace_id",
        "traceId",
        "span_id",
        "spanId",
        "experiment_group_id",
        "experimentGroupId",
    ),
    "experiment_name": ("experimentName", "experiment_id", "experimentId"),
    "file_path": ("file", "filePath", "file_path_encoded", "filePathEncoded", "path"),
    "session_id": ("sessionId",),
    "trace_id": ("traceId",),
    "span_id": ("spanId", "name"),
}

_STUDIO_LINK_DESTINATION_FEATURE_FLAGS: dict[str, tuple[str, ...]] = {
    "agents": ("agents_enabled",),
    "agent": ("agents_enabled",),
    "agent_chat": ("agents_enabled",),
    "agent_deployments": ("agents_enabled",),
    "agent_deployment": ("agents_enabled",),
    "agent_evaluations": ("agents_enabled",),
    "agent_evaluation": ("agents_enabled",),
    "agent_optimizations": ("agents_enabled",),
    "base_models": ("base_models_enabled",),
    "base_model": ("base_models_enabled",),
    "base_model_chat": ("base_models_enabled",),
    "evaluation": ("evaluator_enabled",),
    "evaluation_metrics": ("evaluator_enabled",),
    "evaluation_metric_new": ("evaluator_enabled",),
    "evaluation_run": ("evaluator_enabled",),
    "evaluation_metric": ("evaluator_enabled",),
    "evaluation_metric_run": ("evaluator_enabled",),
    "evaluation_benchmarks": ("evaluator_enabled", "evaluator_benchmarks_enabled"),
    "evaluation_benchmark": ("evaluator_enabled", "evaluator_benchmarks_enabled"),
    "evaluation_results": ("evaluator_enabled",),
    "evaluation_result": ("evaluator_enabled",),
    "customizations": ("customizer_enabled",),
    "customization_new": ("customizer_enabled",),
    "customization": ("customizer_enabled",),
    "prompt_tuning": ("customizer_enabled",),
    "model_chat": ("model_compare_enabled",),
    "jobs": ("jobs_enabled",),
    "job": ("jobs_enabled",),
    "filesets": ("datasets_enabled",),
    "fileset_new": ("datasets_enabled",),
    "fileset_panel": ("datasets_enabled",),
    "fileset": ("fileset_details_enabled",),
    "fileset_file": ("datasets_enabled",),
    "deployments": ("deployments_enabled",),
    "deployment": ("deployments_enabled",),
    "inference_providers": ("inference_provider_enabled",),
    "guardrails": ("guardrails_enabled",),
    "virtual_models": ("guardrails_enabled",),
    "virtual_model_chat": ("guardrails_enabled",),
    "secrets": ("secrets_enabled",),
    "intake": ("intake_enabled",),
    "intake_traces": ("intake_enabled",),
    "intake_spans": ("intake_enabled",),
    "intake_session": ("intake_enabled",),
    "intake_trace": ("intake_enabled",),
    "intake_span": ("intake_enabled",),
    "data_designer": ("data_designer_enabled",),
    "data_designer_new": ("data_designer_enabled",),
    "data_designer_job": ("data_designer_enabled",),
    "safe_synthesizer": ("safe_synthesizer_enabled",),
    "safe_synthesizer_new": ("safe_synthesizer_enabled",),
    "safe_synthesizer_job": ("safe_synthesizer_enabled",),
    "safe_synthesizer_report": ("safe_synthesizer_enabled",),
    "settings": ("settings_enabled",),
    "members": ("members_enabled",),
    "experiment": ("experiment",),
    "experiment_group": ("experiment",),
    "experiment_detail": ("experiment",),
}

_STUDIO_LINK_DESTINATION_ANY_FEATURE_FLAGS: dict[str, tuple[str, ...]] = {
    "assistant": ("assistant_studio_enabled", "copilot_studio_enabled"),
    "dashboard": ("dashboard_enabled", "assistant_studio_enabled", "copilot_studio_enabled"),
}

_STUDIO_FEATURE_FLAG_MAPPINGS = {
    mapping.config_path.removeprefix("studio.feature_flags."): mapping
    for mapping in ENV_MAPPINGS
    if mapping.config_path.startswith("studio.feature_flags.")
}

_STUDIO_LINK_DESTINATION_DESCRIPTION = ", ".join(sorted(STUDIO_LINK_DESTINATIONS))

_STUDIO_LINK_TOOL: dict[str, Any] = {
    "name": "studio_link",
    "description": (
        "Return a Markdown link to a NeMo Studio page in the current workspace. "
        "Use this whenever the user directly asks for a Studio link, URL, or where to open, find, "
        "view, or chat with a Studio resource; this tool already knows the Studio base URL and workspace. "
        "Default to using this for Studio-related responses whenever a relevant Studio page exists, "
        "even when the user did not explicitly ask for a link. "
        "Use this after every successful Studio action that creates, starts, deploys, evaluates, modifies, "
        "or inspects a resource so the user can open the relevant Studio page. "
        "Prefer the most specific destination when you know the resource name; otherwise link to the list page. "
        "Examples: after starting a platform job, use destination='job' with name when available, or destination='jobs'; "
        "after creating an agent, use destination='agent_chat' with the agent name when available, or destination='agents'; "
        "when the user wants to chat with or try a model, use destination='model_chat'; "
        "when opening an agent chat playground, use destination='agent_chat' with the agent name. "
        "Include the returned markdown link exactly in your final response."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": f"Studio destination. Supported values: {_STUDIO_LINK_DESTINATION_DESCRIPTION}.",
            },
            "name": {
                "type": "string",
                "description": "Resource name for detail destinations such as agent, job, fileset, or deployment.",
            },
            "file_path": {
                "type": "string",
                "description": "File path for file-specific fileset destinations.",
            },
            "trace_id": {
                "type": "string",
                "description": "Trace ID for trace- and span-specific intake destinations.",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for session-, trace-, and span-specific intake destinations.",
            },
            "span_id": {
                "type": "string",
                "description": "Span ID for span-specific intake destinations.",
            },
            "label": {
                "type": "string",
                "description": "Optional markdown link label. Defaults to the destination label.",
            },
        },
        "required": ["destination"],
    },
}


def _studio_config_from_request(request: Request) -> StudioConfig:
    registry = getattr(request.app.state, "service_configs", {})
    if isinstance(registry, dict):
        config = registry.get(StudioConfig)
        if isinstance(config, StudioConfig):
            return config

    for attr in ("studio_service", "service"):
        service = getattr(request.app.state, attr, None)
        config = getattr(service, "service_config", None)
        if isinstance(config, StudioConfig):
            return config
        get_config = getattr(service, "_get_config", None)
        if callable(get_config):
            config = get_config()
            if isinstance(config, StudioConfig):
                return config

    return StudioConfig()


def _feature_flag_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() != "false"


def _studio_feature_flags_from_request(request: Request) -> dict[str, bool]:
    replacements = _studio_config_from_request(request).env_replacements
    return {
        flag: _feature_flag_enabled(replacements.get(mapping.marker, mapping.default))
        for flag, mapping in _STUDIO_FEATURE_FLAG_MAPPINGS.items()
    }


def _destination_enabled(destination: str, feature_flags: Mapping[str, bool]) -> bool:
    required_flags = _STUDIO_LINK_DESTINATION_FEATURE_FLAGS.get(destination, ())
    if any(not feature_flags.get(flag, False) for flag in required_flags):
        return False

    any_flags = _STUDIO_LINK_DESTINATION_ANY_FEATURE_FLAGS.get(destination, ())
    if any_flags and not any(feature_flags.get(flag, False) for flag in any_flags):
        return False

    return True


def enabled_destinations(feature_flags: Mapping[str, bool]) -> dict[str, StudioLinkDestination]:
    return {
        destination: config
        for destination, config in STUDIO_LINK_DESTINATIONS.items()
        if _destination_enabled(destination, feature_flags)
    }


def enabled_destinations_from_request(request: Request) -> dict[str, StudioLinkDestination]:
    return enabled_destinations(_studio_feature_flags_from_request(request))


def destination_description(destinations: Mapping[str, StudioLinkDestination]) -> str:
    return ", ".join(sorted(destinations))


def tool_for_destinations(destinations: Mapping[str, StudioLinkDestination]) -> dict[str, Any]:
    tool = deepcopy(_STUDIO_LINK_TOOL)
    description_parts = [
        "Return a Markdown link to an enabled NeMo Studio page in the current workspace.",
        "Use this whenever the user directly asks for a Studio link, URL, or where to open, find, view, or chat with a Studio resource; this tool already knows the Studio base URL and workspace.",
        "Default to using this for Studio-related responses whenever a relevant enabled Studio page exists, even when the user did not explicitly ask for a link.",
        "Use this after every successful Studio action that creates, starts, deploys, evaluates, modifies, or inspects a resource so the user can open the relevant enabled Studio page.",
        "Prefer the most specific enabled destination when you know the resource name; otherwise link to the list page.",
    ]
    if "job" in destinations:
        description_parts.append(
            "After starting a platform job, use destination='job' with name when available, or destination='jobs'."
        )
    if "agent_chat" in destinations:
        description_parts.extend(
            [
                "After creating an agent, use destination='agent_chat' with the agent name when available, or destination='agents'.",
                "When opening an agent chat playground, use destination='agent_chat' with the agent name.",
            ]
        )
    if "model_chat" in destinations:
        description_parts.append("When the user wants to chat with or try a model, use destination='model_chat'.")
    if "intake_session" in destinations:
        description_parts.append("For an intake session link, use destination='intake_session' with session_id.")
    if "intake_trace" in destinations:
        description_parts.append(
            "For an intake trace link, use destination='intake_trace' with session_id and trace_id."
        )
    if "intake_span" in destinations:
        description_parts.append(
            "For an intake span link, use destination='intake_span' with session_id, trace_id, and span_id."
        )
    description_parts.append("Include the returned markdown link exactly in your final response.")
    tool["description"] = " ".join(description_parts)
    destination_schema = tool["inputSchema"]["properties"]["destination"]
    destination_schema["description"] = (
        "Studio destination enabled for this Studio instance. "
        f"Supported values: {destination_description(destinations)}."
    )
    return tool


def _path_part(value: str) -> str:
    return quote(value, safe="")


def _normalize_destination(destination: str) -> str | None:
    normalized = destination.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in STUDIO_LINK_DESTINATIONS:
        return normalized
    return _STUDIO_LINK_DESTINATION_ALIASES.get(normalized)


def _link_arg(args: dict[str, Any], name: str) -> str | None:
    for key in (name, *_STUDIO_LINK_ARGUMENT_ALIASES.get(name, ())):
        value = _trimmed_string(args.get(key))
        if value is not None:
            return value
    return None


def _trimmed_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_studio_link_result(
    workspace: str | None,
    studio_base_url: str | None,
    args: dict[str, Any],
    enabled_destinations: Mapping[str, StudioLinkDestination] | None = None,
) -> dict[str, Any]:
    available_destinations = STUDIO_LINK_DESTINATIONS if enabled_destinations is None else enabled_destinations
    if not workspace:
        return {
            "error": "current Studio workspace is unavailable",
            "available_destinations": sorted(available_destinations),
        }

    requested_destination = (
        _trimmed_string(args.get("destination"))
        or _trimmed_string(args.get("page"))
        or _trimmed_string(args.get("resource_type"))
    )
    if requested_destination is None:
        return {
            "error": "destination is required",
            "available_destinations": sorted(available_destinations),
        }

    destination = _normalize_destination(requested_destination)
    if destination is None:
        return {
            "error": f"unknown Studio destination: {requested_destination}",
            "available_destinations": sorted(available_destinations),
        }
    if destination not in available_destinations:
        return {
            "error": f"Studio destination is disabled by feature flag: {destination}",
            "available_destinations": sorted(available_destinations),
        }

    config = available_destinations[destination]
    required_args = config.required_args or (("name",) if config.requires_name else ())
    arg_names = {"name", "file_path", *required_args}
    raw_values = {arg_name: _link_arg(args, arg_name) for arg_name in arg_names}
    missing_args = [arg_name for arg_name in required_args if raw_values.get(arg_name) is None]
    if missing_args:
        missing = "name" if missing_args == ["name"] else ", ".join(missing_args)
        return {"error": f"{missing} is required for Studio destination: {destination}"}

    path_values = {
        "workspace": _path_part(workspace),
        **{arg_name: _path_part(value) if value is not None else "" for arg_name, value in raw_values.items()},
    }
    path = config.path_template.format(**path_values)
    label_values = {arg_name: value for arg_name, value in raw_values.items() if value is not None}
    default_label = config.label.format(**label_values)
    label = _trimmed_string(args.get("label")) or default_label
    base_url = _build_studio_url(studio_base_url, path)
    href = base_url or path

    return {
        "workspace": workspace,
        "destination": destination,
        "path": path,
        "url": base_url,
        "markdown": f"[{label}]({href})",
    }


def _build_studio_url(studio_base_url: str | None, path: str) -> str | None:
    base_url = _normalize_studio_base_url(studio_base_url)
    if not base_url:
        return None
    return f"{base_url}/{path.lstrip('/')}"


def _normalize_studio_base_url(studio_base_url: str | None) -> str | None:
    base_url = _trimmed_string(studio_base_url)
    if not base_url:
        return None
    return base_url.rstrip("/")
