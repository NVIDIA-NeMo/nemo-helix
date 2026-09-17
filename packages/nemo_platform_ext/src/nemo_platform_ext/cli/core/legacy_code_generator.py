# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code generation for ``--output-format code`` on the generated (Stainless SDK) commands.

Kept only while ``cli/commands/api/`` exists; hand-written commands use
:mod:`nemo_platform_ext.cli.core.code_generator`, which dispatches here when
handed a resource path instead of a typed client class.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from nemo_platform_ext.cli.core.code_generator import format_code_output
from nemo_platform_ext.cli.core.context import CLIContext

_INFERENCE_DEPLOYMENT_LIFECYCLE = "inference_deployment"
_PLATFORM_JOB_LIFECYCLE = "platform_job"
_LIFECYCLE_TYPES_WITH_DEADLINES = {_INFERENCE_DEPLOYMENT_LIFECYCLE}
_LIFECYCLE_TYPES_WITH_STATUS_ERROR_HANDLING = {_INFERENCE_DEPLOYMENT_LIFECYCLE}


def handle_code_generation(
    resource_path: list[str],
    method: str,
    sdk_kwargs: dict[str, Any],
    output_format: str,
    context: CLIContext,
    wait_config: dict[str, Any] | None = None,
    wait_options: dict[str, Any] | None = None,
    watch_config: dict[str, Any] | None = None,
    watch_options: dict[str, Any] | None = None,
) -> bool:
    """
    Check if in code generation mode and generate code if needed.

    Args:
        resource_path: Path to the resource (e.g., ["models"], ["customization", "configs"])
        method: Method name (e.g., "list", "create", "retrieve")
        sdk_kwargs: Arguments to pass to the SDK method
        output_format: Output format

    Returns:
        True if code was generated (don't execute command), False otherwise
    """
    if output_format == "code":
        base_url = context.get_base_url("http://localhost:8080")
        code = generate_python_code(
            resource_path=resource_path,
            method=method,
            args=sdk_kwargs,
            base_url=base_url,
            wait_config=wait_config,
            wait_options=wait_options,
            watch_config=watch_config,
            watch_options=watch_options,
        )
        formatted_code = format_code_output(code, language="python")
        print(formatted_code)
        return True

    return False


def generate_python_code(
    resource_path: list[str],
    method: str,
    args: dict[str, Any],
    base_url: str | None = None,
    wait_config: dict[str, Any] | None = None,
    wait_options: dict[str, Any] | None = None,
    watch_config: dict[str, Any] | None = None,
    watch_options: dict[str, Any] | None = None,
) -> str:
    """
    Generate Python SDK code equivalent to a CLI command.

    Args:
        resource_path: Path to the resource (e.g., ["models"], ["customization", "configs"])
        method: Method name (e.g., "list", "create", "retrieve")
        args: Dictionary of arguments to pass to the method
        base_url: Base URL for the client (if specified)

    Returns:
        Python code string
    """
    lines = []

    if wait_config and watch_config:
        raise ValueError("Only one of wait_config or watch_config may be provided")

    lifecycle_config = watch_config or wait_config
    lifecycle_options = watch_options if watch_config else wait_options
    lifecycle_type = lifecycle_config.get("type") if lifecycle_config else None

    lifecycle_mode = "watch" if watch_config else "wait" if wait_config else None

    if _lifecycle_uses_deadline(lifecycle_type):
        lines.append("import time")
    if _lifecycle_uses_status_error_handling(lifecycle_type):
        lines.append(
            "from nemo_platform import APIConnectionError, APIStatusError, APITimeoutError, NeMoPlatform, NotFoundError"
        )
    else:
        lines.append("from nemo_platform import NeMoPlatform")
        if lifecycle_type == _PLATFORM_JOB_LIFECYCLE:
            lines.append("from nemo_platform_plugin.client.adapter import client_from_platform")
            lines.append("from nemo_platform_plugin.jobs.client import JobsClient")
            if lifecycle_mode == "wait":
                lines.append("from nemo_platform_plugin.jobs.watch_types import JobStatusEvent, JobWatchTimeoutError")
    lines.append("")

    if base_url:
        lines.append(f"client = NeMoPlatform(base_url={_format_python_literal(base_url)})")
    else:
        lines.append("client = NeMoPlatform()")
    lines.append("")
    if lifecycle_type == _PLATFORM_JOB_LIFECYCLE:
        lines.append("jobs_client = client_from_platform(client, JobsClient)")
        lines.append("")

    resource_chain = "client." + ".".join(resource_path)
    _append_method_call(lines, resource_chain, method, _format_method_args(args))

    if lifecycle_config:
        lines.extend(
            [
                "",
                _render_lifecycle_code(
                    resource_path,
                    args,
                    lifecycle_config,
                    lifecycle_options or {},
                    mode=lifecycle_mode,
                ),
            ]
        )

    if lifecycle_type != _PLATFORM_JOB_LIFECYCLE:
        lines.append("")
        lines.append("print(response)")

    return "\n".join(lines)


def _format_method_args(args: dict[str, Any]) -> list[str]:
    return [f"{key}={_format_python_literal(value)}" for key, value in args.items() if value is not None]


def _append_method_call(lines: list[str], resource_chain: str, method: str, formatted_args: list[str]) -> None:
    if not formatted_args:
        lines.append(f"response = {resource_chain}.{method}()")
        return

    if len(formatted_args) <= 3 and all(len(arg) <= 40 for arg in formatted_args):
        lines.append(f"response = {resource_chain}.{method}({', '.join(formatted_args)})")
        return

    lines.append(f"response = {resource_chain}.{method}(")
    for i, arg in enumerate(formatted_args):
        comma = "," if i < len(formatted_args) - 1 else ""
        lines.append(f"    {arg}{comma}")
    lines.append(")")


def _format_keyword_args(args: dict[str, Any], keys: list[str]) -> str:
    formatted_args = []
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        formatted_args.append(f"{key}={_format_python_literal(value)}")
    return ", " + ", ".join(formatted_args) if formatted_args else ""


def _format_python_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _lifecycle_uses_deadline(lifecycle_type: object) -> bool:
    return lifecycle_type in _LIFECYCLE_TYPES_WITH_DEADLINES


def _lifecycle_uses_status_error_handling(lifecycle_type: object) -> bool:
    return lifecycle_type in _LIFECYCLE_TYPES_WITH_STATUS_ERROR_HANDLING


def _require_timeout(timeout: Any, lifecycle_type: object, mode: str | None) -> Any:
    if timeout is not None:
        return timeout
    mode_label = f"{mode} " if mode else ""
    raise ValueError(f"{mode_label}{lifecycle_type!r} lifecycle code generation requires timeout")


def _require_resource_label(lifecycle_config: dict[str, Any], lifecycle_type: object, mode: str | None) -> str:
    mode_label = f"{mode} " if mode else ""
    try:
        resource_label = lifecycle_config["resource_label"]
    except KeyError as exc:
        raise ValueError(
            f"{mode_label}{lifecycle_type!r} lifecycle code generation requires a non-empty resource_label"
        ) from exc
    if not isinstance(resource_label, str) or not resource_label.strip():
        raise ValueError(
            f"{mode_label}{lifecycle_type!r} lifecycle code generation requires a non-empty resource_label"
        )
    return resource_label


def _render_lifecycle_code(
    resource_path: list[str],
    args: dict[str, Any],
    lifecycle_config: dict[str, Any],
    lifecycle_options: dict[str, Any],
    *,
    mode: str | None,
) -> str:
    lifecycle_type = lifecycle_config.get("type")
    timeout = lifecycle_options.get("timeout")
    poll_interval = lifecycle_options.get("poll_interval", 3)
    resource_chain = "client." + ".".join(resource_path)
    status_kwargs = _format_keyword_args(args, ["workspace"])
    resource_name = 'getattr(response, "name", None)'
    if args.get("name") is not None:
        resource_name = f"{resource_name} or {_format_python_literal(args['name'])}"

    prelude = dedent(
        f"""
        resource_name = {resource_name}
        if not resource_name:
            raise RuntimeError("Unable to determine created resource name for --wait")
        """
    ).strip()
    if mode == "watch":
        prelude = prelude.replace("--wait", "--watch")

    if lifecycle_type == _INFERENCE_DEPLOYMENT_LIFECYCLE:
        timeout = _require_timeout(timeout, lifecycle_type, mode)
        prelude = "\n".join([prelude, f"deadline = time.monotonic() + {timeout}"])
        workspace_literal = _format_python_literal(args["workspace"]) if args.get("workspace") is not None else "None"
        return "\n\n".join(
            [
                prelude,
                _render_inference_deployment_wait_code(
                    resource_chain,
                    status_kwargs,
                    workspace_literal,
                    poll_interval,
                ),
            ]
        )

    if lifecycle_type == _PLATFORM_JOB_LIFECYCLE and mode == "watch":
        return "\n\n".join(
            [
                prelude,
                _render_platform_job_watch_code(args, timeout, poll_interval),
            ]
        )
    if lifecycle_type == _PLATFORM_JOB_LIFECYCLE and mode == "wait":
        timeout = _require_timeout(timeout, lifecycle_type, mode)
        resource_label = _require_resource_label(lifecycle_config, lifecycle_type, mode)
        return "\n\n".join(
            [
                prelude,
                _render_platform_job_wait_code(args, timeout, poll_interval, resource_label),
            ]
        )

    raise ValueError(f"Unsupported lifecycle config type: {lifecycle_type!r}")


def _render_inference_deployment_wait_code(
    resource_chain: str,
    status_kwargs: str,
    workspace_literal: str,
    poll_interval: int,
) -> str:
    return dedent(
        f"""
        while True:
            deployment = {resource_chain}.retrieve(resource_name{status_kwargs})
            history = getattr(deployment, "status_history", None)
            status = history[-1].status if history else deployment.status
            if status == "READY":
                response = deployment
                provider_name = resource_name
                provider_workspace = {workspace_literal}
                model_provider_id = getattr(deployment, "model_provider_id", None)
                if model_provider_id:
                    provider_workspace, _, provider_name = model_provider_id.partition("/")
                    if not provider_workspace or not provider_name:
                        provider_workspace = {workspace_literal}
                        provider_name = resource_name
                break
            if status in {{"ERROR", "LOST"}}:
                raise RuntimeError(f"Deployment {{resource_name!r}} ended with status {{status!r}}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for deployment {{resource_name!r}} to become READY")
            time.sleep(min({poll_interval}, remaining))

        while True:
            try:
                if provider_workspace is None:
                    client.inference.gateway.provider.ready(provider_name)
                else:
                    client.inference.gateway.provider.ready(provider_name, workspace=provider_workspace)
                break
            except NotFoundError:
                pass
            except (APIConnectionError, APITimeoutError):
                pass
            except APIStatusError as exc:
                if exc.status_code not in {{429, 502, 503, 504}}:
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for gateway readiness for {{resource_name!r}}")
            time.sleep(min({poll_interval}, remaining))
        """
    ).strip()


def _render_platform_job_watch_code(
    args: dict[str, Any],
    timeout: int | None,
    poll_interval: int,
) -> str:
    workspace = _format_python_literal(args["workspace"]) if args.get("workspace") is not None else "None"

    return dedent(
        f"""
        for event in jobs_client.watch_job(
            resource_name,
            workspace={workspace},
            timeout={timeout},
            poll_interval={poll_interval},
        ):
            print(event)
        """
    ).strip()


def _render_platform_job_wait_code(
    args: dict[str, Any],
    timeout: int,
    poll_interval: int,
    resource_label: str,
) -> str:
    workspace = _format_python_literal(args["workspace"]) if args.get("workspace") is not None else "None"

    return dedent(
        f"""
        resource_label = {_format_python_literal(resource_label)}
        try:
            for event in jobs_client.watch_job(
                resource_name,
                workspace={workspace},
                timeout={timeout},
                poll_interval={poll_interval},
                include_logs=False,
            ):
                if not isinstance(event, JobStatusEvent):
                    continue
                if not event.terminal:
                    continue
                if event.successful:
                    break
                raise RuntimeError(
                    f"{{resource_label.title()}} {{resource_name!r}} ended with status {{event.status!r}}"
                )
        except JobWatchTimeoutError as exc:
            raise TimeoutError(f"Timed out waiting for {{resource_label}} {{resource_name!r}} to complete") from exc
        """
    ).strip()
