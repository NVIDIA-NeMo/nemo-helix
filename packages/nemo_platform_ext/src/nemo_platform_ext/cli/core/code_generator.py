# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code generation for ``--output-format code``.

Renders the typed-client Python equivalent of a CLI invocation: the service
client import, the client construction, the method call with the same keyword
arguments (request models rendered as constructor calls), and how to consume
the response. Optional wait/watch lifecycle blocks are appended for commands
that support ``--wait`` / ``--watch``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, RootModel, SecretBytes, SecretStr

from nemo_platform_ext.cli.core.context import CLIContext

ResultKind = Literal["entity", "list", "none", "binary"]

_INFERENCE_DEPLOYMENT_LIFECYCLE = "inference_deployment"


def handle_code_generation(
    client_cls: type,
    method: str,
    kwargs: Mapping[str, Any],
    output_format: str | None,
    context: CLIContext,
    *,
    result: ResultKind = "entity",
    wait_config: dict[str, Any] | None = None,
    wait_options: dict[str, Any] | None = None,
    watch_config: dict[str, Any] | None = None,
    watch_options: dict[str, Any] | None = None,
) -> bool:
    """Print generated code and return True when *output_format* is ``code``.

    Args:
        client_cls: Typed service client class the command uses (e.g. ``SecretsClient``).
        method: Client method name (e.g. ``"create_secret"``).
        kwargs: Keyword arguments passed to the method. Pydantic models, enums,
            dicts, and primitives are rendered as Python source.
        output_format: Resolved output format.
        context: CLI context, used for the base URL.
        result: How the response is consumed in the snippet.
    """
    if output_format != "code":
        return False

    code = generate_python_code(
        client_cls,
        method,
        kwargs,
        base_url=context.get_base_url("http://localhost:8080"),
        result=result,
        wait_config=wait_config,
        wait_options=wait_options,
        watch_config=watch_config,
        watch_options=watch_options,
    )
    print(format_code_output(code, language="python"))
    return True


def generate_python_code(
    client_cls: type,
    method: str,
    kwargs: Mapping[str, Any],
    *,
    base_url: str | None = None,
    result: ResultKind = "entity",
    wait_config: dict[str, Any] | None = None,
    wait_options: dict[str, Any] | None = None,
    watch_config: dict[str, Any] | None = None,
    watch_options: dict[str, Any] | None = None,
) -> str:
    """Generate the typed-client Python code equivalent to a CLI command.

    Args:
        client_cls: Typed client class the command uses (e.g. ``ModelsClient``).
        method: Method name on that client (e.g. ``"list_models"``).
        kwargs: Keyword arguments the CLI passed; ``None`` values are omitted and
            request models render as constructor calls with their imports.
        base_url: Base URL for the client constructor, when configured.
        result: How the response is consumed: ``entity`` prints ``response.data()``,
            ``list`` iterates the first page, ``none`` and ``binary`` render no read.
        wait_config / wait_options: Emit a status wait after the call.
        watch_config / watch_options: Emit a job watch after the call.

    Returns:
        Python source for the snippet, imports first.
    """
    if wait_config and watch_config:
        raise ValueError("Only one of wait_config or watch_config may be provided")

    lifecycle_config = watch_config or wait_config
    lifecycle_options = watch_options if watch_config else wait_options
    lifecycle_mode = "watch" if watch_config else "wait" if wait_config else None
    lifecycle_type = lifecycle_config.get("type") if lifecycle_config else None

    imports = _ImportCollector()
    imports.add(client_cls)
    rendered_args = [f"{key}={_render_value(value, imports)}" for key, value in kwargs.items() if value is not None]

    lines: list[str] = []
    if lifecycle_type == _INFERENCE_DEPLOYMENT_LIFECYCLE:
        lines.append("import time")
        imports.add_name("nemo_platform_plugin.client.errors", "NemoHTTPError")
        imports.add_name("nemo_platform_plugin.client.errors", "NemoTransportError")
        imports.add_name("nemo_platform_plugin.client.errors", "NotFoundError")
        imports.add_name("nemo_platform_plugin.inference_gateway.client", "InferenceGatewayClient")
    lines.extend(imports.render())
    lines.append("")

    if base_url:
        lines.append(f"client = {client_cls.__name__}(base_url={_format_python_literal(base_url)})")
    else:
        lines.append(f"client = {client_cls.__name__}.from_config()")
    lines.append("")

    _append_method_call(lines, "client", method, rendered_args)
    lines.extend(_render_result(result))

    if lifecycle_config:
        lines.append("")
        lines.append(
            _render_lifecycle_code(
                kwargs,
                lifecycle_config,
                lifecycle_options or {},
                mode=lifecycle_mode,
            )
        )

    return "\n".join(lines)


def format_code_output(code: str, language: str = "python") -> str:
    """Syntax-highlight generated code when writing to a terminal; plain text otherwise."""
    from rich.console import Console
    from rich.syntax import Syntax

    from nemo_platform_ext.cli.core.api import is_tty

    if not is_tty():
        return code

    console = Console()
    syntax = Syntax(
        code,
        language,
        line_numbers=False,
        background_color="black",
        padding=(1, 2),
    )

    with console.capture() as capture:
        console.print(syntax)

    return capture.get()


class _ImportCollector:
    """Collects ``from module import Name`` lines for types used in the snippet."""

    def __init__(self) -> None:
        self._names: dict[str, set[str]] = {}

    def add(self, cls: type) -> None:
        self.add_name(cls.__module__, cls.__name__)

    def add_name(self, module: str, name: str) -> None:
        self._names.setdefault(module, set()).add(name)

    def render(self) -> list[str]:
        return [f"from {module} import {', '.join(sorted(names))}" for module, names in sorted(self._names.items())]


def _render_value(value: Any, imports: _ImportCollector) -> str:
    """Render *value* as Python source, registering imports for models and enums.

    Secret fields are rendered as a masked placeholder so generated code never
    embeds credentials.
    """
    if isinstance(value, (SecretStr, SecretBytes)):
        return _format_python_literal("***")
    if isinstance(value, RootModel):
        # The payload is the root value, whatever its shape (model, dict, list, scalar).
        imports.add(type(value))
        return f"{type(value).__name__}({_render_value(value.root, imports)})"
    if isinstance(value, BaseModel):
        imports.add(type(value))
        fields = value.model_dump(exclude_unset=True)
        rendered = ", ".join(
            f"{name}={_render_value(getattr(value, name, field_value), imports)}"
            for name, field_value in fields.items()
        )
        return f"{type(value).__name__}({rendered})"
    if isinstance(value, Enum):
        imports.add(type(value))
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, dict):
        items = ", ".join(f"{_format_python_literal(k)}: {_render_value(v, imports)}" for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        items = ", ".join(_render_value(v, imports) for v in value)
        return f"[{items}]"
    return _format_python_literal(value)


def _render_result(result: ResultKind) -> list[str]:
    if result == "entity":
        return ["", "print(response.data())"]
    if result == "list":
        return ["", "for item in response.page().items:", "    print(item)"]
    if result == "binary":
        return ["", "with response.stream() as chunks:", "    for chunk in chunks:", "        ..."]
    return []


def _append_method_call(lines: list[str], target: str, method: str, formatted_args: list[str]) -> None:
    if not formatted_args:
        lines.append(f"response = {target}.{method}()")
        return

    if len(formatted_args) <= 3 and all(len(arg) <= 40 for arg in formatted_args):
        lines.append(f"response = {target}.{method}({', '.join(formatted_args)})")
        return

    lines.append(f"response = {target}.{method}(")
    for i, arg in enumerate(formatted_args):
        comma = "," if i < len(formatted_args) - 1 else ""
        lines.append(f"    {arg}{comma}")
    lines.append(")")


def _format_python_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (datetime, date)):
        # Pydantic accepts ISO-8601 strings for date/datetime fields, and the
        # string round-trips into a model without a datetime import.
        return json.dumps(value.isoformat())
    return repr(value)


def _require_timeout(timeout: Any, lifecycle_type: object, mode: str | None) -> Any:
    if timeout is not None:
        return timeout
    mode_label = f"{mode} " if mode else ""
    raise ValueError(f"{mode_label}{lifecycle_type!r} lifecycle code generation requires timeout")


def _render_lifecycle_code(
    args: Mapping[str, Any],
    lifecycle_config: dict[str, Any],
    lifecycle_options: dict[str, Any],
    *,
    mode: str | None,
) -> str:
    lifecycle_type = lifecycle_config.get("type")
    if lifecycle_type != _INFERENCE_DEPLOYMENT_LIFECYCLE:
        raise ValueError(f"Unsupported lifecycle config type: {lifecycle_type!r}")

    timeout = _require_timeout(lifecycle_options.get("timeout"), lifecycle_type, mode)
    poll_interval = lifecycle_options.get("poll_interval", 3)
    resource_name = 'getattr(response.data(), "name", None)'
    body = args.get("body")
    body_name = getattr(body, "name", None) if body is not None else None
    if body_name is not None:
        resource_name = f"{resource_name} or {_format_python_literal(body_name)}"
    workspace_literal = _format_python_literal(args["workspace"]) if args.get("workspace") is not None else "None"
    flag = "--watch" if mode == "watch" else "--wait"

    prelude = dedent(
        f"""
        resource_name = {resource_name}
        if not resource_name:
            raise RuntimeError("Unable to determine created resource name for {flag}")
        deadline = time.monotonic() + {timeout}
        """
    ).strip()
    return "\n\n".join([prelude, _render_inference_deployment_wait_code(workspace_literal, poll_interval)])


def _render_inference_deployment_wait_code(workspace_literal: str, poll_interval: int) -> str:
    return dedent(
        f"""
        while True:
            deployment = client.get_deployment(name=resource_name, workspace={workspace_literal}).data()
            history = deployment.status_history
            status = (history[-1].status if history else deployment.status).value
            if status == "READY":
                provider_name = resource_name
                provider_workspace = {workspace_literal}
                if deployment.model_provider_id:
                    provider_workspace, _, provider_name = deployment.model_provider_id.partition("/")
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

        gateway = InferenceGatewayClient.from_client(client)
        while True:
            try:
                gateway.provider_ready(name=provider_name, workspace=provider_workspace)
                break
            except (NotFoundError, NemoTransportError):
                pass
            except NemoHTTPError as exc:
                if exc.status_code not in {{429, 502, 503, 504}}:
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for gateway readiness for {{resource_name!r}}")
            time.sleep(min({poll_interval}, remaining))
        """
    ).strip()
