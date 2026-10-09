# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-container auth probe and the host-side report checks.

The probe source is sent as ``python -c`` because the task image does not
include ``nemo_helix_ext``. It uses ``get_task_nemo_client``, which exchanges a
workload-identity token when ``NHX_WORKLOAD_IDENTITY_TOKEN_FILE`` is set and
otherwise acts as the jobs or deployments service on behalf of ``NHX_PRINCIPAL``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

AUTH_REPORT_PREFIX = "INSTANCE_CHECK_AUTH "

# Executed inside the job or deployment. Keep this free of secret values and
# token contents; the host parses the single prefixed JSON line.
WORKLOAD_PROBE_SOURCE = r"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PREFIX = "INSTANCE_CHECK_AUTH "


def _env_info(name):
    value = os.environ.get(name)
    if not value:
        return {"set": False, "length": 0}
    return {"set": True, "length": len(value)}


def _principal():
    raw = os.environ.get("NHX_PRINCIPAL")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"id": "", "email": "", "on_behalf_of": ""}
    if not isinstance(parsed, dict):
        return {"id": "", "email": "", "on_behalf_of": ""}
    return {
        "id": str(parsed.get("id") or ""),
        "email": str(parsed.get("email") or ""),
        "on_behalf_of": str(parsed.get("on_behalf_of") or ""),
    }


def _short_error(exc):
    return f"{type(exc).__name__}: {exc}"[:400]


def _retryable(exc):
    text = _short_error(exc)
    return any(token in text for token in ("Connect", "timed out", "Timeout", "Connection refused"))


def main():
    workspace = os.environ.get("INSTANCE_CHECK_WORKSPACE", "")
    kind = os.environ.get("INSTANCE_CHECK_KIND", "job")
    service = os.environ.get("INSTANCE_CHECK_SERVICE", "jobs")
    token_file = os.environ.get("NHX_WORKLOAD_IDENTITY_TOKEN_FILE") or ""
    token_exists = bool(token_file) and Path(token_file).is_file()
    principal = _principal()
    if token_exists:
        mode = "workload_identity"
    elif principal and principal.get("id"):
        mode = "on_behalf_of"
    else:
        mode = "anonymous"

    report = {
        "kind": kind,
        "workspace": workspace,
        "mode": mode,
        "env": {
            "NHX_BASE_URL": _env_info("NHX_BASE_URL"),
            "NHX_PRINCIPAL": _env_info("NHX_PRINCIPAL"),
            "NHX_WORKLOAD_IDENTITY_TOKEN_FILE": _env_info("NHX_WORKLOAD_IDENTITY_TOKEN_FILE"),
            "NHX_ACCESS_TOKEN": _env_info("NHX_ACCESS_TOKEN"),
        },
        "principal": principal,
        "workload_token_file_exists": token_exists,
        "workload_identity_requested": os.environ.get("INSTANCE_CHECK_WORKLOAD_IDENTITY") == "1",
        "workspace_get": {"ok": False},
        "secret_expected": bool(os.environ.get("INSTANCE_CHECK_SECRET_SHA256")),
        "secret_matches": None,
        "model_call": {"skipped": True},
        "report_stored": False,
        "auth_proxy": os.environ.get("INSTANCE_CHECK_AUTH_PROXY") == "1",
    }
    exit_code = 0
    client = None
    deadline = time.monotonic() + 30
    while True:
        try:
            from nhx.common.client_factory import get_task_nemo_client
            from nemo_helix_plugin.workspaces.client import WorkspacesClient

            client = get_task_nemo_client(service, workspace=workspace or None)
            found = WorkspacesClient.from_client(client).get_workspace(name=workspace).data()
            report["workspace_get"] = {"ok": True, "name": getattr(found, "name", workspace)}
            exit_code = 0
            break
        except Exception as exc:
            report["workspace_get"] = {"ok": False, "error": _short_error(exc)}
            exit_code = 1
            client = None
            if not _retryable(exc) or time.monotonic() >= deadline:
                break
            time.sleep(2)

    if client is not None:
        try:
            from nemo_helix_plugin.auth.client import AuthenticationClient

            who = AuthenticationClient.from_client(client).authenticate_bearer_token_get().data()
            seen_id = str(getattr(who, "principal", "") or "")
            seen_obo = str(getattr(who, "on_behalf_of", "") or "")
            if seen_id or seen_obo:
                report["principal"] = {
                    "id": seen_id,
                    "email": str(getattr(who, "email", "") or ""),
                    "on_behalf_of": seen_obo,
                }
                report["mode"] = "on_behalf_of"
        except Exception:
            pass

    expected_hash = os.environ.get("INSTANCE_CHECK_SECRET_SHA256")
    if expected_hash:
        actual = os.environ.get("INSTANCE_CHECK_SECRET", "")
        matches = hashlib.sha256(actual.encode()).hexdigest() == expected_hash
        report["secret_matches"] = matches
        if not matches and exit_code == 0:
            exit_code = 1

    model = os.environ.get("INSTANCE_CHECK_MODEL")
    if model and client is not None and exit_code == 0:
        try:
            from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
            from nemo_helix_plugin.inference_gateway.types import JsonBody

            owner, sep, model_name = model.partition("/")
            model_workspace = owner if sep and owner and model_name and "/" not in model_name else workspace
            body = (
                InferenceGatewayClient.from_client(client)
                .openai_post(
                    workspace=model_workspace,
                    trailing_uri="v1/chat/completions",
                    body=JsonBody(
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                            "max_tokens": 16,
                        }
                    ),
                )
                .data()
            )
            choices = body.get("choices") if isinstance(body, dict) else None
            if not choices:
                raise RuntimeError("chat completion returned no choices")
            report["model_call"] = {"ok": True, "model": model}
        except Exception as exc:
            report["model_call"] = {"ok": False, "model": model, "error": _short_error(exc)}
            exit_code = 2

    fileset = os.environ.get("INSTANCE_CHECK_FILESET")
    report_path = os.environ.get("INSTANCE_CHECK_REPORT_PATH")
    if client is not None and fileset and report_path:
        try:
            from nemo_helix_plugin.files.client import FilesClient

            report["report_stored"] = True
            FilesClient.from_client(client).upload_file(
                workspace=workspace,
                name=fileset,
                path=report_path,
                content=json.dumps(report).encode(),
            )
        except Exception as exc:
            report["report_stored"] = False
            report["report_store_error"] = _short_error(exc)

    print(PREFIX + json.dumps(report), flush=True)
    return 0 if report.get("report_stored") else exit_code


if __name__ == "__main__":
    sys.exit(main())
"""


@dataclass(frozen=True)
class AuthEvaluation:
    """Problems fail the check. Notes are printed and the check still passes."""

    problems: list[str]
    notes: list[str]


def workload_probe_script() -> str:
    """Return the in-container probe, ready to pass as ``python -c``."""
    return WORKLOAD_PROBE_SOURCE.strip() + "\n"


def parse_auth_report(text: str) -> dict[str, Any] | None:
    """Return the last ``INSTANCE_CHECK_AUTH`` JSON object in *text*."""
    found: dict[str, Any] | None = None
    for line in text.splitlines():
        marker = line.find(AUTH_REPORT_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(AUTH_REPORT_PREFIX) :].strip()
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            found = parsed
    return found


def evaluate_auth_report(
    report: dict[str, Any],
    *,
    auth_enabled: bool,
    token_exchange_enabled: bool,
    require_token_exchange: bool,
    model_required: bool,
) -> AuthEvaluation:
    """Decide whether a workload auth report meets the acceptance checks.

    Profiles select the identity assertion. An anonymous server accepts a
    workload with no principal. A trusted-headers server must show on-behalf-of
    principal injection. A token-exchange server must present a workload-identity
    token file; principal injection alone fails that profile.
    """
    problems: list[str] = []
    notes: list[str] = []
    workspace_get = report.get("workspace_get")
    workspace_ok = isinstance(workspace_get, dict) and workspace_get.get("ok") is True
    if not workspace_ok:
        detail = ""
        if isinstance(workspace_get, dict) and workspace_get.get("error"):
            detail = f": {workspace_get['error']}"
        problems.append(f"workspace get failed{detail}")

    mode = report.get("mode")
    principal = report.get("principal")
    principal_id = principal.get("id") if isinstance(principal, dict) else ""
    if auth_enabled and mode == "anonymous" and not (report.get("auth_proxy") and workspace_ok):
        problems.append("authentication is enabled but the workload ran anonymously")
    elif auth_enabled and mode == "anonymous" and report.get("auth_proxy") and workspace_ok:
        notes.append("deployment authenticated through the auth-proxy sidecar")
    if mode == "workload_identity" and not report.get("workload_token_file_exists"):
        problems.append("workload-identity mode without a token file")
    if mode == "on_behalf_of" and not principal_id:
        problems.append("on-behalf-of mode without a principal id")

    identity_requested = report.get("workload_identity_requested") is True
    exchange_missing = token_exchange_enabled and mode != "workload_identity"
    if identity_requested and mode != "workload_identity":
        problems.append("workload identity was requested but no token file was present")
    elif exchange_missing:
        problems.append("token exchange is enabled but the workload did not present a workload-identity token")
    elif require_token_exchange and mode != "workload_identity":
        problems.append("workload-identity token required but the workload did not present one")

    if report.get("secret_expected") and report.get("secret_matches") is not True:
        problems.append("injected secret did not match")

    model_call = report.get("model_call")
    model_ok = isinstance(model_call, dict) and model_call.get("ok") is True
    if model_required and not model_ok:
        detail = ""
        if isinstance(model_call, dict) and model_call.get("error"):
            detail = f": {model_call['error']}"
        problems.append(f"model call failed{detail}")
    return AuthEvaluation(problems=problems, notes=notes)
