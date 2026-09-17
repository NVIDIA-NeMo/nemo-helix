# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only readiness checks for Insights analysis."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from nemo_insights_plugin.analyst.analyst_backend import make_analyst_backend
from nemo_insights_plugin.contracts.checks import CheckResult, make_check_result
from nemo_insights_plugin.platform_client import make_client
from nemo_insights_plugin.profile import AnalysisProfile
from nemo_platform import AsyncNeMoPlatform, NeMoPlatformError
from nemo_platform_plugin.nooa_model_client import configured_model_refs

_EXPECTED_PLATFORM_ERRORS = (NeMoPlatformError, httpx.HTTPError, OSError, RuntimeError, ValueError)


def _default_http_ok(base_url: str) -> bool:
    try:
        return (
            httpx.get(
                f"{base_url.rstrip('/')}/health/ready",
                timeout=5,
                follow_redirects=True,
            ).status_code
            < 500
        )
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return False


async def _default_workspace_ok(base_url: str, workspace: str, agent: str) -> bool:
    client: AsyncNeMoPlatform | None = None
    try:
        client = make_client(base_url)
        backend = make_analyst_backend(client=client, insights_output=None)
        await backend.count_agent_sessions(agent=agent, workspace=workspace)
        return True
    except _EXPECTED_PLATFORM_ERRORS:
        return False
    finally:
        if client is not None:
            try:
                await client.close()
            except _EXPECTED_PLATFORM_ERRORS:
                pass


@dataclass(frozen=True)
class AnalysisProbes:
    """Dependencies used by read-only environment checks."""

    http_ok: Callable[[str], bool] = _default_http_ok
    workspace_ok: Callable[[str, str, str], Awaitable[bool]] = _default_workspace_ok


def check_profile(
    profile: AnalysisProfile | None,
    profile_error: str | None,
) -> list[CheckResult]:
    """Check that a profile was found and parsed."""
    if profile_error is not None:
        return [
            CheckResult(
                name="profile-parse",
                group="profile",
                status="fail",
                severity="required",
                message=profile_error,
                hint="fix optimizer.yaml or pass --profile with a valid file",
            )
        ]
    if profile is None:
        return [
            CheckResult(
                name="profile-found",
                group="profile",
                status="fail",
                severity="required",
                message="no optimizer.yaml found (searched cwd and parents)",
                hint="create optimizer.yaml with at least `agent: <name>`",
            )
        ]
    return [
        CheckResult(
            name="profile-found",
            group="profile",
            status="pass",
            severity="required",
            message=f"profile for agent {profile.agent!r} at {profile.profile_dir}",
        )
    ]


def check_ethos(
    ethos_path: Path | None,
    ethos_error: str | None,
) -> list[CheckResult]:
    """Check the optional Ethos content, including explicit UTF-8 readability."""
    return read_ethos(ethos_path, ethos_error)[1]


def read_ethos(
    ethos_path: Path | None,
    ethos_error: str | None,
) -> tuple[str | None, list[CheckResult]]:
    """Read the optional Ethos as UTF-8 and return its readiness check."""
    if ethos_error is not None:
        return None, [
            CheckResult(
                name="ethos",
                group="artifacts",
                status="fail",
                severity="required",
                message=ethos_error,
            )
        ]
    if ethos_path is None:
        return None, [
            CheckResult(
                name="ethos",
                group="artifacts",
                status="pass",
                severity="advisory",
                message="Ethos omitted (optional)",
            )
        ]
    try:
        content = ethos_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [
            CheckResult(
                name="ethos",
                group="artifacts",
                status="fail",
                severity="required",
                message=f"Could not read Ethos {ethos_path} as UTF-8: {exc}",
                hint="ensure the file is readable and encoded as UTF-8",
            )
        ]
    return content, [
        CheckResult(
            name="ethos",
            group="artifacts",
            status="pass",
            severity="required",
            message=f"Ethos readable at {ethos_path}",
        )
    ]


def check_models() -> list[CheckResult]:
    """Check that the active Platform context has a complete model pair."""
    try:
        refs = configured_model_refs()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return [
            CheckResult(
                name="agent-models",
                group="models",
                status="fail",
                severity="required",
                message=str(exc),
                hint="run `nemo setup` and select default and fast agent models",
            )
        ]
    return [
        CheckResult(
            name="agent-models",
            group="models",
            status="pass",
            severity="required",
            message=f"default={refs.default}; fast={refs.fast}",
        )
    ]


async def check_environment(
    *,
    agent: str | None,
    workspace: str | None,
    base_url: str,
    profile_dir: Path | None,
    probes: AnalysisProbes | None = None,
) -> list[CheckResult]:
    """Run model-selection and advisory platform checks without persisting state.

    The workspace probe is profile-dependent and skipped when *agent* or
    *workspace* is unknown; model selection and reachability checks always run.
    """
    active = probes or AnalysisProbes()
    del profile_dir
    results = check_models()
    reachable = active.http_ok(base_url)
    results.append(
        make_check_result(
            "platform-reachable",
            "platform",
            reachable,
            "advisory",
            f"{base_url} reachable",
            f"{base_url} unreachable",
            hint="check --base-url/NMP_BASE_URL and platform health",
        )
    )
    if agent is not None and workspace is not None:
        queryable = await active.workspace_ok(base_url, workspace, agent)
        results.append(
            make_check_result(
                "workspace-query",
                "platform",
                queryable,
                "advisory",
                f"workspace {workspace!r} can be queried for agent {agent!r}",
                f"workspace {workspace!r} could not be queried for agent {agent!r}",
                hint="check the workspace, authentication context, and Intake availability",
            )
        )
    return results
