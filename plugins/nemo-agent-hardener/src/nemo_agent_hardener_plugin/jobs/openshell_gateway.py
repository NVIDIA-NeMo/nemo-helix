# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the configured OpenShell gateway for a war-game job that runs without `nemo agent-hardener setup`.

agent-hardener addresses its gateway by name only (``--gateway auto-defender``) and leaves endpoint and
auth to the ``openshell`` CLI's own registry under ``$XDG_CONFIG_HOME/openshell``. On a host, ``setup``
fills that registry. In a container nothing does, so the job registers
:attr:`~nemo_agent_hardener_plugin.config.AgentHardenerConfig.openshell_gateway_endpoint` into a private
config dir and hands the agent-hardener subprocess that dir as ``XDG_CONFIG_HOME``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from nemo_agent_hardener_plugin.cli.checks import OPENSHELL_GATEWAY
from nemo_agent_hardener_plugin.config import AgentHardenerConfig, write_private_file
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_MISSING_CREDENTIAL,
    CATEGORY_PROVISIONING,
    AgentHardenerRunError,
)

SecretResolver = Callable[[str], str | None]

XDG_CONFIG_HOME_ENVVAR = "XDG_CONFIG_HOME"
_REGISTER_TIMEOUT_SECONDS = 30
# The CLI only needs to find itself, write its registry and reach the endpoint; job credentials stay out.
_CLI_ENVVARS = (
    "PATH",
    "HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def register_openshell_gateway(
    plugin_config: AgentHardenerConfig, *, resolve_secret: SecretResolver, config_home: Path
) -> dict[str, str]:
    """Register the configured gateway as ``auto-defender`` under *config_home*; return the env pointing at it.

    *resolve_secret* maps a NeMo Secret name to its value. Returns ``{}`` when no endpoint is configured,
    leaving the host's own registration in force. The caller owns *config_home* and removes it, mTLS key
    included, once the job is done.
    """
    endpoint = plugin_config.openshell_gateway_endpoint
    if not endpoint:
        return {}
    openshell = shutil.which("openshell")
    if openshell is None:
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            "The openshell CLI is not on PATH, so the configured OpenShell gateway cannot be registered.",
            remediation=(
                "On a host, run `nemo agent-hardener setup` or put openshell on the platform's PATH; in a "
                "container, use the nhx-agent-hardener-tasks image, which ships it."
            ),
        )

    args = [openshell, "gateway", "add", endpoint, "--name", OPENSHELL_GATEWAY]
    if urlsplit(endpoint).scheme == "https":
        _write_mtls_bundle(config_home, plugin_config, resolve_secret)
        # `--local` is how the CLI registers an mTLS gateway whose bundle is already on disk; without it an
        # https:// endpoint is treated as edge-authenticated and opens a browser.
        args.append("--local")

    env = {name: value for name in _CLI_ENVVARS if (value := os.environ.get(name))}
    env[XDG_CONFIG_HOME_ENVVAR] = str(config_home)
    try:
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=_REGISTER_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING, f"Could not register the OpenShell gateway at {endpoint}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            f"Could not register the OpenShell gateway at {endpoint}: {(result.stderr or result.stdout).strip()}",
        )
    return {XDG_CONFIG_HOME_ENVVAR: str(config_home)}


def _write_mtls_bundle(home: Path, plugin_config: AgentHardenerConfig, resolve_secret: SecretResolver) -> None:
    mtls_dir = home / "openshell" / "gateways" / OPENSHELL_GATEWAY / "mtls"
    mtls_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    bundle = {
        "ca.crt": plugin_config.openshell_ca_secret,
        "tls.crt": plugin_config.openshell_client_cert_secret,
        "tls.key": plugin_config.openshell_client_key_secret,
    }
    for filename, secret_name in bundle.items():
        value = resolve_secret(secret_name) if secret_name else None
        if not value:
            raise AgentHardenerRunError(
                CATEGORY_MISSING_CREDENTIAL,
                f"The OpenShell mTLS Secret '{secret_name}' ({filename}) is missing or empty.",
                remediation="Create the Secret in this workspace with `nemo secrets create`.",
            )
        write_private_file(mtls_dir / filename, value)
