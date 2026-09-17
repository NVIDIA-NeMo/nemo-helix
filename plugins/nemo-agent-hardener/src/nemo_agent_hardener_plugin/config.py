# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Agent Hardener plugin.

Declares :attr:`plugin_name` / :attr:`plugin_description` as ``ClassVar`` strings and
plugin-specific fields with defaults, following the
:class:`~nemo_platform_plugin.config.NemoConfig` pattern.

Operators set values via environment variables (``NEMO_AGENT_HARDENER_*``) or the Helm
``platformConfig.agent_hardener`` key. agent-hardener runs in its own isolated venv (:attr:`venv_path`)
and the plugin invokes its CLI by subprocess rather than importing it. garak — which agent-hardener's
agent_breaker attacker spawns — lives in a *second* dedicated venv (:attr:`garak_venv_path`), kept
separate because garak pulls ``litellm`` (``httpx>=0.28``) and ``torch`` that would otherwise
conflict with agent-hardener's dependencies. The plugin points agent-hardener at it via the
``AGENT_HARDENER_GARAK_PYTHON`` environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import yaml
from nemo_platform_plugin.config import NemoConfig
from pydantic import Field, field_validator

# Env var agent-hardener reads to locate the garak venv its agent_breaker attacker spawns. The plugin
# exports it (to ``garak_python``) for both ``agent-hardener setup`` (provision) and ``agent-hardener run``.
GARAK_PYTHON_ENVVAR = "AGENT_HARDENER_GARAK_PYTHON"

# agent-hardener's orchestrator reads this directly from the process env (no IGW routing).
INFERENCE_API_KEY_ENVVAR = "INFERENCE_API_KEY"  # pragma: allowlist secret


def _default_venv_path() -> Path:
    """Default location for agent-hardener's dedicated venv (created by ``nemo agent-hardener setup``)."""
    return Path.home() / ".agent-hardener" / "venv"


def _default_garak_venv_path() -> Path:
    """Default location for the dedicated garak venv agent-hardener's agent_breaker spawns.

    Matches agent-hardener's own default (``~/.agent-hardener/garak-venv``) so the
    ``AGENT_HARDENER_GARAK_PYTHON`` export and agent-hardener's fallback agree.
    """
    return Path.home() / ".agent-hardener" / "garak-venv"


def _default_operator_env_file() -> Path:
    """Default location for agent-hardener's own operator dotenv (provisioned by ``setup``)."""
    return Path.home() / ".agent-hardener" / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal dotenv reader: skips blank/`#` lines, strips `export `/quotes. `{}` if missing."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def write_env_file(path: Path, values: Mapping[str, str]) -> None:
    """Write *values* as a dotenv at *path*, mode 0600 from creation.

    Opened with the mode applied up front rather than chmod'd afterwards: a plain write lands at the
    default umask (typically 0644), leaving the credentials world-readable until the chmod lands.
    Every dotenv this plugin writes holds provider keys, so both call sites go through here.
    """
    body = "".join(f"{key}={value}\n" for key, value in values.items())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.chmod(path, 0o600)  # O_CREAT ignores the mode when the file already exists


def missing_secrets(
    manifest_path: Path,
    *,
    env_files: Iterable[Path] = (),
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the manifest's declared victim secrets that no available source provides.

    Sources: *environ* (defaults to ``os.environ``), the manifest's own ``secrets_file`` (resolved
    next to the manifest), and each dotenv in *env_files* (e.g. the operator env + a ``--env-file``).
    A name set to an empty value counts as missing — ``KEY=`` in a dotenv or ``export KEY=""`` would
    otherwise pass this gate and resurface minutes later as a provider auth error.
    Returns the missing names in declaration order; ``[]`` when none are declared or the manifest
    can't be read (parsing is not this check's job).
    """
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    agent = data.get("agent", {}) if isinstance(data, dict) else {}
    declared = [name for name in (agent.get("secrets") or []) if isinstance(name, str)]
    if not declared:
        return []

    available = _non_empty_keys(os.environ if environ is None else environ)
    secrets_file = agent.get("secrets_file")
    if isinstance(secrets_file, str) and secrets_file:
        # `secrets_file` may be relative to the manifest or already absolute (see jobs/manifest.py);
        # `/` handles both.
        available |= _non_empty_keys(read_env_file(manifest_path.parent / secrets_file))
    for path in env_files:
        available |= _non_empty_keys(read_env_file(Path(path)))
    return [name for name in declared if name not in available]


def _non_empty_keys(values: Mapping[str, str]) -> set[str]:
    """Names in *values* that carry an actual value; a blank one provides nothing."""
    return {name for name, value in values.items() if value and value.strip()}


class AgentHardenerConfig(NemoConfig):
    """Configuration for the NeMo Platform Agent Hardener plugin.

    All fields have defaults so the plugin loads without operator configuration; the
    agent-hardener venv itself is provisioned on demand by ``nemo agent-hardener setup``.
    """

    plugin_name: ClassVar[str] = "agent_hardener"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Platform Agent Hardener plugin."

    default_workspace: str = Field(
        default="default",
        description="Workspace used to resolve agents and store run records when none is given.",
    )
    venv_path: Path = Field(
        default_factory=_default_venv_path,
        description=(
            "Directory holding agent-hardener's dedicated venv. The plugin invokes "
            "{venv_path}/bin/agent-hardener by subprocess. Set NEMO_AGENT_HARDENER_VENV_PATH to override."
        ),
    )
    spec: str = Field(
        default="nvidia-agent-hardener>=0.0.11",
        description=(
            "Package spec `nemo agent-hardener setup` installs into the venv (e.g. 'agent-hardener', "
            "'agent-hardener==0.0.1', or a local path/VCS URL for development). The floor is the release "
            "that added `init --dockerfile/--binary`, which the BYO launch mode depends on."
        ),
    )
    index_url: str | None = Field(
        default=None,
        description=(
            "Extra package index `setup` resolves agent-hardener from, passed as uv's `--index`. Additive "
            "to PyPI rather than a replacement, and scoped to this one install so the platform's own "
            "dependencies are never resolved against it. Accepts a bare URL or uv's named form "
            "`<name>=<url>` — use the named form when authenticating via "
            "UV_INDEX_<NAME>_USERNAME/PASSWORD, since those variables key off the index name and a "
            "bare URL gets an auto-generated one they won't match (a ~/.netrc entry works with "
            "either). Unset by default: agent-hardener installs from PyPI. Set NEMO_AGENT_HARDENER_INDEX_URL "
            "to override."
        ),
    )

    @field_validator("index_url")
    @classmethod
    def _require_https_index(cls, value: str | None) -> str | None:
        """Reject a plaintext index: ~/.netrc or an embedded-URL credential would cross the wire in the clear."""
        if not value:
            return value
        url = value.split("=", 1)[1] if "=" in value else value
        parsed = urlsplit(url)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError(
                "NEMO_AGENT_HARDENER_INDEX_URL must use https:// (a plaintext index would send its "
                "credentials unencrypted). Use http://localhost or http://127.0.0.1 for local development."
            )
        return value

    index_strategy: str | None = Field(
        default=None,
        description=(
            "uv `--index-strategy` for the agent-hardener install; unset uses uv's default, "
            "'first-index'. Use 'unsafe-best-match' when the extra index also carries packages that "
            "shadow their PyPI counterparts — first-index stops at the first index containing a "
            "package and would fail to resolve them. It relaxes uv's dependency-confusion protection "
            "for this resolution, which is why it is opt-in and scoped to this one install. Set "
            "NEMO_AGENT_HARDENER_INDEX_STRATEGY to override."
        ),
    )
    garak_venv_path: Path = Field(
        default_factory=_default_garak_venv_path,
        description=(
            "Directory holding the dedicated garak venv. agent-hardener's agent_breaker spawns garak "
            "from {garak_venv_path}/bin/python; the plugin exports AGENT_HARDENER_GARAK_PYTHON to it so "
            "`agent-hardener setup` provisions there (the garak version pin lives in agent-hardener). "
            "Set NEMO_AGENT_HARDENER_GARAK_VENV_PATH to override."
        ),
    )
    require_sandbox: bool = Field(
        default=True,
        description=(
            "When True, init/run preflight (doctor) fails hard if Docker or the OpenShell "
            "gateway is unavailable. Set False only for dry-run/manifest-only flows."
        ),
    )
    operator_env_file: Path = Field(
        default_factory=_default_operator_env_file,
        description=(
            "Dotenv holding agent-hardener's own inference credential, provisioned by `setup` and "
            "injected into every `run`. Set NEMO_AGENT_HARDENER_OPERATOR_ENV_FILE to override."
        ),
    )
    inference_secret_name: str = Field(
        default="agent-hardener-inference-key",
        description="NeMo Secret name `setup` reads agent-hardener's own inference key from, if present.",
    )

    @property
    def state_dir(self) -> Path:
        """Base dir for agent-hardener on-host state (the venvs live under it; also run-event logs)."""
        return self.venv_path.parent

    @property
    def agent_hardener_bin(self) -> Path:
        """Path to the agent-hardener CLI inside the dedicated venv."""
        return self.venv_path / "bin" / "agent-hardener"

    @property
    def garak_python(self) -> Path:
        """Path to the Python interpreter inside the dedicated garak venv."""
        return self.garak_venv_path / "bin" / "python"
