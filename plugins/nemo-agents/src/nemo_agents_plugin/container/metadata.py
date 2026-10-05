# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent identity extraction for OCI image labels.

Extracts structured metadata from agent config and project files to populate
OCI ``LABEL`` instructions in generated Dockerfiles.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT

NEMO_HELIX_AGENT_FRAMEWORK = "nemo_helix_agent"


def extract_agent_metadata(
    agent_config: Path,
    pyproject: Path | None = None,
    *,
    agent_version: str | None = None,
    agent_author: str | None = None,
    build_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Extract OCI label values for an agent image.

    Resolution order for each field:

    * **agent_name**: ``pyproject [project].name`` → Platform config ``name`` →
      config file stem
    * **agent_version**: *agent_version* arg → ``pyproject [project].version`` → ``YY.MM.DD``
    * **agent_author**: *agent_author* arg → ``git config user.name`` (run in the
      project's git repo) → ``"unknown"``
    * **agent_framework**: ``"nemo_helix_agent"`` for Platform-owned agent
      specs, ``"nemo_agent_toolkit"`` when config has a ``workflow`` key
    * **agent_id**: truncated SHA-256 of config + pyproject + build-env inputs,
      so changing ``--nat-version`` (etc.) yields a distinct identifier.
    * **build_timestamp**: honors ``SOURCE_DATE_EPOCH`` →
      ``git log -1 --format=%cI`` of the project repo → current UTC time
    * **description**: ``pyproject [project].description`` → Platform config
      ``description`` → ``"{workflow._type} agent"``
    * **licenses**: ``pyproject [project].license`` → ``""``
    * **revision**: ``git rev-parse HEAD`` in the project repo → ``""``
    * **source**: ``git remote get-url origin`` in the project repo → ``""``

    All ``git`` invocations are scoped to ``cwd=pyproject.parent`` (or
    ``agent_config.parent`` when no pyproject is given), so the labels never
    reflect an unrelated repo just because the CLI happened to be invoked
    from a different working directory.
    """
    pyproject_data = _load_pyproject(pyproject)
    config_text = agent_config.read_text(encoding="utf-8") if agent_config.exists() else ""
    config_data = _parse_config(config_text)

    # Git commands and timestamp resolution all operate against the project's
    # repo, not the CLI's cwd. Without this, running the packager from `~`
    # against a config in `~/repos/agent/` would stamp `~`'s git revision
    # (or empty string) into the image labels.
    cwd = pyproject.resolve().parent if pyproject is not None else agent_config.resolve().parent

    name = _resolve_name(pyproject_data, config_data, agent_config)
    version = _resolve_version(agent_version, pyproject_data)
    author = _resolve_author(agent_author, cwd=cwd)
    framework = _detect_framework(config_data)
    agent_id = _compute_agent_id(config_text, pyproject, build_env=build_env)
    timestamp = _resolve_timestamp(cwd=cwd)
    description = _resolve_description(pyproject_data, config_data)
    licenses = _resolve_licenses(pyproject_data)
    revision = _git_revision(cwd=cwd)
    source = _git_source(cwd=cwd)

    return {
        "agent_name": name,
        "agent_version": version,
        "agent_author": author,
        "agent_framework": framework,
        "agent_id": agent_id,
        "build_timestamp": timestamp,
        "description": description,
        "licenses": licenses,
        "revision": revision,
        "source": source,
    }


def _load_pyproject(pyproject: Path | None) -> dict:
    if pyproject is None or not pyproject.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # ty: ignore[unresolved-import]
    # Only swallow the *parse* failure: a malformed ``pyproject.toml`` should
    # not stop a build that is otherwise valid (we fall back to filename-based
    # name resolution, env-var version, etc.).  ``OSError`` (permission /
    # races with file deletion) and ``UnicodeDecodeError`` (binary file
    # passed in) are real bugs and propagate so the operator sees them.
    try:
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return {}


def _parse_config(config_text: str) -> dict:
    """Parse agent YAML once for all metadata field resolvers."""
    try:
        data = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_name(pyproject_data: dict, config_data: dict, agent_config: Path) -> str:
    project_name = pyproject_data.get("project", {}).get("name", "")
    if isinstance(project_name, str) and project_name:
        return project_name
    if config_data.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        config_name = config_data.get("name", "")
        if isinstance(config_name, str) and config_name:
            return config_name
    return agent_config.stem


def _resolve_version(explicit: str | None, pyproject_data: dict) -> str:
    if explicit:
        return explicit
    version = pyproject_data.get("project", {}).get("version", "")
    if version:
        return version
    today = date.today()
    return f"{today.year % 100}.{today.month:02d}.{today.day:02d}"


def _resolve_author(explicit: str | None, cwd: Path | None = None) -> str:
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # git missing or hung: fall back to the documented sentinel.
        return "unknown"
    return "unknown"


def _resolve_timestamp(cwd: Path | None = None) -> str:
    """Return an ISO-8601 timestamp for the ``image.created`` label.

    Honors ``SOURCE_DATE_EPOCH`` (per reproducible-builds.org), then falls
    back to the project repo's HEAD commit time, then to wall-clock UTC.
    A stable timestamp lets ``docker build`` produce byte-identical images
    across CI runs when the source has not changed.
    """
    sde = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if sde:
        try:
            return datetime.fromtimestamp(int(sde), UTC).isoformat()
        except ValueError:
            # SOURCE_DATE_EPOCH was set but not a parseable integer (e.g.
            # "" after strip, "abc", "1.5"). Don't fail the build — fall
            # through to the git-commit-time / wall-clock fallbacks below
            # so a malformed env var degrades gracefully instead of
            # aborting the package step on every CI run.
            pass
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # git missing or hung: fall back to wall-clock UTC.
        return datetime.now(UTC).isoformat()
    return datetime.now(UTC).isoformat()


def _detect_framework(config_data: dict) -> str:
    if config_data.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        return NEMO_HELIX_AGENT_FRAMEWORK
    if "workflow" in config_data:
        return "nemo_agent_toolkit"
    return "unknown"


def _compute_agent_id(
    config_text: str,
    pyproject: Path | None,
    build_env: dict[str, str] | None = None,
) -> str:
    """Return a 12-char content hash over the agent's identity-defining inputs.

    Domain-separated so distinct inputs cannot accidentally collide (e.g.
    config="ab", pyproject="cdef" vs config="abc", pyproject="def"). Includes
    *build_env* (runtime contract, dependency pins, base image, and Python
    toolchain) so a rebuild with a different runtime produces a distinct id,
    instead of silently re-tagging an incompatible image with the same suffix.
    """
    hasher = hashlib.sha256()
    hasher.update(b"agent_config\0")
    hasher.update(config_text.encode("utf-8"))
    hasher.update(b"\0pyproject\0")
    if pyproject is not None and pyproject.exists():
        hasher.update(pyproject.read_text(encoding="utf-8").encode("utf-8"))
    hasher.update(b"\0build_env\0")
    if build_env:
        for key in sorted(build_env):
            hasher.update(f"{key}={build_env[key]}\0".encode())
    return hasher.hexdigest()[:12]


def _resolve_description(pyproject_data: dict, config_data: dict) -> str:
    desc = pyproject_data.get("project", {}).get("description", "")
    if isinstance(desc, str) and desc:
        return desc
    if config_data.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        config_description = config_data.get("description", "")
        if isinstance(config_description, str) and config_description:
            return config_description
    wf = config_data.get("workflow", {})
    if isinstance(wf, dict) and wf.get("_type"):
        return f"{wf['_type']} agent"
    return ""


def _resolve_licenses(pyproject_data: dict) -> str:
    project = pyproject_data.get("project", {})
    # PEP 639: [project].license is a string SPDX expression
    lic = project.get("license", "")
    if isinstance(lic, str) and lic:
        return lic
    # Legacy: [project].license = {text = "..."}
    if isinstance(lic, dict):
        return lic.get("text", "")
    return ""


def _git_revision(cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # git missing or hung: leave the OCI revision label empty.
        return ""
    return ""


def _git_source(cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _strip_credentials(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # git missing or hung: leave the OCI source label empty.
        return ""
    return ""


def _strip_credentials(url: str) -> str:
    """Strip embedded credentials from a git remote URL.

    Prevents GitLab PATs, GitHub tokens, or passwords embedded in the
    developer's git remote (e.g. an ``https://`` URL with a ``<user>:<token>@``
    userinfo segment) from being baked into the
    ``org.opencontainers.image.source`` OCI label.

    Handles:
      * HTTPS/HTTP with ``user:password@``, ``token@``, or ``oauth2:token@``
      * HTTPS/HTTP with credentials in the query string
        (``?token=...``, ``?access_token=...``) — dropped wholesale, since git
        remote URLs never carry meaningful query strings or fragments
      * SSH (``git@host:path``) — returned unchanged, no credentials possible
      * ``ssh://...`` URLs — userinfo segment stripped if present
      * Any other scheme containing ``@`` in the authority — stripped defensively
    """
    from urllib.parse import urlsplit, urlunsplit

    # SCP-like SSH remote (no scheme, e.g. "git@host:org/repo.git"): the "@" is
    # part of the canonical syntax and does not encode a secret, keep verbatim.
    if "://" not in url:
        return url

    try:
        parts = urlsplit(url)
    except ValueError:
        # Unparseable URL (e.g. invalid IPv6 bracketing) — best-effort
        # fallback returns the raw string so the OCI label is at least
        # populated; the credential-leak risk is limited because the
        # offending URL also failed Python's basic syntax check.
        return url

    scheme = parts.scheme
    netloc = parts.netloc

    # Defensively scrub query/fragment for HTTP(S) git remotes: they are
    # never semantically meaningful on a git URL (clone never reads them),
    # but they are a common vehicle for short-lived tokens in mirror setups
    # and CI helpers (?token=..., ?access_token=...).
    query = "" if scheme in ("http", "https") else parts.query
    fragment = "" if scheme in ("http", "https") else parts.fragment

    if "@" not in netloc:
        return urlunsplit((scheme, netloc, parts.path, query, fragment))

    userinfo, _, host = netloc.rpartition("@")

    # A bare username on non-HTTP schemes (e.g. ``ssh://git@host``) is the
    # canonical git identity, not a secret — preserve it. Strip whenever
    # userinfo contains a password/token separator (``:``) or when the
    # scheme is http/https (where a bare user field is often itself a token,
    # as with GitHub PATs and GitLab job tokens).
    if ":" in userinfo or scheme in ("http", "https"):
        return urlunsplit((scheme, host, parts.path, query, fragment))
    return urlunsplit((scheme, netloc, parts.path, query, fragment))
