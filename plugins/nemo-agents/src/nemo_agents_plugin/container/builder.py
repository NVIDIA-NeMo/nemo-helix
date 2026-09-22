# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker image builder for NeMo Platform agents.

Builds a Docker image either from a pre-existing Dockerfile or by rendering
one on-the-fly via :func:`~nemo_agents_plugin.container.template.render_nat_dockerfile`
or :func:`~nemo_agents_plugin.container.template.render_fabric_dockerfile`.

Uses `python-on-whales <https://github.com/gabrieldemarmiesse/python-on-whales>`_
for Docker operations so callers never need to shell out manually.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from nemo_agents_plugin.container.errors import (
    AgentConfigValidationError,
    ContainerToolingUnavailableError,
    ImageBuildError,
    ManagedFileConflictError,
)
from nemo_agents_plugin.entities import (
    NAT_WORKFLOW_CONFIG_FORMAT,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]
"""Sink for human-facing build progress. The CLI passes ``typer.echo``."""


def emit_progress(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)
    else:
        logger.info("%s", message)


def docker_build(
    *,
    context_dir: Path,
    dockerfile: Path | None = None,
    tag: str,
    build_args: dict[str, str] | None = None,
    platforms: list[str] | None = None,
    push: bool = False,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Build a Docker image and return the tag.

    Args:
        context_dir: Docker build context directory.
        dockerfile: Path to an existing Dockerfile.  When ``None`` the
            caller is expected to have already written a rendered Dockerfile
            into *context_dir*.
        tag: Image tag (e.g. ``"my-agent:1.0"``).
        build_args: Extra ``--build-arg`` key/value pairs forwarded to
            ``docker build``.
        platforms: ``--platform`` values forwarded to ``docker build``. Up
            to one entry — multi-arch builds via buildx are not yet
            implemented and are rejected at the CLI layer.
        push: Push the image as part of the build (single round-trip).
        on_progress: Sink for progress lines; defaults to the module logger.

    Returns:
        The image tag that was built.

    Raises:
        ContainerToolingUnavailableError: When python-on-whales is missing.
        ImageBuildError: On build failure.
    """
    try:
        from python_on_whales import docker
    except ImportError as exc:
        raise ContainerToolingUnavailableError("building images") from exc

    # The plugin's Dockerfile uses BuildKit cache mounts (``RUN --mount=...``)
    # which silently fail on older daemons without ``DOCKER_BUILDKIT=1``.
    # ``setdefault`` so a user who explicitly sets ``DOCKER_BUILDKIT=0``
    # (e.g. to debug a layer) is not surprised by the override.
    os.environ.setdefault("DOCKER_BUILDKIT", "1")

    file_arg = str(dockerfile) if dockerfile else None
    emit_progress(on_progress, f"Building image '{tag}' from context {context_dir} ...")
    try:
        docker.build(
            str(context_dir),
            file=file_arg,
            tags=[tag],
            build_args=build_args or {},
            platforms=platforms or None,
            push=push,
        )
    except Exception as exc:
        raise ImageBuildError(f"Docker build failed: {exc}") from exc

    emit_progress(on_progress, f"Successfully built {tag}")
    return tag


def resolve_image_id(tag: str) -> str:
    """Return the immutable image ID currently bound to *tag*.

    Callers that build then later publish should resolve this right after the
    build call and push by ID rather than by *tag* — a concurrent job that
    rebuilds the same (daemon-global) tag name between the two steps cannot
    silently swap out what gets published.

    Raises:
        ContainerToolingUnavailableError: When python-on-whales is missing.
        ImageBuildError: When *tag* cannot be inspected (e.g. the build that
            produced it never loaded into the local daemon).
    """
    try:
        from python_on_whales import docker
    except ImportError as exc:
        raise ContainerToolingUnavailableError("resolving the built image ID") from exc

    try:
        return docker.image.inspect(tag).id
    except Exception as exc:
        raise ImageBuildError(f"Unable to resolve the image ID for '{tag}': {exc}") from exc


#: Distribution a supplied wheel must be, normalized per PEP 503.
_WHEEL_DISTRIBUTION = "nemo-platform"
_WHEEL_COPY_CHUNK_SIZE = 1024 * 1024


def wheel_contract_version(wheel: Path) -> str:
    """Return the nemo-platform version a wheel would install.

    The image's ``contract-version`` label and its content-addressable
    ``agent_id`` both describe the runtime inside it. Deriving them from the
    build host while installing a wheel would let an image advertise one version
    while running another, and let two genuinely different images share an id.

    Parsed from the filename, which PEP 427 fixes as
    ``{distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl``.
    """
    parts = wheel.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"not a PEP 427 wheel filename: {wheel.name}")
    distribution, version = parts[0], parts[1]
    if re.sub(r"[-_.]+", "-", distribution).lower() != _WHEEL_DISTRIBUTION:
        raise ValueError(
            f"wheel is {distribution!r}, not {_WHEEL_DISTRIBUTION!r}: {wheel.name}. "
            "Build it with `uv build --package nemo-platform --wheel`."
        )
    return version


def build_nat_agent_image(
    agent_config: Path,
    pyproject: Path | None = None,
    dockerfile: Path | None = None,
    tag: str | None = None,
    *,
    nat_version: str | None = None,
    base_image_url: str | None = None,
    base_image_tag: str | None = None,
    python_version: str | None = None,
    uv_version: str | None = None,
    allow_root: bool = False,
    sandbox_runtime: str | None = None,
    agent_version: str | None = None,
    agent_author: str | None = None,
    template_path: str | None = None,
    skip_validation: bool = False,
    generate_ignore: bool = True,
    platforms: list[str] | None = None,
    push: bool = False,
    on_progress: ProgressCallback | None = None,
) -> str:
    """High-level helper: validate, render (if needed), then build a NAT image.

    When *dockerfile* is ``None``, a Dockerfile is rendered via the template
    module and written into a temporary file inside the build context.

    Returns:
        The Docker image tag.
    """
    from nemo_agents_plugin.container.metadata import extract_agent_metadata
    from nemo_agents_plugin.container.template import render_dockerignore, render_nat_dockerfile, resolve_value
    from nemo_agents_plugin.container.validator import validate_agent_config

    if not skip_validation:
        result = validate_agent_config(agent_config)
        # Soft warnings (e.g. unknown workflow._type) are surfaced regardless
        # of overall validity so the operator can see them even when the
        # config is otherwise fine.  Hard errors still abort the build.
        for warn in result.warnings:
            emit_progress(on_progress, f"warning: {warn}")
        if not result.valid:
            raise AgentConfigValidationError(result.errors)

    if pyproject is not None and pyproject.exists():
        context_dir = pyproject.resolve().parent
    else:
        context_dir = agent_config.resolve().parent

    resolved_nat = resolve_value("nat_version", nat_version)
    resolved_python = resolve_value("python_version", python_version)
    resolved_base_url = resolve_value("base_image_url", base_image_url)
    resolved_base_tag = resolve_value("base_image_tag", base_image_tag)
    # Feed the resolved build environment into the agent_id hash so a rebuild
    # with a different toolchain (different NAT release, base image, or
    # Python) yields a distinct id, instead of silently re-tagging an
    # ABI-incompatible image with the same suffix.
    build_env_for_id = {
        "nat_version": resolved_nat,
        "python_version": resolved_python,
        "base_image_url": resolved_base_url,
        "base_image_tag": resolved_base_tag,
    }
    # Extract metadata once and thread it through both the tag computation
    # and the Dockerfile render — avoids three duplicate ``git`` subprocess
    # calls and two redundant yaml/toml parses per build.
    meta = extract_agent_metadata(
        agent_config,
        pyproject,
        agent_version=agent_version,
        agent_author=agent_author,
        build_env=build_env_for_id,
    )

    if tag is None:
        tag = _default_tag_from_meta(meta)

    build_args: dict[str, str] = {"NAT_VERSION": resolved_nat}
    if python_version:
        build_args["PYTHON_VERSION"] = python_version
    if base_image_url:
        build_args["BASE_IMAGE_URL"] = base_image_url
    if base_image_tag:
        build_args["BASE_IMAGE_TAG"] = base_image_tag

    if dockerfile is not None:
        return docker_build(
            context_dir=context_dir,
            dockerfile=dockerfile,
            tag=tag,
            build_args=build_args,
            platforms=platforms,
            push=push,
            on_progress=on_progress,
        )

    content = render_nat_dockerfile(
        agent_config,
        pyproject,
        base_image_url=base_image_url,
        base_image_tag=base_image_tag,
        python_version=python_version,
        nat_version=nat_version,
        uv_version=uv_version,
        allow_root=allow_root,
        sandbox_runtime=sandbox_runtime,
        agent_version=agent_version,
        agent_author=agent_author,
        template_path=template_path,
        metadata=meta,
    )

    tmp_dockerfile = context_dir / "Dockerfile.generated"
    # The temp Dockerfile is auto-cleaned in ``finally``; refuse to clobber a
    # pre-existing file by the same name, since the cleanup would delete the
    # user's file along with our own.
    if tmp_dockerfile.exists():
        raise ManagedFileConflictError(tmp_dockerfile)
    ignore_file: Path | None = None
    # Snapshot the pre-existing ``.dockerignore`` state so the ``finally``
    # cleanup only deletes files this run actually *created*.  Without this,
    # a committed-and-checked-in plugin-managed ``.dockerignore`` (sentinel
    # header on first line, intentionally kept in the repo) would be wiped:
    # ``render_dockerignore`` regenerates plugin-managed files in place and
    # returns the path, and the cleanup below would treat that returned path
    # as a transient artifact.  Two cases the cleanup must distinguish:
    #   (a) file did NOT exist before this run -> we just created it ->
    #       safe to unlink (the user never put it there).
    #   (b) file existed before this run -> the user committed/wrote it,
    #       even if plugin-managed -> must NOT unlink.
    ignore_path = context_dir / ".dockerignore"
    ignore_pre_existed = ignore_path.exists()
    try:
        tmp_dockerfile.write_text(content, encoding="utf-8")

        if generate_ignore:
            # ``render_dockerignore`` returns ``None`` when a user-owned file
            # was preserved — keeping ``ignore_file`` None means the
            # ``finally`` clause leaves it alone.
            ignore_file = render_dockerignore(context_dir)

        return docker_build(
            context_dir=context_dir,
            dockerfile=tmp_dockerfile,
            tag=tag,
            build_args=build_args,
            platforms=platforms,
            push=push,
            on_progress=on_progress,
        )
    finally:
        tmp_dockerfile.unlink(missing_ok=True)
        if ignore_file is not None and not ignore_pre_existed:
            ignore_file.unlink(missing_ok=True)


def _source_checkout_dist_dir() -> Path | None:
    """Locate ``dist`` in the checkout this package is running from, if it is one."""
    import importlib.util

    spec = importlib.util.find_spec("nemo_agents_plugin")
    locations = spec.submodule_search_locations if spec else None
    if not locations:
        return None
    for parent in Path(next(iter(locations))).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "plugins").is_dir():
            dist = parent / "dist"
            return dist if dist.is_dir() else None
    return None


def resolve_latest_wheel() -> Path:
    """Return the most recently built wheel in the source checkout's ``dist``."""
    from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

    dist = _source_checkout_dist_dir()
    if dist is None:
        raise ValueError(
            f"{WHEEL_ENV}={WHEEL_LATEST} needs a source checkout with a 'dist' directory, "
            "which this install is not. Point it at a wheel path instead."
        )
    wheels = sorted(dist.glob("nemo_platform-*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not wheels:
        raise ValueError(
            f"{WHEEL_ENV}={WHEEL_LATEST} found no nemo-platform wheels in {dist}. Build one with "
            "`uv build --package nemo-platform --wheel --out-dir dist`."
        )
    return wheels[0]


def build_fabric_agent_image(
    agent_config: Path,
    pyproject: Path | None = None,
    dockerfile: Path | None = None,
    tag: str | None = None,
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
    tag_namespace: str | None = None,
    skip_validation: bool = False,
    generate_ignore: bool = True,
    platforms: list[str] | None = None,
    push: bool = False,
    on_progress: ProgressCallback | None = None,
    wheel: Path | None = None,
) -> str:
    """Build a Fabric-backed NeMo agent image.

    *wheel* installs a local nemo-platform wheel instead of resolving the pinned
    contract version, which is the only way to package from a source checkout.

    This is intentionally separate from ``build_nat_agent_image`` so Fabric
    packaging can grow without inheriting NAT-specific args such as
    ``nat_version`` or ``NAT_CONFIG_FILE``.

    *tag_namespace* prefixes the final reference, supplied or derived, so callers
    that do not own the daemon cannot repoint a reference somebody else owns.
    """
    if pyproject is not None and pyproject.exists():
        context_dir = pyproject.resolve().parent
    else:
        context_dir = agent_config.resolve().parent

    if not skip_validation:
        import asyncio

        from nemo_agents_plugin.container.fabric_validator import validate_fabric_agent_package

        asyncio.run(validate_fabric_agent_package(agent_config, context_dir=context_dir))

    from nemo_agents_plugin.container.template import (
        get_contract_version,
        render_dockerignore,
        render_fabric_dockerfile,
        resolve_value,
    )

    resolved_base_url = resolve_value("base_image_url", base_image_url)
    resolved_base_tag = resolve_value("base_image_tag", base_image_tag)
    resolved_python = resolve_value("python_version", python_version)
    resolved_uv = resolve_value("uv_version", uv_version)

    from nemo_agents_plugin.container.metadata import NEMO_PLATFORM_AGENT_FRAMEWORK, extract_agent_metadata
    from nemo_agents_plugin.container.template import WHEEL_ENV, WHEEL_LATEST

    # The env var is the only caller-facing way in, so it reaches the platform
    # job as well as the CLI.
    if wheel is None:
        from_env = os.environ.get(WHEEL_ENV, "").strip()
        if from_env and dockerfile is not None:
            # Decided before resolving, so LATEST does not have to find a wheel
            # this build was never going to install.
            emit_progress(
                on_progress, f"warning: ignoring {WHEEL_ENV}={from_env}; --dockerfile decides what is installed"
            )
        # A real path wins over the sentinel, so a file named LATEST is still usable.
        elif from_env and not Path(from_env).exists() and from_env.upper() == WHEEL_LATEST:
            wheel = resolve_latest_wheel()
            emit_progress(on_progress, f"{WHEEL_ENV}={WHEEL_LATEST} resolved to {wheel.name}")
        elif from_env:
            wheel = Path(from_env)
    if wheel is not None and dockerfile is not None:
        # A supplied Dockerfile installs whatever it wants and never sees the
        # staged wheel, so the wheel must not speak for the version either.
        emit_progress(on_progress, f"warning: ignoring {WHEEL_ENV}={wheel}; --dockerfile decides what is installed")
        wheel = None
    if wheel is not None:
        if not wheel.is_file():
            raise ValueError(f"wheel not found: {wheel}")
        if wheel.suffix != ".whl":
            raise ValueError(f"not a wheel: {wheel}")

    # A wheel is what the image actually installs, so it — not the build host —
    # decides the version the image advertises and hashes into its id.
    contract_version = wheel_contract_version(wheel) if wheel is not None else get_contract_version()
    if wheel is not None and contract_version != get_contract_version():
        # Not fatal: the wheel is installed instead of the pin, so the pin never resolves.
        emit_progress(
            on_progress,
            f"warning: {wheel.name} builds {contract_version}, but this host runs {get_contract_version()}",
        )

    tmp_dockerfile = context_dir / "Dockerfile.generated"

    # Docker can only COPY from inside the context. For an external wheel,
    # compute the identity digest while staging it so the source is read once
    # and the digest describes the exact bytes placed in the build context.
    staged_wheel: Path | None = None
    wheel_was_staged = False
    wheel_sha256: str | None = None
    if wheel is not None:
        staged_wheel = context_dir / wheel.name
        if staged_wheel.exists():
            if not staged_wheel.samefile(wheel):
                raise ManagedFileConflictError(staged_wheel)
            with wheel.open("rb") as wheel_file:
                wheel_sha256 = hashlib.file_digest(wheel_file, "sha256").hexdigest()
        else:
            try:
                destination = open(staged_wheel, "xb")
            except FileExistsError as exc:
                raise ManagedFileConflictError(staged_wheel) from exc
            wheel_was_staged = True
            digest = hashlib.sha256()
            try:
                with destination, wheel.open("rb") as source:
                    while chunk := source.read(_WHEEL_COPY_CHUNK_SIZE):
                        destination.write(chunk)
                        digest.update(chunk)
            except BaseException:
                staged_wheel.unlink(missing_ok=True)
                raise
            wheel_sha256 = digest.hexdigest()
            emit_progress(on_progress, f"Staged {wheel.name} into the build context")

    build_env_for_id = {
        "agent_framework": NEMO_PLATFORM_AGENT_FRAMEWORK,
        "contract_version": contract_version,
        "base_image_url": resolved_base_url,
        "base_image_tag": resolved_base_tag,
        "python_version": resolved_python,
        "uv_version": resolved_uv,
    }
    if wheel_sha256 is not None:
        build_env_for_id["wheel_sha256"] = wheel_sha256
    try:
        meta = extract_agent_metadata(
            agent_config,
            pyproject,
            agent_version=agent_version,
            agent_author=agent_author,
            build_env=build_env_for_id,
        )
        if tag is None:
            tag = _default_tag_from_meta(meta)
        if tag_namespace:
            tag = f"{tag_namespace}/{tag}"

        build_args = {
            "BASE_IMAGE_URL": resolved_base_url,
            "BASE_IMAGE_TAG": resolved_base_tag,
            "PYTHON_VERSION": resolved_python,
        }

        if dockerfile is not None:
            return docker_build(
                context_dir=context_dir,
                dockerfile=dockerfile,
                tag=tag,
                build_args=build_args,
                platforms=platforms,
                push=push,
                on_progress=on_progress,
            )

        content = render_fabric_dockerfile(
            agent_config,
            pyproject,
            base_image_url=resolved_base_url,
            base_image_tag=resolved_base_tag,
            python_version=resolved_python,
            uv_version=resolved_uv,
            allow_root=allow_root,
            sandbox_runtime=sandbox_runtime,
            agent_version=agent_version,
            agent_author=agent_author,
            template_path=template_path,
            metadata=meta,
            wheel_filename=staged_wheel.name if staged_wheel else "",
            contract_version=contract_version,
        )
    except BaseException:
        if wheel_was_staged and staged_wheel is not None:
            staged_wheel.unlink(missing_ok=True)
        raise

    ignore_file: Path | None = None
    ignore_path = context_dir / ".dockerignore"
    ignore_pre_existed = ignore_path.exists()
    scoped_ignore: Path | None = None
    scoped_ignore_identity: tuple[int, int] | None = None
    tmp_dockerfile_identity: tuple[int, int] | None = None
    try:
        try:
            with tmp_dockerfile.open("x", encoding="utf-8") as tmp_dockerfile_file:
                created_stat = os.fstat(tmp_dockerfile_file.fileno())
                tmp_dockerfile_identity = (created_stat.st_dev, created_stat.st_ino)
                tmp_dockerfile_file.write(content)
        except FileExistsError as exc:
            raise ManagedFileConflictError(tmp_dockerfile) from exc

        if generate_ignore:
            ignore_file = render_dockerignore(context_dir)

        if staged_wheel is not None:
            # BuildKit prefers `<dockerfile>.dockerignore` over `.dockerignore`,
            # so the user's rules are carried over verbatim with the staged wheel
            # re-included after them. Guaranteeing the wheel arrives beats
            # detecting that it would not and reimplementing Docker's matcher.
            existing = ignore_path.read_text(encoding="utf-8") if ignore_path.exists() else ""
            scoped_ignore = context_dir / f"{tmp_dockerfile.name}.dockerignore"
            try:
                with scoped_ignore.open("x", encoding="utf-8") as scoped_ignore_file:
                    created_stat = os.fstat(scoped_ignore_file.fileno())
                    scoped_ignore_identity = (created_stat.st_dev, created_stat.st_ino)
                    scoped_ignore_file.write(f"{existing.rstrip()}\n!{staged_wheel.name}\n".lstrip())
            except FileExistsError as exc:
                raise ManagedFileConflictError(scoped_ignore) from exc

        return docker_build(
            context_dir=context_dir,
            dockerfile=tmp_dockerfile,
            tag=tag,
            build_args=build_args,
            platforms=platforms,
            push=push,
            on_progress=on_progress,
        )
    finally:
        if tmp_dockerfile_identity is not None:
            try:
                current_stat = tmp_dockerfile.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (current_stat.st_dev, current_stat.st_ino) == tmp_dockerfile_identity:
                    tmp_dockerfile.unlink(missing_ok=True)
        if scoped_ignore is not None and scoped_ignore_identity is not None:
            try:
                current_stat = scoped_ignore.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (current_stat.st_dev, current_stat.st_ino) == scoped_ignore_identity:
                    scoped_ignore.unlink(missing_ok=True)
        if wheel_was_staged and staged_wheel is not None:
            staged_wheel.unlink(missing_ok=True)
        if ignore_file is not None and not ignore_pre_existed:
            ignore_file.unlink(missing_ok=True)


def detect_agent_config_format(agent_config: Path) -> str:
    try:
        raw = agent_config.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read config file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Config file is not valid UTF-8: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in agent config {agent_config}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Agent config root must be a YAML mapping.")

    config_format = data.get("config_format", NAT_WORKFLOW_CONFIG_FORMAT)
    if config_format not in {NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT}:
        raise ValueError(f"Unsupported agent config format: {config_format!r}")
    return config_format


_TAG_NAME_INVALID = re.compile(r"[^a-z0-9._-]")
_TAG_VERSION_INVALID = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_image_name(raw: str) -> str:
    """Coerce *raw* into a valid Docker image name component.

    Docker reference syntax requires lowercase, ``[a-z0-9]`` plus the
    separators ``.`` ``-`` ``_``; uppercase letters (legal in PEP 621
    project names like ``"HelloWorld"``) and most punctuation are
    rejected by ``docker build`` with an opaque "invalid reference
    format" error.  Lowercase, replace illegal runs with ``-``, trim
    leading/trailing separators.
    """
    if not raw:
        return "agent"
    out = _TAG_NAME_INVALID.sub("-", raw.lower())
    out = out.strip("._-")
    return out or "agent"


def _sanitize_image_tag(raw: str) -> str:
    """Coerce *raw* into a valid Docker image tag.

    PEP 440 versions legitimately contain characters Docker rejects in
    tags — ``+`` (local-version separator), ``!`` (epoch), spaces — and
    the empty string is illegal.  Replace each illegal run with ``.``
    (Docker-valid and preserves the visual delimiter intent), strip
    leading non-alphanumeric characters (Docker requires the first byte
    to be ``[a-zA-Z0-9_]``), and bound the length at the 128-char
    reference limit.
    """
    if not raw:
        return "latest"
    out = _TAG_VERSION_INVALID.sub(".", raw)
    out = re.sub(r"^[^a-zA-Z0-9_]+", "", out)
    out = out[:128]
    return out or "latest"


def _default_tag_from_meta(meta: dict[str, str]) -> str:
    """Build a default image reference from precomputed metadata.

    Format: ``{agent_name}-{agent_id}:{agent_version}``, with both
    components sanitized for the Docker reference grammar.
    """
    name = _sanitize_image_name(meta["agent_name"])
    aid = meta["agent_id"]
    version = _sanitize_image_tag(meta["agent_version"])
    return f"{name}-{aid}:{version}"


def _default_tag(
    agent_config: Path,
    pyproject: Path | None = None,
    *,
    agent_version: str | None = None,
    agent_author: str | None = None,
) -> str:
    """Derive a default image tag as ``{agent_name}-{agent_id}:{agent_version}``.

    Uses :func:`~nemo_agents_plugin.container.metadata.extract_agent_metadata`
    to resolve the name, version, and content-addressable ID.  Kept as a
    thin wrapper around :func:`_default_tag_from_meta` so callers that
    don't already have a metadata dict (notably the unit tests) still
    have a single-argument entry point.
    """
    from nemo_agents_plugin.container.metadata import extract_agent_metadata

    meta = extract_agent_metadata(
        agent_config,
        pyproject,
        agent_version=agent_version,
        agent_author=agent_author,
    )
    return _default_tag_from_meta(meta)
