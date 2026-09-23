# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task sandbox image build — the **BuildKit** approach (current default).

SINGLE-APPROACH layer: everything in this module is specific to building with
BuildKit. Given an uploaded tarball, it drives a `buildctl` build against the
daemon at `settings.buildkit_addr` over gRPC and pushes the image to
`settings.image_registry`. No docker-in-docker, no docker socket; the `buildctl`
client is baked into the API image (see the project Dockerfile).

The approach-agnostic job that *runs* a build and records its outcome lives in
`worker.py`. Additional build backends (e.g. a managed Cloud Build) would be
sibling modules here that the worker delegates to instead — only this file's
internals change per approach. See docs/internals/ARCHITECTURE.md § Container Build.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import tarfile
import tempfile
from contextlib import suppress
from pathlib import Path

from scaled_evals.api import artifacts
from scaled_evals.api.build.errors import BuildError
from scaled_evals.api.settings import settings

_BUILDKIT_READYZ_TIMEOUT_SECONDS = 3


def check_buildkit() -> None:
    """Confirm BuildKit is reachable at ``settings.buildkit_addr``.

    Uses the same TCP (or unix) address as task builds — not the default
    ``/run/buildkit/buildkitd.sock`` that bare ``buildctl`` would pick. Raises on
    failure; used by ``GET /v1/readyz``.
    """
    try:
        proc = subprocess.run(
            ["buildctl", "--addr", settings.buildkit_addr, "debug", "workers"],
            capture_output=True,
            timeout=_BUILDKIT_READYZ_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("buildctl timed out") from exc
    if proc.returncode == 0:
        return
    detail = proc.stderr.decode("utf-8", "replace").strip() or proc.stdout.decode("utf-8", "replace").strip()
    raise RuntimeError(detail or f"buildctl exited {proc.returncode}")


async def build_revision_image(task_id: str, revision: int, tarball_object_key: str) -> tuple[str, str]:
    """Build and push the sandbox image for one task revision.

    Downloads the revision tarball, extracts it, and runs `buildctl` against
    `settings.buildkit_addr`, pushing to
    `{settings.image_registry}/{task_id}:rev{revision}`.

    Returns `(image_ref, image_digest)` where the digest is the pushed manifest
    digest captured from buildctl's metadata file. Raises `BuildError` (with the
    captured build log) on any build failure.
    """
    image_ref = f"{settings.image_registry}/{task_id}:rev{revision}"

    with tempfile.TemporaryDirectory(prefix="se-build-") as tmp:
        tmp_path = Path(tmp)
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        tarball_path = tmp_path / "tarball.tar.gz"
        metadata_path = tmp_path / "metadata.json"

        # Download + extract the uploaded pack (Harbor task dirs + Dockerfile).
        _validate_tarball_size(tarball_object_key)
        artifacts.download_object(tarball_object_key, str(tarball_path))
        _extract_tarball(tarball_path, context_dir)

        await _run_buildctl(context_dir, image_ref, metadata_path, tmp_path)

        return image_ref, _read_digest(metadata_path)


def _validate_tarball_size(tarball_object_key: str) -> None:
    size_bytes = artifacts.object_size(tarball_object_key)
    max_size = settings.task_pack_max_size_bytes
    if size_bytes is None:
        raise BuildError("task pack object has no Content-Length; refusing to build")
    if size_bytes > max_size:
        raise BuildError(
            f"task pack object exceeds configured size limit ({size_bytes} bytes > {max_size} bytes); refusing to build"
        )


def _extract_tarball(tarball_path: Path, dest: Path) -> None:
    """Extract a gzip tarball into `dest`, rejecting unsafe member paths.

    Uses the 3.12 `data` extraction filter, which blocks absolute paths and
    `..` traversal out of `dest` (docs/API.md § Security: no `..` paths).
    """
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getmembers()
            if len(members) > settings.task_pack_max_members:
                raise BuildError(
                    f"task pack exceeds configured member limit ({len(members)} > {settings.task_pack_max_members})"
                )
            extracted_size = sum(member.size for member in members)
            if extracted_size > settings.task_pack_max_extracted_size_bytes:
                raise BuildError(
                    "task pack exceeds configured extracted-size limit "
                    f"({extracted_size} bytes > {settings.task_pack_max_extracted_size_bytes} bytes)"
                )
            tar.extractall(dest, members=members, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise BuildError(f"could not extract tarball: {exc}") from exc


def _buildctl_args(context_dir: Path, image_ref: str, metadata_path: Path) -> list[str]:
    """Assemble the ``buildctl build`` argv.

    Pins ``--opt platform`` (``settings.image_build_platform``) so the image
    targets the runtime cluster's architecture rather than the build host's: an
    arm64 build host (Apple Silicon under Colima) otherwise produces an image
    that ``exec format error``s on amd64 cluster nodes. An empty setting omits
    the pin (build for the host arch).
    """
    output = f"type=image,name={image_ref},push=true"
    if settings.registry_insecure:
        # Tell BuildKit to push over HTTP / skip TLS verify for this registry.
        output += ",registry.insecure=true"

    args = [
        "buildctl",
        "build",
        "--frontend",
        "dockerfile.v0",
        "--local",
        f"context={context_dir}",
        "--local",
        f"dockerfile={context_dir}",
        "--output",
        output,
        # Captures `containerimage.digest` (the pushed manifest digest) reliably,
        # rather than scraping it out of the human-readable build log.
        "--metadata-file",
        str(metadata_path),
    ]
    if settings.image_build_platform:
        args += ["--opt", f"platform={settings.image_build_platform}"]
    return args


async def _run_buildctl(context_dir: Path, image_ref: str, metadata_path: Path, work_dir: Path) -> None:
    """Invoke `buildctl build` as an async subprocess against BuildKit.

    Runs via `asyncio.create_subprocess_exec` so the build never blocks the
    event loop. stdout+stderr are merged and captured; on a non-zero exit the
    full log is raised as `BuildError`.
    """
    proc = await asyncio.create_subprocess_exec(
        *_buildctl_args(context_dir, image_ref, metadata_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=_buildctl_env(work_dir),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.buildkit_timeout_seconds)
    except TimeoutError as exc:
        with suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.communicate(), timeout=10)
        except TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.communicate()
        raise BuildError(f"buildctl timed out after {settings.buildkit_timeout_seconds}s building {image_ref}") from exc
    log = stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise BuildError(f"buildctl exited {proc.returncode} building {image_ref}:\n{log}")


def _buildctl_env(work_dir: Path) -> dict[str, str]:
    """Environment for buildctl: the BuildKit address, plus push auth if set.

    `BUILDKIT_HOST` selects the daemon (compose: tcp://buildkit:1234; cluster:
    the buildkitd Service). When registry credentials are configured, a transient
    docker config is written and pointed at via `DOCKER_CONFIG` so BuildKit can
    authenticate the push (e.g. NGC); locally the insecure registry needs none.
    """
    env = {**os.environ, "BUILDKIT_HOST": settings.buildkit_addr}
    config_dir = _write_docker_config(work_dir)
    if config_dir is not None:
        env["DOCKER_CONFIG"] = str(config_dir)
    return env


def _write_docker_config(work_dir: Path) -> Path | None:
    """Write a docker `config.json` with registry auth, returning its dir.

    Returns None when no credentials are configured (the local insecure
    registry), leaving buildctl to push anonymously.
    """
    if not settings.registry_username or not settings.registry_password:
        if not settings.task_image_registry_auth_file:
            return None
        return _copy_docker_config(work_dir, Path(settings.task_image_registry_auth_file))
    token = base64.b64encode(f"{settings.registry_username}:{settings.registry_password}".encode()).decode()
    config_dir = work_dir / "docker"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps({"auths": {settings.image_registry: {"auth": token}}}))
    return config_dir


def _copy_docker_config(work_dir: Path, auth_file: Path) -> Path:
    """Copy a mounted docker config secret into the transient buildctl config dir."""

    try:
        document = json.loads(auth_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not read registry auth file for buildkit push: {exc}") from exc
    if not isinstance(document.get("auths"), dict) or not document["auths"]:
        raise BuildError("registry auth file for buildkit push contains no auth entries")

    config_dir = work_dir / "docker"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(document))
    return config_dir


def _read_digest(metadata_path: Path) -> str:
    """Pull `containerimage.digest` out of buildctl's metadata file."""
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not read build metadata: {exc}") from exc
    digest = metadata.get("containerimage.digest")
    if not digest:
        raise BuildError(f"build metadata missing containerimage.digest: {metadata}")
    return digest
