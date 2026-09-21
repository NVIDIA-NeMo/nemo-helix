# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A :class:`SandboxProvider` backed by a ``nemo-sandboxed-gym`` episode broker.

Where the Docker and Compose providers drive a container runtime on the host, this one is an HTTP
client. The broker holds the backend credential and enforces policy -- approved images, TTL and
concurrency caps, egress -- so this process never holds one; a local PoC found that putting stock
OpenSandbox credentials on the untrusted side lets ``provider_options`` escalate to socket or host
mounts, which is what the broker exists to prevent (sandboxed-GRPO RFC §6.7).

That split is also why this module imports only ``sandboxed_gym.wire``: the request and response
models, so client and server cannot drift. The broker, its backends and NeMo-Gym all live on the
far side of the HTTP boundary, in whatever job runs the broker.

Deployment note: nothing in this repository starts a broker yet. Customizer runs one as a Ray actor
inside the training pod, which does not transfer to an eval job. Until that is settled, point
``broker_url`` at a broker you started.
"""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from typing import Any

import httpx
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import (
    SANDBOX_RUNTIME_RETURN_CODE,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from sandboxed_gym.wire import (
    BROKER_AUTH_HEADER,
    BROKER_PROTOCOL_VERSION,
    EpisodeCreateRequest,
    EpisodeCreateResponse,
    EpisodeDirDownloadResponse,
    EpisodeDirUploadRequest,
    EpisodeExecRequest,
    EpisodeExecResponse,
    EpisodeFileDownloadResponse,
    EpisodeFileUploadRequest,
    EpisodeResources,
    EpisodeStatusResponse,
)


class UnsupportedSandboxOperationError(RuntimeError):
    """The broker cannot honour the request as written and will not silently downgrade it."""


class BrokerSandboxError(RuntimeError):
    """The broker refused or failed a request."""


def _episode_id(handle: SandboxHandle) -> str:
    if not isinstance(handle.raw, str):
        raise BrokerSandboxError(f"handle was not created by this provider: {handle!r}")
    return handle.raw


def _archive_dir(source_dir: Path) -> bytes:
    """Pack ``source_dir``'s *contents* into a gzipped tar, members relative to the directory.

    Contents rather than the directory itself, so unpacking into a differently-named target does
    not nest the tree one level deeper than the caller asked for.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for entry in sorted(source_dir.rglob("*")):
            tar.add(entry, arcname=str(entry.relative_to(source_dir)), recursive=False)
    return buffer.getvalue()


def _extract_archive(archive: bytes, target_dir: Path) -> None:
    """Unpack into ``target_dir``, refusing any member that escapes it.

    The archive is assembled inside the episode, which runs untrusted code, so a member named
    ``../../etc/passwd`` is an expected thing to receive rather than a hypothetical.

    Two passes, so a rejected archive leaves nothing partial behind, then a per-member
    ``extract()`` rather than a bulk ``extractall()`` -- CodeQL's tar-slip tracking does not follow
    a separate validation loop into the bulk call, and `filter="data"` is a second line of defence
    rather than the only one.

    This mirrors ``nemo_platform_plugin.jobs.archive.safe_extract_tar`` rather than importing it:
    that lives in a platform-internal package, and this SDK is published standalone.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = target_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            destination = (target_dir / member.name).resolve()
            if destination != resolved_root and resolved_root not in destination.parents:
                raise BrokerSandboxError(f"archive member escapes {target_dir}: {member.name}")
            if member.issym() or member.islnk():
                raise BrokerSandboxError(f"archive member is a link, which is not extracted: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise BrokerSandboxError(f"archive member is a special file, which is not extracted: {member.name}")
        for member in members:
            tar.extract(member, path=target_dir, filter="data")


class BrokerSandboxProvider:
    """Provision episode sandboxes through a ``nemo-sandboxed-gym`` broker over HTTP."""

    name = "broker"

    def __init__(
        self,
        *,
        broker_url: str,
        token: str,
        default_timeout_s: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Construct a provider.

        Args:
            broker_url: Base URL of the broker, e.g. the cluster Service DNS name it advertises.
            token: Job-scoped broker token. Presented on every request; the broker compares it
                with :func:`hmac.compare_digest`.
            default_timeout_s: Request timeout when a call does not supply its own.
            client: Pre-built HTTP client, for tests. When given, this provider does not own it and
                :meth:`aclose` leaves it open.
        """
        self._base_url = broker_url.rstrip("/")
        self._default_timeout_s = default_timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            headers={BROKER_AUTH_HEADER: token},
            timeout=default_timeout_s,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise self._to_error(response)
        return response

    def _to_error(self, response: httpx.Response) -> Exception:
        """Turn a broker error body into the closest local exception.

        ``501`` is the broker saying it will not silently downgrade the request -- exec as a user
        it cannot provide, or an operation this backend lacks -- and is worth distinguishing from a
        transport failure, because the caller can act on it.
        """
        try:
            body = response.json()
            detail = body.get("message") or body.get("detail") or response.text
        except ValueError:
            detail = response.text
        message = f"broker {response.request.method} {response.request.url.path} -> {response.status_code}: {detail}"
        if response.status_code == 501:
            return UnsupportedSandboxOperationError(message)
        return BrokerSandboxError(message)

    async def health(self) -> str:
        """Return the broker's protocol version, raising if it cannot serve this client.

        Directory transfer arrived in ``"2"``; against a ``"1"`` broker those endpoints 404 partway
        through a run, which is a worse failure than refusing at construction.
        """
        response = await self._request("GET", "/health")
        version = str(response.json().get("protocol_version", ""))
        if version != BROKER_PROTOCOL_VERSION:
            raise BrokerSandboxError(
                f"broker speaks protocol {version!r}, this client speaks {BROKER_PROTOCOL_VERSION!r}"
            )
        return version

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.image is None:
            # The broker evaluates its approved-image policy against this field, so a request
            # without one is a request it cannot evaluate.
            raise SandboxCreateError("spec.image is required for the broker provider")

        request = EpisodeCreateRequest(
            image=spec.image,
            ttl_s=float(spec.ttl_s) if spec.ttl_s is not None else None,
            workdir=spec.workdir,
            env=dict(spec.env),
            metadata=dict(spec.metadata),
            resources=EpisodeResources(
                cpu=spec.resources.cpu,
                memory_mib=spec.resources.memory_mib,
                disk_gib=spec.resources.disk_gib,
                gpu=spec.resources.gpu,
                gpu_type=spec.resources.gpu_type,
            ),
            files_b64={path: base64.b64encode(content.encode()).decode() for path, content in spec.files.items()},
            provider_options=dict(spec.provider_options),
        )
        try:
            response = await self._request("POST", "/episodes", json=request.model_dump(mode="json"))
        except BrokerSandboxError as error:
            raise SandboxCreateError(str(error)) from error

        created = EpisodeCreateResponse.model_validate(response.json())
        return SandboxHandle(sandbox_id=created.episode_id, provider_name=self.name, raw=created.episode_id)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        if stdin is not None:
            # Not a gap in this provider: `stdin` exists nowhere below the wire -- not on the
            # broker's backend contract, not on NeMo-Gym's provider `exec` -- so there is nothing
            # to forward it to. Synthesizing `cmd < file` would change the command's meaning
            # silently, which is worse than refusing.
            raise UnsupportedSandboxOperationError(
                "the broker's exec contract has no stdin; write the input to a file with "
                "upload_file and redirect from it in the command instead"
            )

        request = EpisodeExecRequest(
            command=command,
            cwd=cwd,
            env=dict(env) if env else None,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
        )
        try:
            response = await self._request(
                "POST",
                f"/episodes/{_episode_id(handle)}/exec",
                json=request.model_dump(mode="json"),
            )
        except httpx.TimeoutException as error:
            # The seam's contract: exec reports command failure through the result, and a sandbox
            # runtime failure through `error_type`. Neither raises.
            return SandboxExecResult(
                stdout=None, stderr=str(error), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
            )

        result = EpisodeExecResponse.model_validate(response.json())
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            error_type=result.error_type,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        request = EpisodeFileUploadRequest(
            path=target_path,
            content_b64=base64.b64encode(source_path.read_bytes()).decode(),
        )
        await self._request("PUT", f"/episodes/{_episode_id(handle)}/files", json=request.model_dump(mode="json"))

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        response = await self._request("GET", f"/episodes/{_episode_id(handle)}/files", params={"path": source_path})
        payload = EpisodeFileDownloadResponse.model_validate(response.json())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(base64.b64decode(payload.content_b64))

    async def upload_dir(self, handle: SandboxHandle, source_dir: Path, target_dir: str) -> None:
        request = EpisodeDirUploadRequest(
            path=target_dir,
            archive_b64=base64.b64encode(_archive_dir(source_dir)).decode(),
        )
        await self._request("PUT", f"/episodes/{_episode_id(handle)}/dirs", json=request.model_dump(mode="json"))

    async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
        response = await self._request("GET", f"/episodes/{_episode_id(handle)}/dirs", params={"path": source_dir})
        payload = EpisodeDirDownloadResponse.model_validate(response.json())
        _extract_archive(base64.b64decode(payload.archive_b64), target_dir)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            response = await self._request("GET", f"/episodes/{_episode_id(handle)}")
        except BrokerSandboxError:
            # A broker that no longer knows the episode has, from the caller's side, a gone
            # sandbox -- which the seam models as a status rather than an exception.
            return SandboxStatus.UNKNOWN
        payload = EpisodeStatusResponse.model_validate(response.json())
        try:
            return SandboxStatus(payload.status.value)
        except ValueError:
            return SandboxStatus.UNKNOWN

    async def close(self, handle: SandboxHandle) -> None:
        try:
            await self._request("DELETE", f"/episodes/{_episode_id(handle)}")
        except BrokerSandboxError:
            # Close is idempotent for callers: an already-reaped episode is the desired end state.
            return

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
