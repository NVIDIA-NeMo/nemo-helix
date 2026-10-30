# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Helix credential driver beside OpenShell on a private Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import signal
import socket
import stat
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import grpc
import httpx
from nemo_helix_plugin.secrets.client import AsyncSecretsClient

from .registry import BindingRegistry, require_private_directory
from .service import CredentialDriver
from .tickets import RegistrationTickets

# Defensive input bound, not an OpenShell/NeMo protocol limit. Allows ample room
# for a service-account JWT while avoiding an unbounded read of a mounted file.
MAX_CREDENTIAL_FILE_BYTES = 64 * 1024


def read_private_file(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Credential files must be private regular files owned by the driver user.")
        # Read one extra byte to distinguish an exact-limit file from overflow.
        value = file.read(MAX_CREDENTIAL_FILE_BYTES + 1)
        if not value or len(value) > MAX_CREDENTIAL_FILE_BYTES:
            raise ValueError("Invalid credential file size.")
        return value


class FileTokenProvider:
    """Authenticate the driver's Helix requests with a token reread from a mounted file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        pass

    async def get_access_token_async(self) -> str:
        token = (await asyncio.to_thread(read_private_file, self.path)).decode().strip()
        if not token:
            raise ValueError("Empty NeMo token file.")
        return token

    async def get_access_token_or_none_async(self) -> str:
        return await self.get_access_token_async()


@contextmanager
def socket_lease(socket_path: Path) -> Generator[None, None, None]:
    """Serialize driver starts and recover an owned socket left by a crash."""
    if not socket_path.is_absolute():
        raise ValueError("The credential-driver socket path must be absolute.")
    require_private_directory(socket_path.parent)
    descriptor = os.open(socket_path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Credential-driver lock must be a private regular file owned by this user.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another credential driver owns this socket.") from None
        if os.path.lexists(socket_path):
            info = socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("Refusing to replace a non-socket or another user's socket.")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(str(socket_path))
                except ConnectionRefusedError:
                    socket_path.unlink()
                else:
                    raise ValueError("An active listener already owns this socket.")
        yield
    finally:
        # Keep the lock file: unlinking it would allow two distinct lock inodes.
        os.close(descriptor)


@asynccontextmanager
async def running_server(driver: CredentialDriver, socket_path: Path) -> AsyncGenerator[grpc.aio.Server, None]:
    with socket_lease(socket_path):
        async with _running_server(driver, socket_path) as server:
            yield server


@asynccontextmanager
async def _running_server(driver: CredentialDriver, socket_path: Path) -> AsyncGenerator[grpc.aio.Server, None]:
    server = grpc.aio.server(
        options=(("grpc.max_receive_message_length", 65536), ("grpc.max_send_message_length", 1048576)),
        maximum_concurrent_rpcs=32,
    )
    driver.add_to_server(server)
    socket_identity = None
    try:
        if server.add_insecure_port(f"unix:{socket_path}") == 0:
            raise RuntimeError("Could not bind the credential-driver socket.")
        await server.start()
        os.chmod(socket_path, 0o600)
        socket_identity = socket_path.stat().st_ino
        yield server
    finally:
        await server.stop(grace=5)
        if socket_path.exists() and socket_path.stat().st_ino == socket_identity:
            socket_path.unlink()


async def serve(args: argparse.Namespace) -> None:
    url = urlsplit(args.helix_url)
    if url.scheme != "https" and not (
        args.allow_insecure_localhost and url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
    ):
        raise ValueError("Use an HTTPS NeMo endpoint, or explicitly allow HTTP localhost for development.")
    tickets = RegistrationTickets(read_private_file(args.registration_key_file), args.gateway_id)
    registry = BindingRegistry(args.state_path, tickets)
    auth = None
    if not args.auth_disabled:
        auth = FileTokenProvider(args.helix_token_file)
        await auth.get_access_token_async()
    async with AsyncSecretsClient(base_url=args.helix_url, auth=auth) as secrets_client:
        driver = CredentialDriver(
            registry, secrets_client, auth_enabled=not args.auth_disabled, cache_lifetime=args.cache_lifetime_seconds
        )
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stopped.set)
        try:
            async with running_server(driver, args.socket_path):
                await stopped.wait()
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental NeMo registration-ticket credential driver.")
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--gateway-id", required=True)
    parser.add_argument("--registration-key-file", type=Path, required=True)
    parser.add_argument("--helix-url", required=True)
    parser.add_argument(
        "--cache-lifetime-seconds",
        type=int,
        default=300,
        help="Resolved credential lifetime in OpenShell (1–3600 seconds; default: 300).",
    )
    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument("--helix-token-file", type=Path)
    auth.add_argument(
        "--auth-disabled", action="store_true", help="Connect to a Helix installation with auth disabled."
    )
    parser.add_argument("--allow-insecure-localhost", action="store_true")
    args = parser.parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
