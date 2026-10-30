# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container initialization and a credential-driver protocol readiness probe."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import tempfile
from pathlib import Path

import grpc

from . import wire
from .registry import require_private_directory
from .server import MAX_CREDENTIAL_FILE_BYTES, read_private_file
from .service import CAPABILITY, DRIVER_NAME


class CredentialInitializationError(ValueError):
    """An initialization failure with a credential-free message for the operator."""


def _read_mount(path: Path) -> bytes:
    """Read an operator-supplied read-only bind mount; host UIDs may differ."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise CredentialInitializationError(
            f"Cannot open mounted {path.name}; provide a readable regular file, not a symlink."
        ) from None
    with os.fdopen(descriptor, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise CredentialInitializationError("Mounted credentials must be regular files.")
        value = file.read(MAX_CREDENTIAL_FILE_BYTES + 1)
    if not value or len(value) > MAX_CREDENTIAL_FILE_BYTES:
        raise CredentialInitializationError("Mounted credentials must contain between 1 byte and 64 KiB.")
    return value


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_credentials(source: Path, destination: Path, *, auth_disabled: bool = False) -> None:
    """Copy only operator-mounted credentials into private, container-owned files.

    Atomic token replacement supports rotation without restarting the driver.
    A different signing key needs an explicit migration; never silently rotate it.
    """
    key = _read_mount(source / "registration.key")
    if len(key) < 32:
        raise CredentialInitializationError("Provide a signing key of at least 32 bytes.")
    files = [("registration.key", key)]
    if not auth_disabled:
        token = _read_mount(source / "helix.token")
        try:
            valid_token = bool(token.decode().strip())
        except UnicodeDecodeError:
            valid_token = False
        if not valid_token:
            raise CredentialInitializationError("Provide a nonempty UTF-8 token.")
        files.append(("helix.token", token))
    require_private_directory(destination)
    key_path = destination / "registration.key"
    if os.path.lexists(key_path) and read_private_file(key_path) != key:
        raise CredentialInitializationError(
            "Registration key differs from the existing key; explicit migration is required."
        )
    for name, value in files:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-", dir=destination)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(value)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination / name)
            _sync_directory(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if auth_disabled:
        (destination / "helix.token").unlink(missing_ok=True)
        _sync_directory(destination)


async def check_ready(socket_path: Path) -> None:
    """Check an actual RPC, without reading a service secret or querying Helix."""
    async with grpc.aio.insecure_channel(f"unix:{socket_path}") as channel:
        request_name, response_name = wire.METHODS["GetCapabilities"]
        call = channel.unary_unary(
            f"/{wire.SERVICE}/GetCapabilities",
            request_serializer=wire.message_type(request_name).SerializeToString,
            response_deserializer=wire.message_type(response_name).FromString,
        )
        response = wire.data(
            await call(
                wire.message(
                    request_name,
                    {
                        "gateway": {
                            "protocol_version": {"major": 1},
                            "supported_capabilities": [CAPABILITY],
                            "required_capabilities": [CAPABILITY],
                        }
                    },
                ),
                timeout=3,
            )
        )
    if response.get("driver_name") != DRIVER_NAME or not response.get("supports_expires_at"):
        raise ValueError("Unexpected credential-driver capabilities.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Stage operator-mounted credentials into private files.")
    initialize.add_argument("--source", type=Path, required=True)
    initialize.add_argument("--destination", type=Path, required=True)
    initialize.add_argument("--auth-disabled", action="store_true")
    health = commands.add_parser("healthcheck", help="Check local credential-driver protocol readiness.")
    health.add_argument("--socket-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            prepare_credentials(args.source, args.destination, auth_disabled=args.auth_disabled)
        else:
            asyncio.run(check_ready(args.socket_path))
    except CredentialInitializationError as exc:
        parser.exit(1, f"Credential-driver initialization failed: {exc}\n")
    except Exception:
        # Credential-bearing errors must not land in Docker healthcheck/log output.
        parser.exit(1, "Credential-driver initialization or readiness check failed.\n")


if __name__ == "__main__":
    main()
