# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store OpenShell's internal credentials in Helix Secrets and recover interrupted writes."""

from __future__ import annotations

import asyncio
import fcntl
import hmac
import os
import stat
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from pydantic import SecretStr

from .registry import OWNED_HANDLE_PREFIX as OWNED_HANDLE_PREFIX
from .registry import BindingRegistry, OwnedCredential, ProviderOwner
from .tickets import RegistrationDenied


class OwnedCredentials:
    """Keep gateway-owned values separate from references to users' secrets.

    A stored handle owns one immutable, randomly named secret. Registry records
    survive deletion so retries cannot recreate a revoked credential.
    """

    def __init__(self, registry: BindingRegistry, secrets_client: AsyncSecretsClient, workspace: str) -> None:
        self.registry = registry
        self.secrets_client = secrets_client.with_options(timeout=15, retry=RetryPolicy(max_retries=0))
        self.workspace = workspace

    def validate(self, owner: ProviderOwner, handle: str) -> None:
        binding = self.registry.get_owned(owner, handle, active=True)
        self._check_workspace(binding)

    def _check_workspace(self, binding: OwnedCredential) -> None:
        if binding.workspace != self.workspace:
            raise RegistrationDenied("Owned credential belongs to another Secrets workspace.")

    @asynccontextmanager
    async def _write_lock(self) -> AsyncGenerator[None]:
        descriptor = os.open(
            self.registry.path.with_suffix(self.registry.path.suffix + ".owned-lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError("Owned credential lock must be a private regular file.")
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            yield
        finally:
            os.close(descriptor)

    async def _read(self, binding: OwnedCredential) -> str:
        self._check_workspace(binding)
        response = (await self.secrets_client.access_secret(workspace=binding.workspace, name=binding.name)).data()
        if (response.workspace, response.name) != (binding.workspace, binding.name) or not hmac.compare_digest(
            await asyncio.to_thread(self.registry.owned_digest, response.value), binding.value_digest
        ):
            raise RegistrationDenied("Stored value does not match this owned credential.")
        return response.value

    async def store(
        self,
        owner: ProviderOwner,
        value: SecretStr,
        *,
        object_id: str = "",
        existing_handle: str = "",
    ) -> str:
        async with asyncio.timeout(15), self._write_lock():
            binding = await asyncio.to_thread(
                self.registry.reserve_owned,
                owner,
                value,
                workspace=self.workspace,
                object_id=object_id,
                existing_handle=existing_handle,
            )
            if binding.state == "active":
                return binding.handle
            was_uncertain = await asyncio.to_thread(self.registry.begin_owned_create, owner, binding.handle)
            try:
                response = (
                    await self.secrets_client.create_secret(
                        workspace=binding.workspace,
                        body=HelixSecretCreateRequest(name=binding.name, value=value),
                    )
                ).data()
                if (response.workspace, response.name) != (binding.workspace, binding.name):
                    raise RegistrationDenied("Secrets returned a different owned credential.")
            except ConflictError:
                await self._read(binding)
            await asyncio.to_thread(
                self.registry.activate_owned, owner, binding.handle, retain_uncertainty=was_uncertain
            )
            return binding.handle

    async def resolve(self, owner: ProviderOwner, handle: str) -> str:
        async with asyncio.timeout(15):
            binding = await asyncio.to_thread(self.registry.get_owned, owner, handle, active=True)
            value = await self._read(binding)
            await asyncio.to_thread(self.validate, owner, handle)
            return value

    async def _delete(self, binding: OwnedCredential) -> None:
        self._check_workspace(binding)
        if binding.state == "deleted":
            return
        try:
            await self.secrets_client.delete_secret(workspace=binding.workspace, name=binding.name)
        except NotFoundError:
            pass
        await asyncio.to_thread(self.registry.finish_owned_delete, binding.owner, binding.handle)

    async def delete(self, owner: ProviderOwner, handle: str) -> None:
        async with asyncio.timeout(15), self._write_lock():
            binding = await asyncio.to_thread(self.registry.revoke_owned, owner, handle)
            if binding is not None:
                await self._delete(binding)

    async def reconcile(self) -> None:
        failures = 0
        for pending in await asyncio.to_thread(self.registry.pending_owned):
            try:
                async with asyncio.timeout(15), self._write_lock():
                    binding = await asyncio.to_thread(self.registry.get_owned, pending.owner, pending.handle)
                    self._check_workspace(binding)
                    if binding.state == "deleting":
                        await self._delete(binding)
                    elif binding.state == "pending":
                        try:
                            await self._read(binding)
                        except NotFoundError:
                            continue  # OpenShell must retry creation with the original value.
                        await asyncio.to_thread(
                            self.registry.activate_owned,
                            binding.owner,
                            binding.handle,
                            retain_uncertainty=binding.create_uncertain,
                        )
            except Exception:
                failures += 1
        if failures:
            raise RuntimeError(f"Owned credential recovery failed for {failures} records.")
