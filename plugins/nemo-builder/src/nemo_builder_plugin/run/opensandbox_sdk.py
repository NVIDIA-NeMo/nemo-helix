# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox SDK behind :class:`~nemo_builder_plugin.run.opensandbox_sandbox.OpenSandboxApi`.

The only module that imports ``opensandbox``, the plugin's ``opensandbox`` extra, which only the step image installs.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta

from nemo_builder_plugin.run.sandbox import Mount
from nemo_builder_plugin.steps import OpenSandboxServer
from nhx_sandbox.egress import EgressPolicy
from nhx_sandbox.opensandbox_policy import to_opensandbox_policy
from opensandbox.config import ConnectionConfigSync
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.execd_sync import ExecutionHandlersSync
from opensandbox.models.sandboxes import PVC, NetworkPolicy, SandboxFilter, Volume
from opensandbox.sync import SandboxManagerSync, SandboxSync

#: Set on each create, so a create that fails can find what it left by its own label.
ATTEMPT_LABEL = "nhx.nvidia.com/sandbox-attempt"

API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"


class _Sandbox:
    def __init__(self, sandbox: SandboxSync) -> None:
        self._sandbox = sandbox

    @property
    def id(self) -> str:
        return self._sandbox.id

    def run(self, command: str, *, timeout_seconds: int, on_output: Callable[[str], None]) -> int | None:
        # Passed on as it comes, and not kept: a chatty build's output could exhaust this step's memory.
        handlers = ExecutionHandlersSync(
            on_stdout=lambda message: on_output(message.text),
            on_stderr=lambda message: on_output(message.text),
            skip_accumulation=True,
        )
        execution = self._sandbox.commands.run(
            command, opts=RunCommandOpts(timeout=timedelta(seconds=timeout_seconds)), handlers=handlers
        )
        return execution.exit_code

    def applied_egress(self) -> NetworkPolicy:
        return self._sandbox.get_egress_policy()

    def destroy(self) -> None:
        try:
            self._sandbox.kill()
        finally:
            self._sandbox.close()


class SdkApi:
    """One tenant of one OpenSandbox server, reached through the server's proxy."""

    def __init__(self, server: OpenSandboxServer, api_key: str) -> None:
        self._connection = ConnectionConfigSync(
            domain=server.domain,
            api_key=api_key,
            protocol=server.protocol,
            # Through the server, which checks the key and strips it before forwarding. A direct connection would
            # reach execd, which runs inside the sandbox and checks nothing.
            use_server_proxy=True,
            request_timeout=timedelta(minutes=5),
            # A multi-tenant server checks the key on proxied execd calls too, which SDK 0.1.16 sends it on only
            # from these headers. Only ever with use_server_proxy: directly, execd would get the key.
            headers={API_KEY_HEADER: api_key},
        )

    def create(
        self,
        *,
        image: str,
        entrypoint: list[str],
        env: Mapping[str, str],
        claim: str,
        mounts: list[Mount],
        labels: Mapping[str, str],
        cpu: str,
        memory: str,
        egress: EgressPolicy,
        ttl_seconds: int,
        ready_timeout_seconds: int,
    ) -> _Sandbox:
        attempt = uuid.uuid4().hex
        volumes = [
            Volume(
                name=f"work-{index}",
                pvc=PVC(claim_name=claim, create_if_not_exists=False),
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                sub_path=mount.sub_path,
            )
            for index, mount in enumerate(mounts)
        ]
        try:
            sandbox = SandboxSync.create(
                image,
                entrypoint=list(entrypoint),
                env=dict(env),
                volumes=volumes,
                metadata={**labels, ATTEMPT_LABEL: attempt},
                resource={"cpu": cpu, "memory": memory},
                # Set at create, the only time OpenSandbox lets a policy's default action be set.
                network_policy=NetworkPolicy.model_validate(to_opensandbox_policy(egress)),
                timeout=timedelta(seconds=ttl_seconds),
                ready_timeout=timedelta(seconds=ready_timeout_seconds),
                connection_config=self._connection,
            )
        except Exception:
            # The SDK tries to kill what a failed create left; this catches what it misses.
            for sandbox_id in self.labelled({ATTEMPT_LABEL: attempt}):
                self.kill(sandbox_id)
            raise
        return _Sandbox(sandbox)

    def labelled(self, labels: Mapping[str, str]) -> list[str]:
        ids: list[str] = []
        with self._manager() as manager:
            page = 1
            while True:
                # Unfiltered, and matched here: SDK 0.1.16 encodes a filter once more than the server decodes it,
                # so a filter on a key with a `/`, as all of ours have, matches nothing. 1.1.0 fixes it.
                found = manager.list_sandbox_infos(SandboxFilter(page=page))
                ids.extend(
                    info.id
                    for info in found.sandbox_infos
                    if all((info.metadata or {}).get(name) == value for name, value in labels.items())
                )
                if not found.pagination.has_next_page:
                    return ids
                page += 1

    def kill(self, sandbox_id: str) -> None:
        with self._manager() as manager:
            manager.kill_sandbox(sandbox_id)

    @contextmanager
    def _manager(self) -> Iterator[SandboxManagerSync]:
        manager = SandboxManagerSync.create(connection_config=self._connection)
        try:
            yield manager
        finally:
            manager.close()
