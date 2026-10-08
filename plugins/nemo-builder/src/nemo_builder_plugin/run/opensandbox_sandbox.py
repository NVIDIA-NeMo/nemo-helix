# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``opensandbox`` provider: each sandbox is an OpenSandbox sandbox, and each image one command in it.

The sandbox's pod comes from the OpenSandbox server's template, so hardening it is the operator's: see the README.
Its network is the provider's: each sandbox is created with a deny-by-default egress policy, and checked for it.
This module holds the provider's logic; the SDK calls are in ``opensandbox_sdk``, behind :class:`OpenSandboxApi`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from nemo_builder_plugin.run.sandbox import (
    BUILD_TIMEOUT_SECONDS,
    JOB_LABEL,
    SANDBOX_DEADLINE_SECONDS,
    SWEEP_TIMEOUT_SECONDS,
    Mount,
    clear_output,
    job_key,
    kaniko_command,
    mounts,
    sandbox_labels,
)
from nemo_builder_plugin.steps import SandboxGroup, SandboxSpec
from nemo_helix_plugin.log_utils import sanitize_for_log
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from nhx_sandbox.opensandbox_policy import verify_applied_egress

logger = logging.getLogger(__name__)

#: Keeps the sandbox up between commands. An omitted entrypoint would be OpenSandbox's `tail -f /dev/null`,
#: which the kaniko image doesn't have.
KEEPALIVE = ["/busybox/sh", "-c", f"/busybox/sleep {SANDBOX_DEADLINE_SECONDS}"]

#: Where execd, OpenSandbox's agent in the sandbox, writes each command's output. Its default, /tmp, is in what kaniko
#: snapshots: kaniko's own log would land in the image's layers, and a multi-stage build would delete it mid-stream.
#: /opt/opensandbox is execd's own mount, which kaniko ignores; kaniko gives a `RUN` the image's env, not this.
EXECD_TMPDIR = "/opt/opensandbox"

#: How long a create may wait for the sandbox. An admission refusal shows up only as this running out.
READY_TIMEOUT_SECONDS = 5 * 60


class RunningSandbox(Protocol):
    @property
    def id(self) -> str: ...

    def run(self, command: str, *, timeout_seconds: int, on_output: Callable[[str], None]) -> int | None:
        """Run ``command`` to its end, passing its output to ``on_output`` as it comes. Its exit code, if it has one."""
        ...

    def applied_egress(self) -> object:
        """The egress policy the sandbox reports, with ``default_action`` and its ``egress`` rules."""
        ...

    def destroy(self) -> None: ...


class OpenSandboxApi(Protocol):
    """The calls the provider makes, as one tenant of one OpenSandbox server."""

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
    ) -> RunningSandbox:
        """A running sandbox. If the create fails, whatever it left is deleted before this raises."""
        ...

    def labelled(self, labels: Mapping[str, str]) -> list[str]:
        """The IDs of the sandboxes carrying ``labels``."""
        ...

    def kill(self, sandbox_id: str) -> None: ...


class OpenSandboxProvider:
    """:class:`~nemo_builder_plugin.run.sandbox.SandboxProvider` creating OpenSandbox sandboxes."""

    def __init__(
        self, api: OpenSandboxApi, *, sandbox: SandboxSpec, workspace: str, job_id: str, job_sub_path: str
    ) -> None:
        self._api = api
        self._sandbox = sandbox
        self._workspace = workspace
        self._job_id = job_id
        self._job_sub_path = job_sub_path
        # Built here, on the trusted side, which shares the sandbox's resolver in a deployed cluster: its address is
        # kept out of the denied ranges.
        self._egress = build_egress_policy(EgressAllowlist(targets=tuple(sandbox.egress_allow)))

    def sweep(self) -> None:
        """Waits until the server no longer lists an earlier attempt's sandboxes: until then, their builds may still
        write to the outputs this attempt builds into."""
        labels = {JOB_LABEL: job_key(self._workspace, self._job_id)}
        leftovers = self._api.labelled(labels)
        if not leftovers:
            return
        logger.info("deleting %d sandbox(es) an earlier attempt left: %s", len(leftovers), _ids(leftovers))
        for sandbox_id in leftovers:
            try:
                self._api.kill(sandbox_id)
            except Exception:
                # It may have ended at its deadline since it was listed: whether it's gone is what counts.
                logger.warning("sandbox %s could not be deleted", sanitize_for_log(sandbox_id), exc_info=True)
        deadline = time.monotonic() + SWEEP_TIMEOUT_SECONDS
        while leftovers := self._api.labelled(labels):
            if time.monotonic() > deadline:
                raise RuntimeError(f"an earlier attempt's sandboxes are still there: {_ids(leftovers)}")
            time.sleep(1)

    def build(self, index: int, group: SandboxGroup) -> dict[str, int]:
        logger.info("creating sandbox %d for %d image(s)", index, len(group.images))
        sandbox = self._api.create(
            image=self._sandbox.image,
            entrypoint=KEEPALIVE,
            env={"TMPDIR": EXECD_TMPDIR},
            claim=self._sandbox.work_pvc,
            mounts=mounts(group, self._job_sub_path),
            labels=sandbox_labels(self._workspace, self._job_id),
            cpu=self._sandbox.cpu,
            memory=self._sandbox.memory,
            egress=self._egress,
            ttl_seconds=SANDBOX_DEADLINE_SECONDS,
            ready_timeout_seconds=READY_TIMEOUT_SECONDS,
        )
        logger.info("sandbox %d is %s", index, sanitize_for_log(sandbox.id))
        try:
            self._check_egress(sandbox)
            return self._build_images(sandbox, group)
        finally:
            try:
                sandbox.destroy()
            except Exception:
                # Its TTL ends it anyway; raising here would cost the images it built.
                logger.warning("sandbox %s could not be deleted", sanitize_for_log(sandbox.id), exc_info=True)

    def _check_egress(self, sandbox: RunningSandbox) -> None:
        """Raise, before anything runs in it, unless the sandbox reports a deny-by-default egress policy.

        A server that ignored the policy, or whose egress sidecar isn't configured, would otherwise give the
        sandbox an unrestricted network without a word. Missing rules are only logged: the sidecar reports a merged,
        re-serialized policy.
        """

        class Readback:  # verify_applied_egress awaits the read; this SDK's is synchronous.
            async def get_egress_policy(self) -> object:
                return sandbox.applied_egress()

        label = f"sandbox {sanitize_for_log(sandbox.id)}"
        asyncio.run(
            verify_applied_egress(Readback(), self._egress, mode="default_action", label=label, require_readback=True)
        )

    def _build_images(self, sandbox: RunningSandbox, group: SandboxGroup) -> dict[str, int]:
        deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
        results: dict[str, int] = {}
        for image in group.images:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                logger.error("image %s: the sandbox's hour ran out first", sanitize_for_log(image.image))
                continue
            logger.info("image %s:", sanitize_for_log(image.image))
            if sandbox.run(clear_output(image), timeout_seconds=min(remaining, 60), on_output=_log_output) != 0:
                logger.error("image %s: its output could not be emptied", sanitize_for_log(image.image))
                continue
            code = sandbox.run(kaniko_command(group, image), timeout_seconds=remaining, on_output=_log_output)
            if code is not None:
                results[image.image] = code
        return results


def _log_output(text: str) -> None:
    """Log a command's output as it comes, rather than hold a whole build's. Line by line: sanitizing it all at once
    would run it into one line."""
    for line in text.splitlines():
        logger.info("  %s", sanitize_for_log(line))


def _ids(sandbox_ids: list[str]) -> str:
    return sanitize_for_log(", ".join(sandbox_ids))
