# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``opensandbox`` provider: each sandbox is an OpenSandbox sandbox, and each image one command in it.

The sandbox's pod comes from the OpenSandbox server's template, so hardening it is the operator's: see the README.
This module holds the provider's logic; the SDK calls are in ``opensandbox_sdk``, behind :class:`OpenSandboxApi`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.run.sandbox import (
    BUILD_TIMEOUT_SECONDS,
    JOB_LABEL,
    SANDBOX_DEADLINE_SECONDS,
    Mount,
    clear_output,
    job_key,
    kaniko_command,
    mounts,
    sandbox_labels,
)
from nemo_builder_plugin.steps import SandboxGroup, SandboxSpec
from nemo_helix_plugin.log_utils import sanitize_for_log

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


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int | None
    output: list[str]


class RunningSandbox(Protocol):
    @property
    def id(self) -> str: ...

    def run(self, command: str, *, timeout_seconds: int) -> CommandResult: ...

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
        ttl_seconds: int,
        ready_timeout_seconds: int,
    ) -> RunningSandbox:
        """A running sandbox. If the create fails, whatever it left is deleted before this raises."""
        ...

    def kill_labelled(self, labels: Mapping[str, str]) -> list[str]:
        """Kill every sandbox carrying ``labels``, and return their IDs."""
        ...


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

    def sweep(self) -> None:
        killed = self._api.kill_labelled({JOB_LABEL: job_key(self._workspace, self._job_id)})
        if killed:
            logger.info("deleted %d sandbox(es) an earlier attempt left: %s", len(killed), ", ".join(killed))

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
            ttl_seconds=SANDBOX_DEADLINE_SECONDS,
            ready_timeout_seconds=READY_TIMEOUT_SECONDS,
        )
        logger.info("sandbox %d is %s", index, sanitize_for_log(sandbox.id))
        try:
            return self._build_images(sandbox, group)
        finally:
            try:
                sandbox.destroy()
            except Exception:
                # Its TTL ends it anyway; raising here would cost the images it built.
                logger.warning("sandbox %s could not be deleted", sanitize_for_log(sandbox.id), exc_info=True)

    def _build_images(self, sandbox: RunningSandbox, group: SandboxGroup) -> dict[str, int]:
        deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
        results: dict[str, int] = {}
        for image in group.images:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                logger.error("image %s: the sandbox's hour ran out first", sanitize_for_log(image.image))
                continue
            cleared = sandbox.run(clear_output(image), timeout_seconds=min(remaining, 60))
            if cleared.exit_code != 0:
                self._log(image.image, cleared)
                logger.error("image %s: its output could not be emptied", sanitize_for_log(image.image))
                continue
            built = sandbox.run(kaniko_command(group, image), timeout_seconds=remaining)
            self._log(image.image, built)
            if built.exit_code is not None:
                results[image.image] = built.exit_code
        return results

    @staticmethod
    def _log(image: str, result: CommandResult) -> None:
        # Line by line: sanitizing it all at once would run it into one line.
        logger.info("image %s, exit code %s:", sanitize_for_log(image), result.exit_code)
        for message in result.output:
            for line in message.splitlines():
                logger.info("  %s", sanitize_for_log(line))
