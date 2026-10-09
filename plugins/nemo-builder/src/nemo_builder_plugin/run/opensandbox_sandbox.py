# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``opensandbox`` provider: each image is one command in an OpenSandbox sandbox of its own.

The sandbox's pod comes from the OpenSandbox server's template, so hardening it is mostly the operator's: see the
README. Two things are the provider's. The sandbox's network: each is created with a deny-by-default egress policy,
and checked for it. And kaniko's capabilities, which it drops to the plain pod's before kaniko runs.
This module holds the provider's logic; the SDK calls are in ``opensandbox_sdk``, behind :class:`OpenSandboxApi`.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.run.sandbox import (
    BUILD_TIMEOUT_SECONDS,
    JOB_LABEL,
    KANIKO_FEATURE_FLAGS,
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
from nhx_sandbox.opensandbox_policy import EgressVerificationError, verify_applied_egress

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

#: Runs kaniko with only the plain pod's five capabilities, or not at all. OpenSandbox replaces the template's
#: capability drop list when it adds the egress sidecar, so the sandbox keeps the container runtime's defaults, NET_RAW
#: among them, with which a `RUN` could get past the sidecar's firewall. Built from `docker/builder/nhx-dropcaps`.
DROPCAPS = "/kaniko/nhx-dropcaps"

#: kaniko's pinned feature flags, as assignments before its command: execd runs a command through a shell, and
#: nothing says it passes the sandbox's env on to it. kaniko gives a `RUN` the image's env, not these.
_FLAGS = " ".join(f"{name}={shlex.quote(value)}" for name, value in KANIKO_FEATURE_FLAGS.items())

#: How the egress sidecar must enforce the policy. In `dns`, the server's default, it filters names only, so a sandbox
#: can reach any address it doesn't look up, the cluster's included.
ENFORCEMENT_MODE = "dns+nft"


@dataclass(frozen=True, slots=True)
class AppliedEgress:
    """What a sandbox's egress sidecar reports."""

    enforcement_mode: str | None
    #: With ``default_action`` and its ``egress`` rules.
    policy: object


class RunningSandbox(Protocol):
    @property
    def id(self) -> str: ...

    def run(self, command: str, *, timeout_seconds: int, on_output: Callable[[str], None]) -> int | None:
        """Run ``command`` to its end, passing its output to ``on_output`` as it comes. Its exit code, if it has one."""
        ...

    def applied_egress(self) -> AppliedEgress: ...

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
        # Built here, on the trusted side. No resolver is kept out of the denied ranges: the sidecar reaches its own
        # upstream with packets its firewall lets through, so the sandbox needs no way to a resolver of its own. And
        # this step's resolver can be the metadata server, as on GKE with Cloud DNS.
        self._egress = build_egress_policy(EgressAllowlist(targets=tuple(sandbox.egress_allow), resolver_addresses=()))

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
        """Each image in a sandbox of its own: in a shared one, a later image's build would start through a shell,
        and an ``nhx-dropcaps``, that an earlier image's `RUN` could have replaced. The group's images share its hour.
        """
        deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
        results: dict[str, int] = {}
        for image in group.images:
            try:
                code = self._build_alone(index, group.model_copy(update={"images": [image]}), deadline)
            except Exception:
                # Fails this image only, not the ones the group has built.
                logger.exception("image %s: its sandbox failed", sanitize_for_log(image.image))
                continue
            if code is not None:
                results[image.image] = code
        return results

    def _build_alone(self, index: int, group: SandboxGroup, deadline: float) -> int | None:
        """Build a group of one image in a sandbox of its own, and delete it. kaniko's exit code, if it ran."""
        (image,) = group.images
        name = sanitize_for_log(image.image)
        if time.monotonic() >= deadline:
            logger.error("image %s: group %d's hour ran out first", name, index)
            return None
        logger.info("image %s: creating its sandbox, in group %d", name, index)
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
        logger.info("image %s: sandbox %s", name, sanitize_for_log(sandbox.id))
        try:
            self._check_egress(sandbox)
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                logger.error("image %s: group %d's hour ran out first", name, index)
                return None
            if sandbox.run(clear_output(image), timeout_seconds=min(remaining, 60), on_output=_log_output) != 0:
                logger.error("image %s: its output could not be emptied", name)
                return None
            return sandbox.run(
                f"{_FLAGS} {DROPCAPS} {kaniko_command(group, image)}", timeout_seconds=remaining, on_output=_log_output
            )
        finally:
            try:
                sandbox.destroy()
            except Exception:
                # Its TTL ends it anyway; raising here would cost the image it built.
                logger.warning("sandbox %s could not be deleted", sanitize_for_log(sandbox.id), exc_info=True)

    def _check_egress(self, sandbox: RunningSandbox) -> None:
        """Raise, before anything runs in it, unless the sandbox's egress sidecar enforces the policy it was created with.

        A server that ignored the policy, or whose sidecar isn't configured or filters names only, would otherwise give
        the sandbox an unrestricted network without a word. The sidecar reports the policy it was given, as it was
        given, so a rule it doesn't report is one it doesn't enforce.
        """
        label = f"sandbox {sanitize_for_log(sandbox.id)}"
        applied = sandbox.applied_egress()
        if applied.enforcement_mode != ENFORCEMENT_MODE:
            raise EgressVerificationError(
                f"{label}'s egress sidecar enforces {sanitize_for_log(repr(applied.enforcement_mode))}, not "
                f"{ENFORCEMENT_MODE!r}: it doesn't filter addresses"
            )

        class Readback:  # verify_applied_egress awaits the read; this SDK's is synchronous.
            async def get_egress_policy(self) -> object:
                return applied.policy

        asyncio.run(verify_applied_egress(Readback(), self._egress, mode="strict", label=label, require_readback=True))


def _log_output(text: str) -> None:
    """Log a command's output as it comes, rather than hold a whole build's. Line by line: sanitizing it all at once
    would run it into one line."""
    for line in text.splitlines():
        logger.info("  %s", sanitize_for_log(line))


def _ids(sandbox_ids: list[str]) -> str:
    return sanitize_for_log(", ".join(sandbox_ids))
