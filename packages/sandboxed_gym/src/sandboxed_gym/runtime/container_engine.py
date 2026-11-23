# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rootless container engine for the sandboxed Gym host: Kata check, root-only setup, then drop privileges.

The ``podman`` variant of the Gym host image sets ``NHX_GYM_HOST_CONTAINER_ENGINE`` so resource servers
can run ``docker`` and ``docker compose`` against rootless Podman. Its setup needs root with SYS_ADMIN,
granted by OpenSandbox's ``bootstrap.execd.isolation`` extension. That is only acceptable inside a Kata
VM, where an escape reaches the sandbox's own guest kernel rather than the node's, so the host checks
for one first. After setup the host becomes the unprivileged container user, so neither the Gym servers
nor anything an agent drives through them keeps root or SYS_ADMIN.
"""

import os
import pwd
import subprocess
from collections.abc import Callable, MutableMapping
from pathlib import Path

ENGINE_ENV = "NHX_GYM_HOST_CONTAINER_ENGINE"
ROOTLESS_PODMAN = "rootless-podman"
SETUP_SCRIPT = "/usr/local/libexec/nhx/rootless-podman-up"
CONTAINER_USER = "docker-user"
SETUP_TIMEOUT_S = 300.0
SETUP_LOG = "/tmp/rootless-podman-up.log"

# Kata boots its guest kernel with the agent's systemd target; a runc container sees the node's own
# command line instead. Runtimes that boot without systemd still pass `agent.*` parameters.
_KATA_TARGET = "systemd.unit=kata-containers.target"


def is_kata_guest(kernel_cmdline: str) -> bool:
    """Whether a kernel command line belongs to a Kata Containers guest VM."""
    tokens = kernel_cmdline.split()
    return _KATA_TARGET in tokens or any(token.startswith("agent.") for token in tokens)


def _effective_capabilities(status: str) -> int:
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            return int(line.split()[1], 16)
    raise RuntimeError("CapEff missing from /proc/self/status")


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    os.chown(root, uid, gid)
    for directory, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.chown(os.path.join(directory, name), uid, gid, follow_symlinks=False)


def prepare_container_engine(
    environ: MutableMapping[str, str] = os.environ,
    *,
    read_text: Callable[[str], str] = lambda path: Path(path).read_text(),
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Set up the image's container engine and drop to its user. Returns whether it did anything.

    A no-op unless the image asks for an engine. Raises when the sandbox isn't a Kata VM, when setup
    fails, or when the process still holds capabilities after dropping privileges; the caller reports
    that as a bootstrap failure instead of running Gym with a half-configured or privileged host.
    """
    engine = environ.get(ENGINE_ENV, "")
    if not engine:
        return False
    if engine != ROOTLESS_PODMAN:
        raise RuntimeError(f"{ENGINE_ENV}={engine!r} is not supported; expected {ROOTLESS_PODMAN!r}")
    if os.geteuid() != 0:
        raise RuntimeError(f"{ENGINE_ENV} needs the Gym host to start as root to run its setup")
    if not is_kata_guest(read_text("/proc/cmdline")):
        raise RuntimeError(
            f"{ENGINE_ENV} needs a Kata sandbox: its setup runs with SYS_ADMIN, which is only contained "
            "by a VM. Use a Kata runtime class for the Gym host, or the image without a container engine."
        )

    # Output goes to a file, not a pipe: the setup leaves a long-lived service running, and a pipe
    # it inherited would never reach EOF.
    with open(SETUP_LOG, "w") as log:
        result = run(
            [SETUP_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=SETUP_TIMEOUT_S,
            check=False,
        )
    if result.returncode != 0:
        tail = read_text(SETUP_LOG).strip().splitlines()[-20:]
        raise RuntimeError(f"{SETUP_SCRIPT} exited {result.returncode}: " + " | ".join(tail))

    user = pwd.getpwnam(CONTAINER_USER)
    # The work directory is the only place the host writes; everything else it needs is read-only.
    _chown_tree(Path(environ.get("NHX_WORK_PATH", "/job/work")), user.pw_uid, user.pw_gid)
    os.setgroups([])
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)
    capabilities = _effective_capabilities(read_text("/proc/self/status"))
    if os.geteuid() == 0 or capabilities:
        raise RuntimeError(f"still privileged after dropping to {CONTAINER_USER}: CapEff={capabilities:x}")
    environ.update(
        HOME=user.pw_dir,
        USER=CONTAINER_USER,
        LOGNAME=CONTAINER_USER,
        XDG_RUNTIME_DIR=f"/run/user/{user.pw_uid}",
    )
    print(f"gym-host: rootless Podman ready; running as {CONTAINER_USER} with no capabilities", flush=True)
    return True
