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

import ctypes
import os
import subprocess
from collections.abc import Callable, MutableMapping
from os import chown, geteuid, setgid, setgroups, setuid
from pathlib import Path
from pwd import getpwnam

ENGINE_ENV = "NHX_GYM_HOST_CONTAINER_ENGINE"
ROOTLESS_PODMAN = "rootless-podman"
SETUP_SCRIPT = "/usr/local/libexec/nhx/rootless-podman-up"
CONTAINER_USER = "docker-user"
SETUP_TIMEOUT_S = 300.0
SETUP_LOG = "/tmp/rootless-podman-up.log"
# Operator settings for the setup, set by the platform from its own config and never from a job's variables:
# the size of the file container storage lives in, in bytes, and "1" to mount it with discard.
STORAGE_SIZE_ENV = "NHX_CONTAINER_STORAGE_SIZE"
STORAGE_DISCARD_ENV = "NHX_CONTAINER_STORAGE_DISCARD"

# prctl(2) options. Dropping from the bounding set needs CAP_SETPCAP, so it happens before the uid change.
_PR_SET_DUMPABLE = 4
_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_CAP_SYS_ADMIN = 21

# What the root-only setup runs with: a fixed PATH and the image's and operator's settings for it, nothing else.
_SETUP_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SETUP_ENV_KEYS = (STORAGE_SIZE_ENV, STORAGE_DISCARD_ENV, "NHX_CONTAINER_ENGINE_LOG")

# Kata boots its guest kernel with the agent's systemd target; a runc container sees the node's own
# command line instead. Runtimes that boot without systemd still pass `agent.*` parameters.
_KATA_TARGET = "systemd.unit=kata-containers.target"


def is_kata_guest(kernel_cmdline: str) -> bool:
    """Whether a kernel command line belongs to a Kata Containers guest VM."""
    tokens = kernel_cmdline.split()
    return _KATA_TARGET in tokens or any(token.startswith("agent.") for token in tokens)


def _status_field(status: str, name: str) -> str:
    for line in status.splitlines():
        if line.startswith(f"{name}:"):
            return line.split()[1]
    raise RuntimeError(f"{name} missing from /proc/self/status")


def _libc_prctl(option: int, arg: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, arg, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl({option}, {arg}) failed: {os.strerror(errno)}")


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    """Give everything under ``root`` to ``uid``:``gid``, never following a symbolic link out of it.

    This runs as root over a workspace an earlier attempt of the job may still be writing to. ``os.fwalk``
    holds each directory open and names entries relative to it, so swapping a directory for a link while
    the walk runs can't redirect a change of ownership outside the tree. An entry removed between the listing
    and its ``chown`` has nothing left to hand over, so it is skipped; any other failure still raises.
    """
    for _path, _dirnames, filenames, dir_fd in os.fwalk(root, follow_symlinks=False):
        chown(dir_fd, uid, gid)
        for name in filenames:
            try:
                chown(name, uid, gid, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue


def prepare_container_engine(
    environ: MutableMapping[str, str] = os.environ,
    *,
    read_text: Callable[[str], str] = lambda path: Path(path).read_text(),
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    prctl: Callable[[int, int], None] = _libc_prctl,
) -> bool:
    """Set up the image's container engine and drop to its user. Returns whether it did anything.

    A no-op unless the image asks for an engine. Raises when the sandbox isn't a Kata VM, when setup
    fails, or when the process still holds capabilities after dropping privileges; the caller reports
    that as a bootstrap failure instead of running Gym with a half-configured or privileged host.
    """
    engine = environ.get(ENGINE_ENV, "")
    if not engine:
        # Only this image's setup is meant to run with SYS_ADMIN, and only until it drops it. Granted to an
        # image without an engine (the isolation extension with the default image), the Gym servers would keep it.
        try:
            status = read_text("/proc/self/status")
        except FileNotFoundError:
            # Not Linux: a development machine running the runtime directly, never a sandbox.
            return False
        effective = int(_status_field(status, "CapEff"), 16)
        if effective & (1 << _CAP_SYS_ADMIN):
            raise RuntimeError(
                "this Gym host has SYS_ADMIN but no container engine to set up and drop it: the "
                "bootstrap.execd.isolation extension is meant only for the Docker-enabled Gym host image"
            )
        return False
    if engine != ROOTLESS_PODMAN:
        raise RuntimeError(f"{ENGINE_ENV}={engine!r} is not supported; expected {ROOTLESS_PODMAN!r}")
    if geteuid() != 0:
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
            env={"PATH": _SETUP_PATH, **{key: environ[key] for key in _SETUP_ENV_KEYS if key in environ}},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=SETUP_TIMEOUT_S,
            check=False,
        )
    if result.returncode != 0:
        tail = read_text(SETUP_LOG).strip().splitlines()[-20:]
        raise RuntimeError(f"{SETUP_SCRIPT} exited {result.returncode}: " + " | ".join(tail))

    user = getpwnam(CONTAINER_USER)
    # The work directory is the only place the host writes; everything else it needs is read-only.
    _chown_tree(Path(environ.get("NHX_WORK_PATH", "/job/work")), user.pw_uid, user.pw_gid)
    # The image keeps setuid newuidmap/newgidmap for the Podman service, which the setup started already. On
    # Kata's filesystem they can't carry file capabilities, so a setuid run would get the whole bounding set,
    # SYS_ADMIN included. Nothing under Gym needs them -- the docker CLI only talks to the Podman socket -- so
    # this process and everything it starts lose the bounding set and can't gain privileges through exec.
    for capability in range(int(read_text("/proc/sys/kernel/cap_last_cap").strip()) + 1):
        prctl(_PR_CAPBSET_DROP, capability)
    prctl(_PR_SET_NO_NEW_PRIVS, 1)
    setgroups([])
    setgid(user.pw_gid)
    setuid(user.pw_uid)
    # The kernel usually does this on a uid change, unless fs.suid_dumpable says otherwise. Non-dumpable, this
    # process's memory and original environment stay unreadable to the Gym servers running as the same user.
    prctl(_PR_SET_DUMPABLE, 0)
    status = read_text("/proc/self/status")
    effective = int(_status_field(status, "CapEff"), 16)
    bounding = int(_status_field(status, "CapBnd"), 16)
    no_new_privs = _status_field(status, "NoNewPrivs")
    if geteuid() == 0 or effective or bounding or no_new_privs != "1":
        raise RuntimeError(
            f"still privileged after dropping to {CONTAINER_USER}: "
            f"CapEff={effective:x} CapBnd={bounding:x} NoNewPrivs={no_new_privs}"
        )
    environ.update(
        HOME=user.pw_dir,
        USER=CONTAINER_USER,
        LOGNAME=CONTAINER_USER,
        XDG_RUNTIME_DIR=f"/run/user/{user.pw_uid}",
    )
    print(
        f"gym-host: rootless Podman ready; running as {CONTAINER_USER} with no capabilities and no_new_privs",
        flush=True,
    )
    return True
