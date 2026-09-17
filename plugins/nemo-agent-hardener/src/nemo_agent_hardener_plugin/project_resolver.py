# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a war-game manifest from an uploaded project bundle.

The counterpart to :mod:`agent_resolver`, for a victim the platform did not render. A registered
agent states its shape in ``agent.yaml``; a project states it in a Dockerfile, less completely and
less formally. So this module reads what the project *does* say and, crucially, reports what it does
not: ``unresolved`` is the list of fields a human still has to supply.

That distinction is the whole point. The user never writes ``agent-hardener.yaml`` — asking them to fill
four fields is a form, asking them to author a manifest is a spec. Anything derivable is derived,
and the rest is named explicitly rather than silently defaulted, because a wrong value here fails
minutes into a run with an error that does not mention the cause.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

#: Fields a project cannot state about itself, so a caller must.
HARNESS_FIELD = "harness"
RELAY_FIELD = "relay_integration_confirmed"

#: Directories that never hold the agent's own Dockerfile. Skipped so a vendored example or a test
#: fixture does not win the "exactly one Dockerfile" check against the real one at the root.
_IGNORED_DIRS = frozenset({".git", ".venv", "node_modules", "__pycache__", ".agent-hardener", "dist", "build"})

#: An env var whose *name* looks like a credential. Used to seed secret names, never to read values —
#: a value baked into a Dockerfile is a leak, and copying it onto the manifest would spread it.
_SECRET_NAME = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)

_URL = re.compile(r"https?://([A-Za-z0-9.-]+)")


def _iter_dockerfiles(root: Path) -> list[Path]:
    """Every Dockerfile in the project, nearest the root first."""
    found = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name.lower().startswith("dockerfile")
        and not any(part in _IGNORED_DIRS for part in path.relative_to(root).parts)
    ]
    return sorted(found, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def _logical_lines(text: str) -> list[str]:
    """Dockerfile lines with continuations joined and comments dropped."""
    joined = re.sub(r"\\\s*\n", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


def dockerfile_env(text: str) -> dict[str, str]:
    """Parse ``ENV`` declarations, both ``K=V`` and the legacy ``ENV K v`` form."""
    env: dict[str, str] = {}
    for line in _logical_lines(text):
        if not line.upper().startswith("ENV "):
            continue
        body = line[4:].strip()
        if "=" in body:
            try:
                parts = shlex.split(body)
            except ValueError:
                parts = body.split()
            for part in parts:
                key, sep, value = part.partition("=")
                if sep and key:
                    env[key.strip()] = value.strip().strip("\"'")
        else:
            key, _, value = body.partition(" ")
            if key:
                env[key.strip()] = value.strip().strip("\"'")
    return env


def _exec_form(body: str) -> list[str] | None:
    """The argv of a JSON exec-form ``ENTRYPOINT``/``CMD``, or ``None`` for the shell form.

    Only the exec form is machine-readable. A shell form (``ENTRYPOINT python -m x``) runs through
    ``/bin/sh -c`` with the image's own ``PATH``, and OpenShell replaces ``PATH`` — so reusing that
    string verbatim would start nothing, and guessing at the absolute interpreter would be a guess.
    """
    body = body.strip()
    if not body.startswith("["):
        return None
    try:
        argv = json.loads(body)
    except ValueError:
        return None
    return [str(part) for part in argv] if isinstance(argv, list) else None


def derive_start_command(text: str, env: dict[str, str]) -> str:
    """The command that serves the agent, when the Dockerfile states it unambiguously.

    Returns ``""`` when it does not. ``ENTRYPOINT ["sh","-c","exec python -m ..."]`` is the shape a
    Fabric image uses, so the inner script is unwrapped and its ``$VAR`` references resolved from
    ``ENV`` — the sandbox does not propagate them.
    """
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None
    for line in _logical_lines(text):
        upper = line.upper()
        if upper.startswith("ENTRYPOINT "):
            entrypoint = _exec_form(line[11:])
        elif upper.startswith("CMD "):
            cmd = _exec_form(line[4:])

    argv = entrypoint or cmd
    if not argv:
        return ""

    # `sh -c "<script>"` — the script is the real command.
    if len(argv) >= 3 and Path(argv[0]).name in {"sh", "bash"} and argv[1] == "-c":
        script = argv[2]
    else:
        script = shlex.join(argv)

    script = re.sub(r"^exec\s+", "", script.strip())

    # Resolve ${VAR} / $VAR from the image's own ENV; an unresolved one means we cannot state the command.
    def _sub(match: re.Match[str]) -> str:
        return env.get(match.group(1) or match.group(2), match.group(0))

    script = re.sub(r"\$\{(\w+)\}|\$(\w+)", _sub, script)
    if "$" in script:
        return ""
    return _absolutize(script, env)


def _absolutize(script: str, env: dict[str, str]) -> str:
    """Rewrite a bare interpreter name to its absolute path inside the image.

    ``openshell sandbox exec`` replaces ``PATH``, so a command that resolves ``python`` through the
    image's own ``ENV PATH`` starts nothing — the agent never comes up and the run reports an
    uninstrumented victim rather than a bad command. The venv is the only place the name can come
    from, so an image without one leaves the command as-is for a human to correct.
    """
    try:
        argv = shlex.split(script)
    except ValueError:
        return script
    if not argv or argv[0].startswith("/"):
        return script
    venv = env.get("VIRTUAL_ENV", "").rstrip("/")
    if not venv:
        return script
    argv[0] = f"{venv}/bin/{Path(argv[0]).name}"
    return shlex.join(argv)


def derive_port(text: str, env: dict[str, str]) -> int:
    """Victim port from ``ENV PORT`` first, then ``EXPOSE``; 8000 when neither says."""
    port = env.get("PORT", "")
    if port.isdigit():
        return int(port)
    for line in _logical_lines(text):
        if line.upper().startswith("EXPOSE "):
            candidate = line[7:].strip().split("/")[0].split()[0]
            if candidate.isdigit():
                return int(candidate)
    return 8000


def derive_binaries(text: str, env: dict[str, str]) -> list[str]:
    """Propose interpreter globs for the sandbox's egress policy.

    The proposal is the image's virtualenv plus the system interpreters it resolves to. Both halves
    matter: OpenShell matches a policy block against the *resolved* executable, and a venv's
    ``bin/python`` is a symlink to the system one — a block naming only the venv glob matches no
    process and silently grants nothing.
    """
    venv = env.get("VIRTUAL_ENV", "").strip()
    if not venv:
        match = re.search(r"python\s+-m\s+venv\s+(\S+)", text)
        venv = match.group(1) if match else ""
    globs = [f"{venv.rstrip('/')}/bin/**"] if venv else []
    return [*globs, "/usr/local/bin/python*", "/usr/bin/python*"]


def derive_egress(text: str, env: dict[str, str]) -> list[str]:
    """Hosts the project's own files name, for the default-deny sandbox's allow-list."""
    hosts = {match.group(1) for match in _URL.finditer(text)}
    for value in env.values():
        hosts.update(match.group(1) for match in _URL.finditer(value))
    return sorted(host for host in hosts if "." in host)


def derive_secrets(env: dict[str, str], project_root: Path) -> list[str]:
    """Secret *names* from credential-shaped ENV keys and any committed dotenv."""
    names = {key for key in env if _SECRET_NAME.search(key)}
    for candidate in (project_root / ".env", project_root / ".env.example"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                names.add(line.split("=", 1)[0].strip())
    return sorted(names)


def inspect_project(project_root: Path, *, dockerfile: str | None = None) -> dict[str, Any]:
    """Read an extracted project bundle and report what it states, and what it cannot.

    Args:
        project_root: Directory the uploaded bundle was extracted into.
        dockerfile: Caller's choice, when the project holds more than one.

    Returns:
        A mapping matching ``InspectProjectResponse``: derived values plus ``unresolved``.
    """
    warnings: list[str] = []
    candidates = _iter_dockerfiles(project_root)
    relative = [str(path.relative_to(project_root)) for path in candidates]

    if dockerfile:
        # Kept unresolved so the `relative_to(project_root)` below still works when project_root
        # itself runs through a symlink (macOS puts temp dirs under /var -> /private/var). Only the
        # containment check resolves, and it resolves *both* sides.
        chosen = project_root / dockerfile
        if not chosen.resolve().is_relative_to(project_root.resolve()) or not chosen.is_file():
            return {
                "dockerfiles": relative,
                "unresolved": ["dockerfile"],
                "warnings": [f"{dockerfile!r} is not a file in the uploaded bundle."],
            }
    elif len(candidates) == 1:
        chosen = candidates[0]
    elif not candidates:
        return {
            "dockerfiles": [],
            "unresolved": ["dockerfile"],
            "warnings": ["No Dockerfile in the bundle. Agent Hardener hardens the image you ship; it builds none."],
        }
    else:
        return {
            "dockerfiles": relative,
            "unresolved": ["dockerfile"],
            "warnings": [f"{len(candidates)} Dockerfiles found; name the one that builds the agent."],
        }

    text = chosen.read_text(encoding="utf-8", errors="replace")
    env = dockerfile_env(text)
    start_command = derive_start_command(text, env)
    if not start_command:
        warnings.append(
            "The Dockerfile's ENTRYPOINT/CMD is a shell form or references unresolved variables, so the "
            "start command cannot be read from it. OpenShell replaces PATH, so it must be absolute."
        )

    # Credentials are named, never carried: a value baked into an image is already a leak, and copying
    # it onto the manifest would spread it to a second store.
    secrets = derive_secrets(env, project_root)
    plain_env = {
        key: value
        for key, value in env.items()
        # PATH is the sandbox's to set, and OpenShell replaces it anyway — forwarding the image's
        # would be ignored at best. A value still holding `$` was written to be expanded by a shell
        # that never ran, so passing it on sets the literal text as the variable.
        if key not in set(secrets) and key != "PATH" and "$" not in value
    }

    secrets_and_egress = derive_egress(text, env)
    if not secrets:
        warnings.append(
            "No credential-shaped env names in the Dockerfile or a committed dotenv. If the agent calls a "
            "model, name its key in `secrets` or the victim starts and then fails its first call."
        )
    if not secrets_and_egress:
        warnings.append(
            "No hosts named in the Dockerfile. The run also scans the project's source for endpoints, so this "
            "is not necessarily a gap — but the sandbox is default-deny, and a host that neither names (one "
            "reached only from an installed dependency, say) must be listed in `egress` or its traffic is "
            "dropped mid-run."
        )

    unresolved = [name for name, value in (("start_command", start_command),) if not value]
    unresolved += [HARNESS_FIELD, RELAY_FIELD]

    return {
        "dockerfile": str(chosen.relative_to(project_root)),
        "dockerfiles": relative,
        "start_command": start_command,
        "binaries": derive_binaries(text, env),
        "port": derive_port(text, env),
        "secrets": secrets,
        "egress": secrets_and_egress,
        "env": plain_env,
        "unresolved": unresolved,
        "warnings": warnings,
    }


def build_project_manifest_dict(
    *,
    agent_name: str,
    project_dir: str,
    dockerfile: str,
    start_command: str,
    binaries: list[str],
    port: int,
    secrets: list[str],
    secrets_file: str = ".env",
    egress: list[str] | None = None,
    backends: list[dict[str, Any]] | None = None,
    harness: str | None = None,
    relay_integration_confirmed: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the ``agent-hardener.yaml`` mapping for a project victim.

    Deliberately separate from :func:`agent_resolver.build_manifest_dict`, which hard-codes the Fabric
    launch shape it renders itself. Here every one of those fields comes from the author's image, so
    sharing the function would mean threading overrides through a signature whose defaults are only
    ever right for the other caller.
    """
    agent: dict[str, Any] = {
        "name": agent_name,
        "project_dir": project_dir,
        "harness": harness,
        "relay_integration_confirmed": relay_integration_confirmed,
        "dockerfile": dockerfile,
        "start_command": start_command,
        "binaries": binaries,
        "port": port,
        "secrets": secrets,
        "secrets_file": secrets_file,
    }
    if env:
        agent["env"] = env
    if egress:
        agent["egress"] = egress
    return {"agent": agent, "backends": backends or []}
