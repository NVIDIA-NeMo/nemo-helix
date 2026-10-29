# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry a compose task's ``environment:`` values to the sandbox without writing them into Harbor's config.

Harbor copies ``environment.kwargs`` into ``config.json``, ``lock.json`` and every trial's ``result.json``, and the
platform stores them with the Job's results. A compose service's variables (``POSTGRES_PASSWORD``, a mock API's token)
are task content, not a caller's credentials, but they would land in all of those files, and ``HarborEnvironmentConfig``
refuses any whose name looks like a credential.

So the task Job, which parses the compose file, keeps each value in its own process environment and writes a
``${NEMO_COMPOSE_ENV_<n>}`` template in its place (:func:`externalize_env`). The environment, in the same process, puts
the values back when it builds the sandbox (:func:`expand_env`). Harbor persists only the templates.

This module is not part of the OpenSandbox server extension, so changing it doesn't need the extension's ConfigMap
regenerated or the server restarted.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from nhx_sandbox.harbor_opensandbox_services import ComposeServices

ENV_PREFIX = "NEMO_COMPOSE_ENV_"
# Only this prefix is ever expanded, so a task can't make a value read some other variable of the Job's environment.
_TEMPLATE = re.compile(r"\$\{(" + ENV_PREFIX + r"[0-9]+)\}")


class ComposeEnvError(ValueError):
    """A template names a variable this process doesn't have; names the variable, never a value."""


def externalize_env(spec: Mapping[str, Any], environ: MutableMapping[str, str]) -> dict[str, Any]:
    """Return ``spec`` with every service's and ``main``'s env values swapped for templates, the values held in ``environ``.

    Each value becomes ``${NEMO_COMPOSE_ENV_<n>}``. The numbering follows the services in order, then ``main``, with
    each one's variables sorted by name, so the same compose file always gives the same config: Harbor refuses to
    resume a job directory whose stored config differs, and a retried Job would then discard its finished trials.

    Values are stored as the spec carries them. The spec escapes a service's ``$`` for Kubernetes and leaves ``main``'s
    for the environment to escape at create, so :func:`expand_env` only substitutes and never reinterprets.

    Variables of an earlier call are removed first, so ``environ`` holds exactly this spec's values.
    """
    for name in [name for name in environ if name.startswith(ENV_PREFIX)]:
        del environ[name]

    externalized = copy.deepcopy(dict(spec))
    entries = [*externalized.get("services", []), externalized.get("main") or {}]
    index = 0
    for entry in entries:
        env = entry.get("env")
        if not env:
            continue
        for variable in sorted(env):
            name = f"{ENV_PREFIX}{index}"
            environ[name] = env[variable]
            env[variable] = "${" + name + "}"
            index += 1
    return externalized


def expand_env(services: ComposeServices, environ: Mapping[str, str]) -> ComposeServices:
    """Return ``services`` with each ``${NEMO_COMPOSE_ENV_<n>}`` value replaced by its value from ``environ``.

    Run on the validated model, so a malformed spec's error never carries a real value. A value that isn't one of our
    templates is left as it is.

    Raises:
        ComposeEnvError: A template names a variable ``environ`` doesn't have.
    """
    expanded = [
        service.model_copy(update={"env": _expanded(service.env, environ)}) if service.env else service
        for service in services.services
    ]
    main = services.main
    if main.env:
        main = main.model_copy(update={"env": _expanded(main.env, environ)})
    return services.model_copy(update={"services": expanded, "main": main})


def _expanded(env: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for variable, value in env.items():
        match = _TEMPLATE.fullmatch(value)
        if match is None:
            result[variable] = value
        elif match.group(1) in environ:
            result[variable] = environ[match.group(1)]
        else:
            raise ComposeEnvError(
                f"the value of {variable!r} refers to {match.group(1)}, which this process doesn't have; "
                "the task Job sets it before it starts Harbor"
            )
    return result
