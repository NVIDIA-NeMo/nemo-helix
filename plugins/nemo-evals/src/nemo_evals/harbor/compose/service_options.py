# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read one option of a compose service (``environment``, ``ports``, ``healthcheck``, ...) as one typed value.

Every ``parse_*`` function takes the option's raw YAML value and the ``path`` it came from, such as
``services.db.ports``. A value Compose itself would refuse is recorded with :meth:`ComposeProblems.invalid`, and a
feature a pod can't run with :meth:`ComposeProblems.unsupported`. Neither raises, so one error can list every problem.
"""

from __future__ import annotations

from typing import Any

from nemo_evals.harbor.compose.models import ComposeProblems


def _scalar(value: Any) -> str | None:
    """Render a YAML scalar as compose passes it on (``True`` is ``"true"``, ``8`` is ``"8"``); None for any other value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def parse_environment(value: Any, path: str, problems: ComposeProblems) -> dict[str, str | None]:
    """Read ``environment`` (or a build's ``args``) as ``{name: value}`` from either of compose's forms.

    ``{A: 1}`` and ``["A=1"]`` both give ``{"A": "1"}``. A name with no value, such as ``["A"]`` or ``{A: null}``,
    gives ``{"A": None}``: Compose looks it up in the variables it interpolates with (see
    :meth:`Interpolator.variable_value`), and when none has it, the container doesn't get it, even if an ``env_file``
    set it.
    """
    if value is None:
        return {}

    # Both forms become (name, value) pairs first; a value of None means the entry named a variable but gave no value.
    pairs: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        pairs = [(str(name), entry_value) for name, entry_value in value.items()]
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            text = _scalar(entry)
            if text is None:
                problems.invalid(f"{path}[{index}]", "expected KEY=value")
                continue
            # Split at the first `=` only, so a value may itself contain `=`: "URL=http://a?b=c".
            name, equals_sign, entry_value = text.partition("=")
            pairs.append((name, entry_value if equals_sign else None))
    else:
        problems.invalid(path, "expected a mapping or a list")
        return {}

    environment: dict[str, str | None] = {}
    for name, entry_value in pairs:
        # Kept as None, so it's still looked up, and still overrides an `env_file`, once the service's files are read.
        if entry_value is None:
            environment[name] = None
            continue
        text = _scalar(entry_value)
        if text is None:
            problems.invalid(f"{path}.{name}", "expected a string, number or boolean")
        else:
            environment[name] = text
    return environment


def parse_depends_on(value: Any, path: str, problems: ComposeProblems) -> dict[str, dict[str, Any]]:
    """Read ``depends_on`` as ``{service: options}``; the list form means ``service_started`` for each name.

    ``["db"]`` gives ``{"db": {"condition": "service_started"}}``.
    """
    if value is None:
        return {}
    if isinstance(value, list):
        return {str(name): {"condition": "service_started"} for name in value}
    if not isinstance(value, dict):
        problems.invalid(path, "expected a list or a mapping")
        return {}

    dependencies: dict[str, dict[str, Any]] = {}
    for name, options in value.items():
        if options is not None and not isinstance(options, dict):
            problems.invalid(f"{path}.{name}", "expected a mapping such as {condition: service_healthy}")
            continue
        dependencies[str(name)] = dict(options or {})
    return dependencies
