# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-looking values in free-form runner settings: redacted before they are recorded as
provenance, and detected before they are forwarded to a harness that would persist them itself."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "token",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "credential",
    "authorization",
)
_REDACTED = "<redacted>"
_ENV_TEMPLATE = re.compile(r"\$\{[^}]+\}")
#: A marker standing as a word in a path: ``api_keys`` matches, ``tokenizer`` does not.
_SECRET_KEY_MARKER_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(marker) for marker in _SECRET_KEY_MARKERS) + r")s?(?![a-z0-9])"
)
#: Issued-token shapes, for values whose key says nothing. A prefix ending in ``-`` or ``_`` delimits
#: itself and needs no length: nothing ordinary reads as ``glpat-`` at the start of a value. A prefix
#: that is a bare run of letters does not, and runs into real words — ``ASIA`` a region, ``AIza`` the
#: surname Aizawa — so those carry their full body length and a closing boundary instead.
_CREDENTIAL_VALUE = re.compile(
    r"(?:^|[\s:=])"
    r"(?:nvapi-|sk-|hf_|ghp_|gho_|github_pat_|glpat-|xox[bpa]-|xapp-|-----BEGIN"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])"
    r"|AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-]))"
)
_CREDENTIAL_VALUE_MIN_CHARS = 16


def redact_credentials(settings: Mapping[str, Any], _prefix: str = "") -> dict[str, Any]:
    """Redact credential-looking values from free-form settings before they are recorded as provenance.

    Runner configs carry escape hatches forwarded verbatim to the harness (Gym's ``hydra_params``,
    Harbor's ``agent_kwargs``), so nothing stops a caller passing ``{"model": {"api_key": "sk-..."}}``.
    ``RunnerInfo.config`` is persisted into the run bundle, so a value that looks like a credential
    must not be written there.

    The *key* is always kept — knowing that a run set ``model.api_key`` is useful provenance; knowing
    the value is a leak. Matching is on the full dotted path, so a marker anywhere in it redacts, and
    nesting cannot hide a credential behind an innocuous leaf name.

    Lists are walked too, since a mapping inside one — ``{"models": [{"api_key": "sk-..."}]}`` —
    reaches the harness just as a nested mapping does. The index contributes no path segment: what
    marks a value as a credential is the key it sits under, not where in a list it happens to fall.
    """
    redacted: dict[str, Any] = {}
    for key, value in settings.items():
        path = f"{_prefix}{key}"
        if isinstance(value, Mapping):
            redacted[key] = redact_credentials(value, f"{path}.")
        elif any(marker in path.casefold() for marker in _SECRET_KEY_MARKERS):
            redacted[key] = _REDACTED
        elif isinstance(value, (list, tuple)):
            redacted[key] = [_redact_list_item(item, path) for item in value]
        elif is_credential_value(value):
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


def _redact_list_item(item: Any, path: str) -> Any:
    """Redact inside one element of a list-valued setting. See :func:`redact_credentials`."""
    if isinstance(item, Mapping):
        return redact_credentials(item, f"{path}.")
    if isinstance(item, (list, tuple)):
        return [_redact_list_item(nested, path) for nested in item]
    return _REDACTED if is_credential_value(item) else item


def credential_shaped_settings(settings: Mapping[str, Any]) -> list[str]:
    """Dotted paths in ``settings`` holding a plaintext value that looks like a credential.

    The counterpart to :func:`redact_credentials`, for settings that are not merely *recorded* but
    forwarded into a harness that persists them itself. Redaction cannot help there: the harness
    needs the real value to run, so the only way to keep it off disk is to refuse it at the edge.

    Paths are built as :func:`redact_credentials` builds them, and lists are walked the same way
    without contributing an index. Three narrowings apply, because refusing a run is less forgiving
    than redacting a record:

    * Only non-empty strings qualify. Redacting an integer costs nothing; rejecting one would refuse
      a legitimate run.
    * A marker must stand as a word in the path. ``tokenizer`` is a plausible kwarg and not a
      credential, where redaction can afford to match it.
    * ``${NAME}`` templates pass. Naming a credential without carrying its value is the supported
      route, so it is what a rejected caller is redirected to.

    A value carrying a recognised issued-token shape is reported whatever its key, since an ``env``
    mapping forwarded to a harness names its own variables.
    """
    exposed = [path for key, value in settings.items() for path in _exposed_paths(key, value)]
    return list(dict.fromkeys(exposed))


def _exposed_paths(path: str, value: Any) -> list[str]:
    """Paths exposed by one setting. See :func:`credential_shaped_settings`."""
    if isinstance(value, Mapping):
        return [nested for key, item in value.items() for nested in _exposed_paths(f"{path}.{key}", item)]
    if isinstance(value, (list, tuple)):
        return [nested for item in value for nested in _exposed_paths(path, item)]
    if not isinstance(value, str) or not value or _ENV_TEMPLATE.fullmatch(value):
        return []
    if _SECRET_KEY_MARKER_RE.search(path.casefold()):
        return [path]
    return [path] if is_credential_value(value) else []


def is_credential_value(value: Any) -> bool:
    """Whether a scalar carries a recognised issued-token shape, whatever key it sits under.

    Shared by detection and redaction so the two cannot drift: a value refused at a config edge must
    also be one that never reaches the run bundle, and the runtimes that redact without refusing
    (Gym's ``hydra_params`` and ``env_vars``) have nothing else standing between them and a leak.
    """
    if not isinstance(value, str) or len(value) < _CREDENTIAL_VALUE_MIN_CHARS:
        return False
    return not _ENV_TEMPLATE.fullmatch(value) and bool(_CREDENTIAL_VALUE.search(value))


def require_no_plaintext_credentials(settings: Mapping[str, Any], *, field: str, alternative: str) -> None:
    """Raise if ``settings`` would hand a harness a plaintext credential it persists to disk.

    Harbor copies the settings it is given into the job directory's ``config.json`` and ``lock.json``
    and into every trial's ``config.json``, ``lock.json``, ``result.json`` and agent run spec. Six
    durable files from one run, one of which is the result a user attaches to a bug report.
    """
    exposed = credential_shaped_settings(settings)
    if not exposed:
        return
    raise ValueError(
        f"these `{field}` entries look like plaintext credentials, and the harness persists them "
        f"unredacted across the job directory (config.json, lock.json, and every trial's result.json): "
        f"{', '.join(exposed)}. Name the variable in `{alternative}` instead, so the harness is handed a "
        "`${NAME}` template and resolves the value when it creates the agent."
    )
