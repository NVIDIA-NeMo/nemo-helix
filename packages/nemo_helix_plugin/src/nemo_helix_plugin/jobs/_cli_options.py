# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI helpers for ``-o`` passthrough and ``--options-file`` / ``--spec-file``.

The CLI in :mod:`nemo_helix_plugin.commands` is deliberately dumb about the
contents of ``-o`` and ``--options-file``: values are collected as typed
passthrough and forwarded to the plugin service, which validates against
the resolved backend's options schema. This module contains the pure
parsing + merging logic used by that generator.

Surface:

- :func:`parse_dotted_kv_list` — turn ``-o slurm.partition=gpu-long`` style
  entries into the nested wire dict ``{"slurm": {"partition": "gpu-long"}}``.
- :func:`load_options_file` — load a YAML or JSON options file. File
  extension picks the format; ``.json`` / ``.yaml`` / ``.yml`` are
  recognised. Missing PyYAML produces a helpful error when a YAML file is
  requested.
- :func:`load_spec_file` — same contract as :func:`load_options_file`, but
  intended for the job spec rather than the options bag (no difference at
  parse time; separate name for caller clarity).
- :func:`merge_options` — deep-merge two nested dicts. Overlay wins at leaf
  level. Used to combine ``--options-file`` contents (base) with
  per-flag ``-o`` overrides (overlay).

Design notes:

- Values from ``-o`` flags arrive as strings. The CLI **does not** coerce
  them to ``int`` / ``float`` / ``bool`` — server-side validation handles
  type coercion against the real schema, and premature client-side
  coercion would drift from the server's interpretation. YAML / JSON
  files carry types natively and are passed through untouched.
- Dotted keys support arbitrary depth
  (``-o slurm.resources.shm_size=2Gi``). Empty segments and missing
  ``=`` separator raise :class:`ValueError` with a message naming the
  offending entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# -o <dotted-key>=<value> parsing
# ---------------------------------------------------------------------------


def parse_dotted_kv_list(entries: list[str]) -> dict[str, Any]:
    """Turn a list of ``<dotted.key>=<value>`` entries into a nested dict.

    Example::

        parse_dotted_kv_list([
            "slurm.partition=gpu-long",
            "slurm.nodes=4",
            "slurm.resources.shm_size=2Gi",
        ])
        # {'slurm': {'partition': 'gpu-long', 'nodes': '4',
        #            'resources': {'shm_size': '2Gi'}}}

    Args:
        entries: One string per ``-o`` flag occurrence.

    Returns:
        A nested dict. Empty list produces an empty dict.

    Raises:
        ValueError: On missing ``=`` or empty dotted segments. The message
            names the offending entry.
    """
    out: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid -o entry {entry!r}: expected KEY=VALUE (e.g. -o slurm.partition=gpu-long)")
        key, value = entry.split("=", 1)
        segments = key.split(".")
        if not key or any(not s for s in segments):
            raise ValueError(
                f"invalid -o entry {entry!r}: key has empty segment(s). Use dotted notation like 'slurm.partition'."
            )
        _set_nested(out, segments, value)
    return out


def _set_nested(target: dict[str, Any], segments: list[str], value: Any) -> None:
    """Set ``target[s0][s1]...[sN] = value``, creating intermediate dicts."""
    cursor = target
    for segment in segments[:-1]:
        existing = cursor.get(segment)
        if not isinstance(existing, dict):
            existing = {}
            cursor[segment] = existing
        cursor = existing
    cursor[segments[-1]] = value


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_options_file(path: Path) -> dict[str, Any]:
    """Load an options file (YAML or JSON).

    File extension picks the parser:

    - ``.json`` -> :func:`json.loads`
    - ``.yaml`` / ``.yml`` -> PyYAML's ``safe_load``
    - other -> try JSON first, then YAML on parse failure

    Args:
        path: File to read.

    Returns:
        The parsed top-level object. Expected to be a dict; scalar top
        levels raise :class:`ValueError`.

    Raises:
        FileNotFoundError: When *path* doesn't exist.
        ValueError: When the top level isn't a dict or a YAML file is
            requested without PyYAML installed.
        json.JSONDecodeError / yaml.YAMLError: On malformed content.
    """
    raw = path.read_text()
    data = _parse_by_extension(path, raw)
    if not isinstance(data, dict):
        raise ValueError(f"options file {path} must contain a top-level mapping (got {type(data).__name__})")
    return data


def load_spec_file(path: Path) -> dict[str, Any]:
    """Load a spec file (YAML or JSON).

    Same parser selection as :func:`load_options_file`. Separate name for
    caller-site clarity.
    """
    raw = path.read_text()
    data = _parse_by_extension(path, raw)
    if not isinstance(data, dict):
        raise ValueError(f"spec file {path} must contain a top-level mapping (got {type(data).__name__})")
    return data


def _parse_by_extension(path: Path, raw: str) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(raw)
    if suffix in (".yaml", ".yml"):
        return _load_yaml(raw, source=str(path))
    # Unknown extension — try JSON first, fall back to YAML.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _load_yaml(raw, source=str(path))


def _load_yaml(raw: str, *, source: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a transitive dep today
        raise ValueError(
            f"cannot parse YAML file {source}: PyYAML is not installed. Use a .json file or `uv add pyyaml`."
        ) from exc
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_options(base: dict[str, Any] | None, overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge *overlay* on top of *base*.

    Nested dicts are merged recursively; leaf values from *overlay* win.
    Lists and scalars from *overlay* replace the value in *base* — no
    attempt at list concatenation, since the plan's options are
    dict-of-scalars-per-backend and list-valued fields are rare.

    Neither input is mutated.
    """
    if not base and not overlay:
        return {}
    if not overlay:
        return _deepcopy_dict(base or {})
    if not base:
        return _deepcopy_dict(overlay)

    result = _deepcopy_dict(base)
    for key, over_val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(over_val, dict):
            result[key] = merge_options(result[key], over_val)
        else:
            result[key] = _deepcopy_dict(over_val) if isinstance(over_val, dict) else over_val
    return result


def _deepcopy_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deepcopy_dict(v) for k, v in value.items()}
    return value
