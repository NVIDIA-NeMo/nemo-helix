# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schema-to-CLI-flag plumbing for `NemoJob` / `NemoFunction` verbs.

Walks a Pydantic ``spec_schema`` and emits one flat
:class:`SpecLeafField` per scalar leaf (recursing into nested
``BaseModel`` submodels via dotted paths). Callers turn the leaves
into Typer ``Option`` parameters via :func:`make_field_param`,
splice them into a synthetic :class:`inspect.Signature` via
:func:`build_callback_signature`, and overlay user-supplied values
on top of a base spec via :func:`build_overlay` + :func:`deep_merge`.

Why this lives at the package root rather than under ``jobs/`` or
``functions/``: PR #160 (auto-generate CLI flags
from spec_schema") introduces the same plumbing under
``nemo_helix_plugin/jobs/_spec_flags.py`` and threads it through
:func:`add_job_commands`. This PR needs the same
mechanic for :func:`add_function_commands`. Putting the code at
the package root makes the shared boundary explicit; once both
PRs land the unification is just ``git rm
nemo_helix_plugin/jobs/_spec_flags.py`` and re-pointing the jobs
imports here. Until then the two files coexist with identical
semantics — keep changes in lockstep.

Scope of this first cut (matches PR #160's scope so the future
unification is mechanical):

- Supported leaf types: ``str``, ``int``, ``float``, ``bool``, plus
  ``Optional`` of any of those (``str | None`` syntax included).
- Nested ``BaseModel`` fields recurse with a dotted prefix.
- Same-base scalar union arms collapse to one flag of the shared
  base (``AgentRef | EndpointURL | None`` → one ``str`` flag); the
  runtime resolver disambiguates by shape, the wire format stays
  a plain string. Mixed-base unions, lists, dicts, ``Literal``,
  ``Path``, etc. are silently skipped — they remain reachable via
  ``--spec`` / ``--spec-file``.
- Field defaults flow into the Typer flag default. Required fields
  default to a sentinel so the overlay layer can tell "user did
  not pass" from "user passed an empty value".

The synthetic-signature helpers (:func:`make_field_param`,
:func:`build_callback_signature`) are colocated with the walker
so callers only need one import.
"""

from __future__ import annotations

import inspect
import logging
import types
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union, cast, get_args, get_origin

import typer
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)

_SUPPORTED_SCALARS: tuple[type, ...] = (str, int, float, bool)


# Sentinel default for every auto-generated per-field Typer Option.
# Click won't try to coerce ``None`` through the parameter's type
# converter the way it would for an opaque ``object()`` sentinel —
# which would otherwise produce a stringified default in the spec.
# Trade-off: users can't explicitly set an ``Optional`` field back
# to ``None`` via a per-field flag — they have to drop into
# ``--spec`` / ``--spec-file`` for that. Matches PR #160's choice.
UNSET: Any = None


@dataclass(frozen=True)
class SpecLeafField:
    """A scalar leaf in a ``spec_schema``, ready to be turned into a Typer flag.

    Attributes:
        path: Dotted path from the spec root, e.g. ``("dataset", "path")``.
            Joined with ``.`` for the CLI flag (``--dataset.path``).
        param_name: A Python identifier used as the synthetic function's
            kwarg name. Distinct from ``path`` so dots / hyphens don't
            leak into Python parameter names.
        python_type: The non-``None`` scalar type for the flag annotation.
        default: The field's default value, or :data:`PydanticUndefined`
            when the field is required.
        description: Help text for the CLI flag, sourced from
            ``Field(description=...)``. Empty string when unset.
        required: ``True`` when the field has no default.
        metavar: Optional placeholder rendered after the flag in
            ``--help``. ``None`` lets Typer fall back to upper-casing
            the flag name. Resolved by :func:`_derive_metavar` from a
            per-field ``Field(json_schema_extra={"cli_metavar": ...})``
            (wins unconditionally) or from each union arm's
            ``__cli_metavar__`` class attribute (joined with ``" | "``)
            when every arm declares one.
        partial: ``True`` when *python_type* covers only one arm of a
            mixed scalar/model union (e.g. ``str | AgentInline`` emits
            a flag for the ``str`` arm only). The other arm(s) remain
            reachable solely through ``--spec`` / ``--spec-file``.
    """

    path: tuple[str, ...]
    param_name: str
    python_type: type
    default: Any
    description: str
    required: bool
    metavar: str | None = None
    partial: bool = False

    @property
    def flag(self) -> str:
        """The CLI flag name in kebab-case, e.g. ``--agent-endpoint`` or ``--dataset.path-v2``."""
        return "--" + ".".join(seg.replace("_", "-") for seg in self.path)


def walk_spec_leaves(
    model: type[BaseModel] | None,
    *,
    reserved: Iterable[str] = (),
    unavailable: list[str] | None = None,
) -> list[SpecLeafField]:
    """Walk *model* recursively and return one entry per scalar leaf.

    Returns an empty list when *model* is ``None``. Skips fields whose
    types aren't currently supported (lists, dicts, multi-type unions
    that don't share a scalar base and don't reduce to a single scalar
    arm, ``Literal``, ``Path``, plain ``BaseModel`` references that
    aren't nested submodels, ...) — these remain accessible via the
    JSON ``--spec`` / ``--spec-file`` path. A union that mixes exactly
    one scalar arm with one or more model/unsupported arms (e.g.
    ``str | AgentInline``) still emits a flag for the scalar arm; see
    :attr:`SpecLeafField.partial`.

    Args:
        model: A Pydantic v2 ``BaseModel`` subclass, or ``None``.
        reserved: Param names (from the wrapping CLI verb's static
            flags) to skip. The set is verb-specific. A field whose
            name matches one of *reserved* falls through to ``--spec`` /
            ``--spec-file`` for value passing.
        unavailable: When given, appended in place with one dotted
            path per field (or union arm) not fully representable as a
            flag — fields skipped entirely, plus the non-scalar arm of
            each :attr:`SpecLeafField.partial` leaf. Callers use this
            to build a ``--help`` note pointing at ``--spec`` /
            ``--spec-file`` for those paths.

    Returns:
        Flat list of :class:`SpecLeafField` in declaration order.
    """
    if model is None:
        return []
    reserved_set = frozenset(reserved)
    leaves: list[SpecLeafField] = []
    _walk(model, prefix=(), out=leaves, used_param_names=set(), reserved=reserved_set, unavailable=unavailable)
    return leaves


def _walk(
    model: type[BaseModel],
    *,
    prefix: tuple[str, ...],
    out: list[SpecLeafField],
    used_param_names: set[str],
    reserved: frozenset[str],
    unavailable: list[str] | None,
) -> None:
    for name, info in model.model_fields.items():
        path = (*prefix, name)
        annotation = info.annotation
        unwrapped, is_partial = _strip_optional(annotation)
        if unwrapped is None:
            logger.debug(
                "Skipping field %s on %s: unsupported union or unannotated.",
                ".".join(path),
                model.__name__,
            )
            if unavailable is not None:
                unavailable.append(".".join(path))
            continue

        if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
            _walk(
                unwrapped,
                prefix=path,
                out=out,
                used_param_names=used_param_names,
                reserved=reserved,
                unavailable=unavailable,
            )
            continue

        if not _is_supported_scalar(unwrapped):
            logger.debug(
                "Skipping field %s on %s: unsupported leaf type %r.",
                ".".join(path),
                model.__name__,
                unwrapped,
            )
            if unavailable is not None:
                unavailable.append(".".join(path))
            continue

        param_name = _path_to_param_name(path, used=used_param_names)
        if param_name in reserved:
            logger.debug(
                "Skipping spec field %s: name collides with reserved CLI flag %r for this verb. "
                "Use --spec / --spec-file to set this field.",
                ".".join(path),
                param_name,
            )
            if unavailable is not None:
                unavailable.append(".".join(path))
            continue
        used_param_names.add(param_name)

        if is_partial:
            logger.debug(
                "Field %s on %s is a mixed scalar/model union: emitting --%s for the "
                "scalar arm only. Other shapes require --spec / --spec-file.",
                ".".join(path),
                model.__name__,
                param_name,
            )
            if unavailable is not None:
                unavailable.append(f"{'.'.join(path)} (other union forms)")

        out.append(
            SpecLeafField(
                path=path,
                param_name=param_name,
                python_type=unwrapped,
                default=_extract_default(info),
                description=info.description or "",
                required=_is_required(info),
                metavar=_derive_metavar(annotation, info),
                partial=is_partial,
            )
        )


def _strip_optional(annotation: Any) -> tuple[Any | None, bool]:
    """Return the inner CLI-renderable type for *annotation*, and whether it's partial.

    Handles four union shapes:

    - ``Optional[X]`` / ``X | None``: returns ``(X, False)``, except
      when ``X`` is a strict subclass of a supported scalar
      (``str``/``int``/``float``/``bool``) — in that case returns the
      scalar base so Typer can build a flag from a type it knows how
      to render.  Subclass identity is preserved at parse time by the
      Pydantic validator on the spec model.
    - A union of two or more non-``None`` arms that all subclass the
      same supported scalar base: returns ``(base, False)``.
    - A union of two or more non-``None`` arms where exactly one arm
      is scalar-compatible and the rest aren't (e.g.
      ``str | AgentInline``): returns ``(scalar_base, True)`` — the
      flag covers only that arm; the others are only reachable via
      ``--spec`` / ``--spec-file``.
    - Anything else: returns ``(None, False)`` to signal "skip this
      field entirely".
    """
    if annotation is None:
        return None, False
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _scalar_base_or_self(non_none[0]), False
        base = _common_scalar_base(non_none)
        if base is not None:
            return base, False
        return _partial_scalar_arm(non_none)
    return _scalar_base_or_self(annotation), False


def _partial_scalar_arm(types_: list[Any]) -> tuple[type | None, bool]:
    """Return the lone scalar arm of a mixed union, marked partial.

    Applies only when exactly one arm collapses to a supported scalar
    base and at least one other arm doesn't (a model type, another
    unsupported shape, ...). Two or more differently-based scalar arms
    (``str | int | AgentInline``) are ambiguous as to which flag type
    to expose, so those still fall through to "skip this field".
    """
    scalar_arms = [t for t in types_ if _is_supported_scalar(_scalar_base_or_self(t))]
    if len(scalar_arms) != 1:
        return None, False
    return _scalar_base_or_self(scalar_arms[0]), True


def _scalar_base_or_self(tp: Any) -> Any:
    """Collapse a ``_SUPPORTED_SCALARS`` subclass to its scalar base.

    Typer can't introspect arbitrary ``str`` subclasses (``FilesetRef``,
    ``AgentRef``, ...) when building a Click parameter type, so the
    auto-generated flag has to advertise the base type.  Pydantic
    re-coerces the wire string back to the declared subclass when
    the merged spec is validated.

    Returns *tp* unchanged when it isn't a subclass of one of the
    supported scalars or when it *is* one of them (no narrowing
    needed).
    """
    if not isinstance(tp, type):
        return tp
    if tp in _SUPPORTED_SCALARS:
        return tp
    for base in _SUPPORTED_SCALARS:
        if issubclass(tp, base):
            return base
    return tp


def _union_arms(annotation: Any) -> list[Any]:
    """Non-``None`` arms of a ``Union``, or ``[annotation]`` for a scalar."""
    if annotation is None:
        return []
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return [a for a in get_args(annotation) if a is not type(None)]
    return [annotation]


def _derive_metavar(annotation: Any, info: FieldInfo) -> str | None:
    """Return the ``--help`` placeholder for a leaf, or ``None`` for the default.

    Two opt-in surfaces, in priority order:

    1. ``Field(json_schema_extra={"cli_metavar": "..."})`` on the field
       itself — always wins.
    2. ``__cli_metavar__`` class attribute on each non-``None`` arm of
       the field's annotation. When every arm declares one they are
       joined with ``" | "`` to form the placeholder.

    Returns ``None`` when neither path applies, leaving Typer to fall
    back to upper-casing the flag name.
    """
    raw_extra = info.json_schema_extra
    if isinstance(raw_extra, dict):
        # ``ty`` infers the dict value type as ``Never`` for ``json_schema_extra``;
        # rebind through ``Any`` so ``.get("cli_metavar")`` type-checks here.
        extra: dict[str, Any] = cast("dict[str, Any]", raw_extra)
        override = extra.get("cli_metavar")
        if isinstance(override, str) and override:
            return override

    arms = _union_arms(annotation)
    if not arms:
        return None
    metavars: list[str] = []
    for arm in arms:
        meta = getattr(arm, "__cli_metavar__", None)
        if not isinstance(meta, str) or not meta:
            return None
        metavars.append(meta)
    return " | ".join(metavars)


def _common_scalar_base(types_: list[Any]) -> type | None:
    """Return the supported-scalar base shared by every type in *types_*, else ``None``."""
    if not types_:
        return None
    for base in _SUPPORTED_SCALARS:
        if all(isinstance(t, type) and issubclass(t, base) for t in types_):
            return base
    return None


def _is_supported_scalar(tp: Any) -> bool:
    """``True`` iff *tp* is a subclass of one of the scalar types we render.

    ``bool`` is intentionally checked before ``int`` even though
    ``isinstance(True, int)`` is ``True``: Python's ``bool`` *is* an
    ``int`` subclass, but Typer renders ``bool`` as ``--flag/--no-flag``
    and ``int`` as a value-taking flag.
    """
    return isinstance(tp, type) and issubclass(tp, _SUPPORTED_SCALARS)


def _is_required(info: FieldInfo) -> bool:
    """True when the field has no default and no default factory."""
    return info.default is PydanticUndefined and info.default_factory is None


def _extract_default(info: FieldInfo) -> Any:
    """Resolve a usable Typer default. Returns :data:`PydanticUndefined` for required."""
    if info.default is not PydanticUndefined:
        return info.default
    if info.default_factory is not None:
        try:
            return info.default_factory()  # ty: ignore[missing-argument]
        except TypeError:
            # Pydantic supports validated_data-aware factories; we can't
            # safely call those at CLI-build time. Treat as required so
            # the schema validates the missing-value case server-side.
            return PydanticUndefined
    return PydanticUndefined


def _path_to_param_name(path: tuple[str, ...], *, used: set[str]) -> str:
    """Convert a dotted path to a unique, valid Python identifier."""
    base = "__".join(_sanitise(seg) for seg in path)
    if base not in used:
        return base
    i = 2
    while f"{base}__{i}" in used:
        i += 1
    return f"{base}__{i}"


def _sanitise(segment: str) -> str:
    """Make *segment* a safe identifier piece (defensive)."""
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in segment)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "_"


# ---------------------------------------------------------------------------
# Overlay helpers — convert flat ``{param_name: value}`` to a nested dict and
# deep-merge it on top of a base spec.
# ---------------------------------------------------------------------------


def build_overlay(
    leaves: Sequence[SpecLeafField],
    raw_values: dict[str, Any],
    *,
    unset_sentinel: Any = UNSET,
) -> dict[str, Any]:
    """Project *raw_values* (keyed by ``param_name``) back to a nested dict.

    Values equal to *unset_sentinel* are treated as "not provided on the
    CLI" and dropped — they don't override values in ``--spec-file``.
    """
    overlay: dict[str, Any] = {}
    for leaf in leaves:
        if leaf.param_name not in raw_values:
            continue
        value = raw_values[leaf.param_name]
        if value is unset_sentinel:
            continue
        _set_nested(overlay, leaf.path, value)
    return overlay


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overlay* on top of *base*. Neither input is mutated.

    Nested dicts are merged recursively; leaf values from *overlay* win.
    Lists and scalars from *overlay* replace values in *base*.
    """
    if not base and not overlay:
        return {}
    if not overlay:
        return _deepcopy(base)
    if not base:
        return _deepcopy(overlay)
    result = _deepcopy(base)
    for key, over_val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(over_val, dict):
            result[key] = deep_merge(result[key], over_val)
        else:
            result[key] = _deepcopy(over_val) if isinstance(over_val, dict) else over_val
    return result


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for segment in path[:-1]:
        existing = cursor.get(segment)
        if not isinstance(existing, dict):
            existing = {}
            cursor[segment] = existing
        cursor = existing
    cursor[path[-1]] = value


def _deepcopy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deepcopy(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Synthetic-signature helpers — splice spec leaves into a Typer callback's
# signature alongside hand-written static flags.
# ---------------------------------------------------------------------------


def kw(name: str, annotation: object, default: object) -> inspect.Parameter:
    """Shorthand for a keyword-only synthetic parameter.

    Public because both verbs in :mod:`nemo_helix_plugin.commands` build their
    static flags via this helper, and so will the post-merge unified
    jobs CLI. Callers pass an ``inspect.Parameter`` to
    :func:`build_callback_signature`.
    """
    return inspect.Parameter(
        name=name,
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=default,
        annotation=annotation,
    )


def _optional_of(tp: object) -> Any:
    """Return ``Optional[T]`` at runtime for a dynamically-known ``T``.

    Typer introspects parameter annotations to pick its Click type and whether
    the value is required. We need ``Optional[leaf.python_type]`` but cannot
    spell that as a type expression because ``leaf.python_type`` is only a
    runtime value. Delegating to ``__getitem__`` here keeps the annotation
    correct without tripping static type checkers.
    """
    return Optional[tp]  # ty: ignore[invalid-type-form]


_SCALAR_TYPE_NAMES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def make_field_param(
    leaf: SpecLeafField,
    *,
    rich_help_panel: str,
) -> inspect.Parameter:
    """Build the synthetic Typer parameter for one ``spec_schema`` leaf field.

    The flag defaults to the :data:`UNSET` sentinel so the overlay
    builder can tell "user passed --flag" from "user did not pass
    --flag".  Required fields are flagged with a ``[required]`` prefix
    in the help text — Click can't enforce required-ness on a flag
    that defaults to ``None`` without losing the "did the user pass
    it?" distinction, so the actual enforcement happens server-side
    when Pydantic validates the merged spec.
    """
    annotation: Any = _optional_of(leaf.python_type)
    help_text = leaf.description or f"Override `{'.'.join(leaf.path)}` in the spec."
    if leaf.partial:
        scalar_name = _SCALAR_TYPE_NAMES.get(leaf.python_type, leaf.python_type.__name__)
        help_text = (
            f"{help_text} This flag accepts the {scalar_name} form only; use --spec or --spec-file for the other union form(s)."
        ).strip()
    if leaf.required and leaf.default is PydanticUndefined:
        help_text = f"[required] {help_text}".strip()
    option_kwargs: dict[str, Any] = {
        "help": help_text,
        "show_default": False,
        "rich_help_panel": rich_help_panel,
    }
    if leaf.metavar is not None:
        # Click wraps the metavar in angle brackets when rendering the
        # help line, so we pass the raw label (e.g. "PATH | FILESET_REF")
        # and let it render as ``<PATH | FILESET_REF>``.
        option_kwargs["metavar"] = leaf.metavar
    option = typer.Option(UNSET, leaf.flag, **option_kwargs)
    return inspect.Parameter(
        name=leaf.param_name,
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=option,
        annotation=annotation,
    )


def build_callback_signature(
    static_params: Sequence[inspect.Parameter],
    leaves: Sequence[SpecLeafField],
    *,
    rich_help_panel: str,
    trailing_params: Sequence[inspect.Parameter] = (),
) -> inspect.Signature:
    """Compose a Typer-readable signature: static + per-field + trailing.

    Typer reads ``__signature__`` to discover CLI options, so attaching
    a synthetic signature lets us splice in one parameter per leaf
    without writing an explicit function definition. The actual
    callback accepts ``**kwargs`` and pulls values out by name.

    Args:
        static_params: Hand-written Typer flags (``--spec`` / ``--spec-file``
            / verb-specific flags) that come before the auto-generated
            block in the help output.
        leaves: Per-field flags to inject, from :func:`walk_spec_leaves`.
        rich_help_panel: Panel name applied to every auto-generated
            per-field flag (e.g. ``"Function Spec"`` / ``"Job Spec"``).
        trailing_params: Optional flags that come after the
            auto-generated block (e.g. hidden ``--config`` aliases).
    """
    params: list[inspect.Parameter] = list(static_params)
    for leaf in leaves:
        params.append(make_field_param(leaf, rich_help_panel=rich_help_panel))
    params.extend(trailing_params)
    return inspect.Signature(parameters=params)


# ---------------------------------------------------------------------------
# Help-epilog templates. Lifted out of the verb wiring so PR #160's jobs
# CLI and this PR's function CLI can share the exact same wording when
# the post-merge unification happens.
# ---------------------------------------------------------------------------


_EPILOG_WITH_FLAGS = (
    "{kind} Spec flags are generated from the {schema} Pydantic schema. "
    "Precedence: --spec-file (base) → --spec JSON (overlay) → per-flag "
    "values (top)."
)
_EPILOG_NO_FLAGS = (
    "Pass the spec via --spec '<json>' or --spec-file <path>. "
    "{kind} has no spec_schema fields the CLI knows how to expose, so "
    "no per-field flags are generated."
)
_UNAVAILABLE_NOTE = (
    "Some spec fields cannot be represented as CLI flags: {fields}. "
    "Supply them with --spec or --spec-file. Individual flags override "
    "values from the supplied spec."
)


def build_epilog(
    *,
    schema: type[BaseModel] | None,
    leaves: Sequence[SpecLeafField],
    kind: str,
    unavailable: Sequence[str] = (),
) -> str:
    """Compose the multi-line block rendered under all panels in ``--help``.

    The epilog is currently the only Typer hook for explaining what a
    panel means — Typer/rich-click panels themselves are name-only.
    A trailing zero-width space keeps a blank line under the body so
    the rendered output has breathing room before the shell prompt
    (Rich strips trailing whitespace-only lines from epilogs).

    Args:
        schema: The Pydantic ``spec_schema`` driving the auto-generated
            flags, or ``None`` when the verb has no schema in scope.
        leaves: The walked leaves; if empty, the "no flags" template is
            used instead. (A schema with only unsupported fields counts
            as "no flags".)
        kind: Capitalised label inserted into both templates — typically
            ``"Function"`` or ``"Job"``.
        unavailable: Dotted paths (from :func:`walk_spec_leaves`'s
            ``unavailable`` out-param) not fully representable as
            flags — fields skipped entirely, and the non-scalar arm of
            each partial mixed-union leaf. Rendered as an extra
            paragraph naming them so their absence from --help isn't
            silent.
    """
    body = (
        _EPILOG_WITH_FLAGS.format(kind=kind, schema=schema.__name__)
        if (leaves and schema is not None)
        else _EPILOG_NO_FLAGS.format(kind=kind)
    )
    if unavailable:
        fields = ", ".join(f"`{path}`" for path in unavailable)
        body = f"{body}\n\n{_UNAVAILABLE_NOTE.format(fields=fields)}"
    return body + "\n\u200b"
