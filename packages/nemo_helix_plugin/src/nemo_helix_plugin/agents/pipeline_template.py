# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``{{ path }}`` placeholders for pipeline configs. Deliberately not Jinja: paths only, ``{% raw %}``
for literal text, and one rendering pass, so an inserted value is never treated as a template."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PathSegment = str | int

_RAW_OPEN = re.compile(r"\{%\s*raw\s*%\}")
_RAW_CLOSE = re.compile(r"\{%\s*endraw\s*%\}")
# Any spelling of a raw/endraw tag, including the whitespace-control forms ({%- ... -%}) we reject.
_ANY_RAW_TAG = re.compile(r"\{%(-?)\s*(raw|endraw)\s*(-?)%\}")
_TAG = re.compile(r"\{%-?\s*([A-Za-z_]*)")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_INDEX = re.compile(r"\[\s*(\d+)\s*\]")
_QUOTED_KEY = re.compile(r"""\[\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')\s*\]""")
# The JSON escape set, plus \' for single-quoted keys.
_ESCAPES = {"\\": "\\", '"': '"', "'": "'", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_HEX4 = re.compile(r"[0-9A-Fa-f]{4}")


_WHITESPACE_CONTROL = "Whitespace control ({%- or -%}) is not supported; write {% raw %} and {% endraw %}."


class TemplateError(ValueError):
    """A template could not be parsed or rendered."""


class TemplateSyntaxError(TemplateError):
    """A template is not valid placeholder syntax."""


class TemplateRenderError(TemplateError):
    """A placeholder's path does not resolve against the render context."""


@dataclass(frozen=True, slots=True)
class Text:
    """Literal text copied to the output as is."""

    value: str


@dataclass(frozen=True, slots=True)
class Placeholder:
    """One ``{{ path }}`` placeholder."""

    path: tuple[PathSegment, ...]
    source: str

    @property
    def dotted(self) -> str:
        return format_path(self.path)


Part = Text | Placeholder


def format_path(path: Sequence[PathSegment]) -> str:
    """Render a parsed path back to its canonical text form, e.g. ``steps.triage.response[0]``."""
    out = ""
    for index, segment in enumerate(path):
        if isinstance(segment, int):
            out += f"[{segment}]"
        elif _IDENT.fullmatch(segment):
            out += segment if index == 0 else f".{segment}"
        else:
            out += f"[{json.dumps(segment, ensure_ascii=False)}]"
    return out


def parse_path(text: str) -> tuple[PathSegment, ...]:
    """Parse a dotted path such as ``steps.triage.output.findings[0]["score"]``."""
    source = text.strip()
    if not source:
        raise TemplateSyntaxError("Empty placeholder: write a path such as {{ input }}.")
    first = _IDENT.match(source)
    if first is None:
        raise TemplateSyntaxError(f"Invalid placeholder path {source!r}: it must start with a name.")
    segments: list[PathSegment] = [first.group(0)]
    pos = first.end()
    while pos < len(source):
        char = source[pos]
        if char == ".":
            match = _IDENT.match(source, pos + 1)
            if match is None:
                raise TemplateSyntaxError(f"Invalid placeholder path {source!r}: expected a name after '.'.")
            segments.append(match.group(0))
            pos = match.end()
        elif char == "[":
            index = _INDEX.match(source, pos)
            if index is not None:
                segments.append(int(index.group(1)))
                pos = index.end()
                continue
            quoted = _QUOTED_KEY.match(source, pos)
            if quoted is None:
                raise TemplateSyntaxError(
                    f'Invalid placeholder path {source!r}: use [0] for a list index or ["key"] for a key.'
                )
            raw_key = quoted.group(1) if quoted.group(1) is not None else quoted.group(2)
            segments.append(_unescape(raw_key, source))
            pos = quoted.end()
        else:
            raise TemplateSyntaxError(_unsupported_expression(source))
    return tuple(segments)


def _unescape(raw_key: str, source: str) -> str:
    """Decode a quoted key's JSON escapes (and \\' ), the same for both quote styles."""
    out = []
    pos = 0
    while pos < len(raw_key):
        char = raw_key[pos]
        if char != "\\":
            out.append(char)
            pos += 1
            continue
        escaped = raw_key[pos + 1 : pos + 2]
        if escaped == "u" and _HEX4.fullmatch(raw_key[pos + 2 : pos + 6]):
            out.append(chr(int(raw_key[pos + 2 : pos + 6], 16)))
            pos += 6
            continue
        if escaped not in _ESCAPES:
            raise TemplateSyntaxError(f"Invalid escape '\\{escaped}' in placeholder path {source!r}.")
        out.append(_ESCAPES[escaped])
        pos += 2
    return "".join(out)


def _placeholder_end(template: str, start: int) -> int:
    """Index of the ``}}`` closing the placeholder whose body starts at *start*, skipping quoted keys."""
    pos = start
    quote: str | None = None
    while pos < len(template):
        char = template[pos]
        if quote is not None:
            if char == "\\":
                pos += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'" and template[start:pos].rstrip().endswith("["):
            # Quotes only start a string inside a bracketed key; elsewhere parse_path reports them.
            quote = char
        elif template.startswith("}}", pos):
            return pos
        pos += 1
    return -1


def _unsupported_expression(source: str) -> str:
    if "|" in source:
        return f"Filters are not supported in placeholders ({{{{ {source} }}}}): use a plain path."
    return f"Expressions are not supported in placeholders ({{{{ {source} }}}}): use a plain path."


def parse(template: str) -> tuple[Part, ...]:
    """Split *template* into literal text and placeholders, rejecting unsupported syntax."""
    parts: list[Part] = []
    buffer: list[str] = []
    pos = 0
    length = len(template)

    def flush() -> None:
        if buffer:
            parts.append(Text("".join(buffer)))
            buffer.clear()

    while pos < length:
        if template.startswith("{{", pos):
            end = _placeholder_end(template, pos + 2)
            if end == -1:
                raise TemplateSyntaxError("Unclosed placeholder: '{{' has no matching '}}'.")
            source = template[pos + 2 : end]
            flush()
            parts.append(Placeholder(parse_path(source), source.strip()))
            pos = end + 2
        elif template.startswith("{%", pos):
            raw_open = _RAW_OPEN.match(template, pos)
            if raw_open is None:
                any_raw = _ANY_RAW_TAG.match(template, pos)
                if any_raw and (any_raw.group(1) or any_raw.group(3)):
                    raise TemplateSyntaxError(_WHITESPACE_CONTROL)
                tag = _TAG.match(template, pos)
                name = tag.group(1) if tag and tag.group(1) else "tag"
                if name == "endraw":
                    raise TemplateSyntaxError("{% endraw %} has no matching {% raw %}.")
                raise TemplateSyntaxError(
                    f"{{% {name} %}} is not supported: only {{% raw %}}...{{% endraw %}} blocks are allowed."
                )
            close = _RAW_CLOSE.search(template, raw_open.end())
            if close is None:
                loose = _ANY_RAW_TAG.search(template, raw_open.end())
                if loose and loose.group(2) == "endraw":
                    raise TemplateSyntaxError(_WHITESPACE_CONTROL)
                raise TemplateSyntaxError("Unclosed {% raw %} block: add {% endraw %}.")
            body = template[raw_open.end() : close.start()]
            if _RAW_OPEN.search(body):
                raise TemplateSyntaxError("{% raw %} blocks cannot be nested.")
            buffer.append(body)
            pos = close.end()
        elif template.startswith("{#", pos):
            raise TemplateSyntaxError("Comments ({# ... #}) are not supported in templates.")
        else:
            buffer.append(template[pos])
            pos += 1
    flush()
    return tuple(parts)


def placeholders(template: str) -> tuple[Placeholder, ...]:
    """The placeholders in *template*, in order."""
    return tuple(part for part in parse(template) if isinstance(part, Placeholder))


def single_placeholder(template: str) -> Placeholder | None:
    """The placeholder when *template* is exactly one placeholder (surrounding spaces allowed)."""
    parts = [part for part in parse(template) if not (isinstance(part, Text) and not part.value.strip())]
    if len(parts) == 1 and isinstance(parts[0], Placeholder):
        return parts[0]
    return None


def resolve(path: Sequence[PathSegment], context: Mapping[str, Any]) -> Any:
    """Look *path* up in *context*. A missing key or index raises :class:`TemplateRenderError`."""
    value: Any = context
    for depth, segment in enumerate(path):
        if isinstance(segment, int):
            if not isinstance(value, list) or segment >= len(value):
                raise TemplateRenderError(f"{format_path(path)}: no item [{segment}] at {format_path(path[:depth])}.")
            value = value[segment]
        else:
            if not isinstance(value, Mapping) or segment not in value:
                where = format_path(path[:depth]) or "the context"
                raise TemplateRenderError(f"{format_path(path)}: no key {segment!r} in {where}.")
            value = value[segment]
    return value


def to_text(value: Any) -> str:
    """How a value is written into text: strings as is, everything else as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def render(template: str, context: Mapping[str, Any]) -> str:
    """Fill every placeholder in *template* from *context*, in one pass."""
    return "".join(
        part.value if isinstance(part, Text) else to_text(resolve(part.path, context)) for part in parse(template)
    )


def render_value(template: str, context: Mapping[str, Any]) -> Any:
    """Like :func:`render`, but a template that is exactly one placeholder keeps the value's JSON type."""
    only = single_placeholder(template)
    if only is not None:
        return resolve(only.path, context)
    return render(template, context)


def render_tree(value: Any, context: Mapping[str, Any]) -> Any:
    """Render every string inside *value* (objects and lists at any depth); keys are not templates."""
    if isinstance(value, str):
        return render_value(value, context)
    if isinstance(value, Mapping):
        return {key: render_tree(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render_tree(item, context) for item in value]
    return value


def tree_strings(value: Any, location: str) -> list[tuple[str, str]]:
    """Every string inside *value* with its location, e.g. ``("context.a[0]", "{{ input }}")``."""
    if isinstance(value, str):
        return [(location, value)]
    if isinstance(value, Mapping):
        found: list[tuple[str, str]] = []
        for key, item in value.items():
            found.extend(tree_strings(item, f"{location}.{key}"))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found.extend(tree_strings(item, f"{location}[{index}]"))
        return found
    return []
