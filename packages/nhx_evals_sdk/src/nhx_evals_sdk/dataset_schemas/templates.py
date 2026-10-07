# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template parsing helpers for deriving canonical evaluator input schemas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from jinja2 import nodes
from jinja2.sandbox import SandboxedEnvironment
from jinja2.visitor import NodeVisitor
from nhx_evals_sdk.dataset_schemas.common import (
    ARRAY_TOKEN,
    TemplateSchemaInferenceError,
    empty_object_schema,
    validate_json_schema,
)

_UNSUPPORTED_TEMPLATE_NODES = (
    nodes.CallBlock,
    nodes.Import,
    nodes.Include,
    nodes.Macro,
)


@dataclass(frozen=True)
class _PathReference:
    root: str
    segments: tuple[str, ...]


_jinja_env = SandboxedEnvironment()


class _TemplateReferenceVisitor(NodeVisitor):
    def __init__(self, *, ignored_roots: set[str]) -> None:
        self.references: set[_PathReference] = set()
        self.ignored_roots = ignored_roots
        self.local_names: set[str] = set()

    def generic_visit(self, node: nodes.Node, *args: Any, **kwargs: Any) -> None:
        if isinstance(node, _UNSUPPORTED_TEMPLATE_NODES):
            raise TemplateSchemaInferenceError(
                f"unsupported Jinja construct for dataset schema inference: {type(node).__name__}"
            )
        super().generic_visit(node, *args, **kwargs)

    def visit_Output(self, node: nodes.Output, *args: Any, **kwargs: Any) -> None:
        for child in node.nodes:
            self.references.update(
                _collect_expression_references(
                    child,
                    ignored_roots=self.ignored_roots,
                    ignored_names=self.local_names,
                )
            )
            self.visit(child)

    def visit_For(self, node: nodes.For, *args: Any, **kwargs: Any) -> None:
        references = _collect_expression_references(
            node.iter,
            ignored_roots=self.ignored_roots,
            ignored_names=self.local_names,
        )
        if any(reference.root not in self.ignored_roots for reference in references):
            raise TemplateSchemaInferenceError("unsupported Jinja construct for dataset schema inference: For")
        loop_local_names = _collect_target_names(node.target)
        previous_local_names = set(self.local_names)
        self.local_names.update(loop_local_names)
        try:
            for body_node in node.body:
                self.visit(body_node)
            for else_node in node.else_:
                self.visit(else_node)
        finally:
            self.local_names = previous_local_names


def infer_required_schema_from_template(
    template: str | dict | list,
    *,
    ignored_roots: set[str] | None = None,
    optional_fields: set[str] | None = None,
) -> dict:
    """Infer a canonical evaluator input schema from a Jinja template structure.

    This inference intentionally accepts only a narrow subset of Jinja that maps cleanly to a
    canonical evaluator input contract. Dynamic indexing is rejected for schema inference. Control
    flow is rejected unless it only references roots explicitly listed in `ignored_roots`, which
    are treated as runtime-only context rather than dataset-provided inputs. Function/filter
    expressions are supported for dependency extraction, but callable identifiers are not treated
    as dataset-provided inputs.

    We intentionally do not support function-call argument unpacking (`*args` / `**kwargs`) for
    schema inference because this usage is uncommon in evaluator prompts and makes dependency
    extraction ambiguous. For example, `{{ sample.output_json.get(*item.lookup_path) }}` and
    `{{ sample.output_json.get(**item.lookup_kwargs) }}` are not considered supported schema
    inference patterns.
    """
    effective_ignored_roots = set(ignored_roots or ())
    effective_optional_fields = set(optional_fields or ())
    references = list(_extract_template_references(template, ignored_roots=effective_ignored_roots))
    schema = _build_schema_from_references(references, ignored_roots=effective_ignored_roots)
    _drop_optional_fields_from_required(schema, optional_fields=effective_optional_fields)
    validate_json_schema(schema)
    return schema


def _extract_template_references(template: str | dict | list, *, ignored_roots: set[str]) -> set[_PathReference]:
    if isinstance(template, dict):
        references: set[_PathReference] = set()
        for value in template.values():
            references.update(_extract_template_references(value, ignored_roots=ignored_roots))
        return references
    if isinstance(template, list):
        references = set()
        for value in template:
            references.update(_extract_template_references(value, ignored_roots=ignored_roots))
        return references
    if not isinstance(template, str):
        return set()

    parsed = _jinja_env.parse(template)
    visitor = _TemplateReferenceVisitor(ignored_roots=ignored_roots)
    visitor.visit(parsed)
    return visitor.references


def _collect_expression_references(
    node: nodes.Node,
    *,
    ignored_roots: set[str],
    ignored_names: set[str],
) -> set[_PathReference]:
    if isinstance(node, nodes.TemplateData | nodes.Const):
        return set()
    if isinstance(node, nodes.Filter):
        references = _collect_expression_references(
            node.node,
            ignored_roots=ignored_roots,
            ignored_names=ignored_names,
        )
        for arg in node.args:
            references.update(
                _collect_expression_references(
                    arg,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        for keyword in node.kwargs:
            references.update(
                _collect_expression_references(
                    keyword.value,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        return references
    if isinstance(node, nodes.Call):
        references = _collect_callable_base_references(node.node, ignored_names=ignored_names)
        references.update(
            _collect_call_argument_references(
                args=node.args,
                kwargs=node.kwargs,
                ignored_roots=ignored_roots,
                ignored_names=ignored_names,
            )
        )
        return references
    if isinstance(node, nodes.If):
        references = _collect_expression_references(
            node.test,
            ignored_roots=ignored_roots,
            ignored_names=ignored_names,
        )
        for body_node in node.body:
            references.update(
                _collect_expression_references(
                    body_node,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        for else_node in node.else_:
            references.update(
                _collect_expression_references(
                    else_node,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        if any(reference.root not in ignored_roots for reference in references):
            raise TemplateSchemaInferenceError("conditionals are not supported for dataset schema inference")
        return references
    if isinstance(node, nodes.Name | nodes.Getattr | nodes.Getitem):
        reference = _path_reference_from_node(node, ignored_names=ignored_names)
        return set() if reference is None else {reference}
    if isinstance(node, nodes.List | nodes.Tuple):
        references: set[_PathReference] = set()
        for item in node.items:
            references.update(
                _collect_expression_references(
                    item,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        return references
    if isinstance(node, nodes.Dict):
        references: set[_PathReference] = set()
        for item in node.items:
            references.update(
                _collect_expression_references(
                    item.key,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
            references.update(
                _collect_expression_references(
                    item.value,
                    ignored_roots=ignored_roots,
                    ignored_names=ignored_names,
                )
            )
        return references
    if isinstance(node, nodes.Pair):
        references = _collect_expression_references(
            node.key,
            ignored_roots=ignored_roots,
            ignored_names=ignored_names,
        )
        references.update(
            _collect_expression_references(
                node.value,
                ignored_roots=ignored_roots,
                ignored_names=ignored_names,
            )
        )
        return references
    if isinstance(node, _UNSUPPORTED_TEMPLATE_NODES):
        raise TemplateSchemaInferenceError(
            f"unsupported Jinja construct for dataset schema inference: {type(node).__name__}"
        )
    raise TemplateSchemaInferenceError(
        f"unsupported Jinja expression for dataset schema inference: {type(node).__name__}"
    )


def _collect_call_argument_references(
    *,
    args: list[nodes.Node],
    kwargs: list[nodes.Keyword],
    ignored_roots: set[str],
    ignored_names: set[str],
) -> set[_PathReference]:
    references: set[_PathReference] = set()
    for arg in args:
        references.update(
            _collect_expression_references(
                arg,
                ignored_roots=ignored_roots,
                ignored_names=ignored_names,
            )
        )
    for keyword in kwargs:
        references.update(
            _collect_expression_references(
                keyword.value,
                ignored_roots=ignored_roots,
                ignored_names=ignored_names,
            )
        )
    return references


def _collect_callable_base_references(node: nodes.Node, *, ignored_names: set[str]) -> set[_PathReference]:
    # For `input.upper()`, require `input`; for `upper(input)`, do not require `upper`.
    if isinstance(node, nodes.Getattr | nodes.Getitem):
        reference = _path_reference_from_node(node.node, ignored_names=ignored_names)
        return set() if reference is None else {reference}
    return set()


def _path_reference_from_node(node: nodes.Node, *, ignored_names: set[str]) -> _PathReference | None:
    if isinstance(node, nodes.Name):
        if node.name in ignored_names:
            return None
        return _PathReference(root=node.name, segments=())
    if isinstance(node, nodes.Getattr):
        base = _path_reference_from_node(node.node, ignored_names=ignored_names)
        if base is None:
            return None
        return _PathReference(root=base.root, segments=base.segments + (node.attr,))
    if isinstance(node, nodes.Getitem):
        base = _path_reference_from_node(node.node, ignored_names=ignored_names)
        if base is None:
            return None
        if not isinstance(node.arg, nodes.Const):
            raise TemplateSchemaInferenceError("dynamic indexing is not supported for dataset schema inference")
        if isinstance(node.arg.value, int):
            segment = ARRAY_TOKEN
        elif isinstance(node.arg.value, str):
            segment = node.arg.value
        else:
            raise TemplateSchemaInferenceError("unsupported index type for dataset schema inference")
        return _PathReference(root=base.root, segments=base.segments + (segment,))
    raise TemplateSchemaInferenceError(
        f"unsupported Jinja expression for dataset schema inference: {type(node).__name__}"
    )


def _collect_target_names(node: nodes.Node) -> set[str]:
    if isinstance(node, nodes.Name):
        return {node.name}
    if isinstance(node, nodes.Tuple):
        names: set[str] = set()
        for item in node.items:
            names.update(_collect_target_names(item))
        return names
    return set()


def _build_schema_from_references(
    references: Iterable[_PathReference],
    *,
    ignored_roots: set[str],
) -> dict:
    schema = empty_object_schema()

    for reference in references:
        path = _normalize_reference(reference, ignored_roots=ignored_roots)
        if path is not None:
            _add_required_path(schema, path)
    return schema


def _normalize_reference(
    reference: _PathReference,
    *,
    ignored_roots: set[str],
) -> tuple[str, ...] | None:
    if reference.root == "sample" and reference.segments == ("output_text",):
        return ("output",)
    if reference.root in ignored_roots:
        return None
    if reference.root == "item":
        return reference.segments or None
    return (reference.root,) + reference.segments


def _add_required_path(schema: dict, path: tuple[str, ...]) -> None:
    current = schema
    index = 0
    while index < len(path):
        segment = path[index]
        if segment == ARRAY_TOKEN:
            current["type"] = "array"
            current.setdefault("items", {})
            current = current["items"]
            index += 1
            continue

        current.setdefault("type", "object")
        properties = current.setdefault("properties", {})
        required = current.setdefault("required", [])
        if segment not in properties:
            properties[segment] = {}
        if segment not in required:
            required.append(segment)

        current = properties[segment]
        if index + 1 < len(path) and path[index + 1] != ARRAY_TOKEN:
            current.setdefault("type", "object")
        index += 1


def _drop_optional_fields_from_required(schema: dict, *, optional_fields: set[str]) -> None:
    if not optional_fields:
        return
    required = schema.get("required")
    if not isinstance(required, list):
        return
    schema["required"] = [field for field in required if field not in optional_fields]
