# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe, unambiguous YAML (or JSON) authoring for deployment-independent plugins."""

from __future__ import annotations

import json

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode

MAX_MANIFEST_BYTES = 256 * 1024


class _ManifestLoader(yaml.SafeLoader):
    def __init__(self, stream: bytes) -> None:
        super().__init__(stream)
        self._depth = 0
        self._nodes = 0

    def compose_node(self, parent: Node | None, index: object) -> Node:
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise yaml.YAMLError("Manifest anchors and aliases are not supported")
        self._depth += 1
        self._nodes += 1
        try:
            if self._depth > 48 or self._nodes > 10000:
                raise yaml.YAMLError("Manifest is too deeply nested or complex")
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        keys: set[str] = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                raise yaml.YAMLError("Manifest mapping keys must be strings")
            key = key_node.value
            if key in keys or key == "<<":
                raise yaml.YAMLError("Duplicate or merged manifest keys are not supported")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_manifest_document(data: bytes) -> object:
    """No Python constructors, aliases, merges, duplicate keys, or non-JSON values.

    YAML is an authoring format only. Validation and consent digests operate on
    the same normalized data model as JSON, not on whitespace or comments.
    """
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds its size limit")
    try:
        loader = _ManifestLoader(data)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML/JSON encoding") from exc
    try:
        value = loader.get_single_data()
        # Reject YAML-only binary/timestamp/set values and non-finite numbers.
        json.dumps(value, allow_nan=False)
        return value
    except (yaml.YAMLError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid or unsupported YAML/JSON manifest") from exc
    finally:
        loader.dispose()
