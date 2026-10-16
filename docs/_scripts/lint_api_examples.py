# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate explicit OpenAPI operation examples and bodyless response samples.

Run with `uv run --frozen python docs/_scripts/lint_api_examples.py [spec.yaml ...]`.
Required object fields must be nonempty in documentation examples even when {} is
schema-valid: Fern can omit those fields from generated request snippets.
Schema-level examples are not checked here; operation examples are the published
request/response examples and can intentionally override schema-level defaults.
"""

import argparse
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError, validators

_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
_BODY_READ = re.compile(
    r"\.(?:json|text|arrayBuffer|blob)\s*\(|JSON\.parse\s*\("
    r"|io\.ReadAll\s*\([^)]*\.Body\b|json\.NewDecoder\s*\([^)]*\.Body\b"
    r"|\bread_body\b|(?:->|\.)getBody\s*\("
)


def _required(validator: Any, required: list[str], instance: Any, schema: dict) -> Iterator[ValidationError]:
    yield from Draft202012Validator.VALIDATORS["required"](validator, required, instance, schema)
    if isinstance(instance, dict):
        for name in required:
            if name in instance and instance[name] == {}:
                yield ValidationError(f"required object {name!r} is empty; Fern may omit it from snippets")


ExampleValidator = validators.extend(Draft202012Validator, {"required": _required})


def lint_spec(spec: dict) -> list[str]:
    """Return diagnostics for explicit media-type and parameter examples."""
    errors: list[str] = []

    def resolve(node: dict) -> dict:
        seen = set()
        while "$ref" in node:
            ref = node["$ref"]
            if ref in seen:
                raise ValueError(f"Cyclic OpenAPI reference: {ref}")
            seen.add(ref)
            if not ref.startswith("#/"):
                raise ValueError(f"Only local OpenAPI references are supported: {ref}")
            node = spec
            for part in ref[2:].split("/"):
                node = node[part.replace("~1", "/").replace("~0", "~")]
        return node

    def check(node: dict, label: str) -> None:
        node = resolve(node)
        if "schema" not in node:
            return
        # Keep component references rooted in the original OpenAPI document.
        validator = ExampleValidator({**spec, "allOf": [node["schema"]]})
        examples = []
        if "example" in node:
            examples.append(("example", node["example"]))
        for name, example in node.get("examples", {}).items():
            example = resolve(example)
            if "value" in example:
                examples.append((name, example["value"]))
        for name, value in examples:
            for error in validator.iter_errors(value):
                location = ".".join(str(part) for part in error.absolute_path) or "$"
                errors.append(f"{label} example {name} at {location}: {error.message}")

    for path, item in spec.get("paths", {}).items():
        item = resolve(item)
        for method, operation in item.items():
            if method not in _METHODS:
                continue
            label = f"{method.upper()} {path}"
            for parameter in [*item.get("parameters", []), *operation.get("parameters", [])]:
                check(parameter, f"{label} parameter")
            body = resolve(operation.get("requestBody", {}))
            for media, node in body.get("content", {}).items():
                check(node, f"{label} request {media}")
            for code, response in operation.get("responses", {}).items():
                for media, node in resolve(response).get("content", {}).items():
                    check(node, f"{label} response {code} {media}")
            success = [str(code) for code in operation.get("responses", {}) if str(code).startswith("2")]
            if method == "head" or (success and all(code in {"204", "205"} for code in success)):
                for sample in operation.get("x-codeSamples", []):
                    if _BODY_READ.search(sample.get("source", "")):
                        errors.append(f"{label} {sample.get('lang', '')} sample reads a bodyless response")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="*", type=Path, default=[Path("openapi/openapi.yaml")])
    args = parser.parse_args()
    errors = []
    for path in args.specs:
        try:
            errors.extend(f"{path}: {error}" for error in lint_spec(yaml.safe_load(path.read_text())))
        except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
            errors.append(f"{path}: unable to validate spec: {error}")
    for error in errors:
        print(error)
    print(f"API examples: {len(args.specs)} spec(s), {len(errors)} error(s)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
