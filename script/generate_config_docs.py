#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate config reference markdown from service config classes.

Imports platform and service config classes directly; requires running under uv
so workspace packages (nhx.common, nhx.core.*, etc.) are on the import path.

Outputs:
  - docs/set-up/config-reference.mdx  Fern MDX with YAML sections and inline comments

Usage (run from repository root):

  # Recommended: run via uv (uses project venv and workspace packages)
  uv run generate-config-docs

  # Or run the script file with uv
  uv run python script/generate_config_docs.py

  # Options
  uv run generate-config-docs --help
  uv run generate-config-docs --output-to-file  # Also write standalone example YAML to packages/nhx_platform/config/example-config.yaml
  uv run generate-config-docs --output-to-file /path/example.yaml
  uv run generate-config-docs --output-dir /path
  uv run generate-config-docs --markdown docs/ref.md

  # Without uv: ensure workspace packages are installed, then from repo root:
  python -m script.generate_config_docs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from nemo_safe_synthesizer_plugin.config import SafeSynthesizerConfig
from nhx.automodel.config import AutomodelConfig
from nhx.common.config.base import CommonServiceConfig, PlatformConfig
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.entities.config import EntitiesConfig
from nhx.core.files.config import FilesConfig
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from nhx.core.jobs.config import JobsServiceConfig
from nhx.core.models.config import ModelsConfig
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.studio.config import StudioConfig
from nhx.unsloth.config import UnslothConfig
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Repo root (parent of script/) — used for default output paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# All config classes to include in the reference, in display order.
# Excluded services (e.g. hello_world) are omitted from this list.
CONFIG_CLASSES: list[type[Any]] = [
    PlatformConfig,
    CommonServiceConfig,
    AuthServiceConfig,
    EntitiesConfig,
    FilesConfig,
    InferenceGatewayConfig,
    JobsServiceConfig,
    ModelsConfig,
    SecretsServiceConfig,
    AutomodelConfig,
    UnslothConfig,
    SafeSynthesizerConfig,
    StudioConfig,
]


def _load_config_classes() -> list[tuple[str, Any]]:
    """Return (global_settings_key, config_class) for each CONFIG_CLASSES entry."""
    return [(cls.global_settings_key(), cls) for cls in CONFIG_CLASSES]


def _get_possible_values_from_annotation(ann: Any) -> str | None:
    """Extract possible values from Literal or Enum annotation."""
    if ann is None:
        return None
    origin = get_origin(ann) if ann is not None else None
    args = get_args(ann) if ann is not None else ()
    if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Literal" and args:
        return " | ".join(repr(a) for a in args)
    try:
        if isinstance(ann, type) and hasattr(ann, "__members__"):
            members = getattr(ann, "__members__", None)
            if members is not None:
                return " | ".join(repr(m.value) for m in members.values())
    except TypeError:
        pass
    return None


def _get_nested_model_class(annotation: Any) -> type | None:
    """Resolve annotation to a nested Pydantic BaseModel class if any.

    Handles Annotated, Union, and dict[K, V] (returns the value type V when it is a model).
    """
    if annotation is None:
        return None
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation) or ()
    if origin is not None:
        # Annotated[X, ...] -> use X
        if str(origin) == "typing.Annotated" and args:
            return _get_nested_model_class(args[0])
        # dict[K, V] -> use V so we can document the structure of dict values (e.g. backends)
        if origin is dict or getattr(origin, "__name__", None) == "dict":
            if len(args) >= 2:
                return _get_nested_model_class(args[1])
            return None
        if str(origin) == "typing.Union" or (
            hasattr(origin, "__name__") and "Union" in getattr(origin, "__name__", "")
        ):
            for a in args:
                if a is type(None):
                    continue
                nested = _get_nested_model_class(a)
                if nested is not None:
                    return nested
            return None
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    except TypeError:
        pass
    return None


def _scalar_default_str(default: Any) -> str:
    """Short string for default value in comments (no class names)."""
    if default is None:
        return "null"
    if hasattr(default, "value"):
        return repr(getattr(default, "value", default))
    if isinstance(default, (str, int, float, bool)):
        return repr(default) if isinstance(default, str) else str(default)
    if isinstance(default, (list, dict)) and len(default) == 0:
        return "[]" if isinstance(default, list) else "{}"
    return ""


def _build_field_info_tree(model_class: type) -> dict[str, Any]:
    """Recursively build tree of field name -> {description, default, possible, nested?} for all nested models."""
    out: dict[str, Any] = {}
    try:
        fields = getattr(model_class, "model_fields", {})
    except Exception:
        return out
    annotations = getattr(model_class, "__annotations__", {})
    for fname, finfo in fields.items():
        if getattr(finfo, "exclude", None) is True:
            continue
        if (getattr(finfo, "json_schema_extra", None) or {}).get("exclude_from_docs") is True:
            continue
        desc = (getattr(finfo, "description", None) or "").strip().replace("\n", " ")
        default = getattr(finfo, "default", None)
        # Use FieldInfo.annotation (Pydantic v2) so Enum/Literal are resolved correctly
        ann = getattr(finfo, "annotation", None) or annotations.get(fname)
        possible = _get_possible_values_from_annotation(ann)
        nested_class = _get_nested_model_class(ann) if ann is not None else None
        entry: dict[str, Any] = {
            "description": desc or "",
            "default": default,
            "possible": possible,
        }
        if nested_class is not None:
            entry["nested"] = _build_field_info_tree(nested_class)
            # When this field is a dict of backends (e.g. models.controller.backends), add per-key trees
            backend_models = _get_backend_config_models(model_class, fname)
            if backend_models:
                entry["dict_value_trees"] = {name: _build_field_info_tree(cls) for name, cls in backend_models.items()}
        out[fname] = entry
    return out


def _format_comment(desc: str, default: Any, possible: str | None) -> str:
    """Single-line YAML comment: description, default, possible values (no class names). Caller adds indent."""
    parts: list[str] = []
    if desc:
        parts.append(desc)
    default_str = _scalar_default_str(default) if default is not None and not callable(default) else ""
    if default_str:
        parts.append(f"default: {default_str}")
    if possible:
        parts.append(f"values: {possible}")
    if not parts:
        return ""
    return "# " + " | ".join(parts)


def _to_commented_map(value: Any, info_tree: dict[str, Any], key_indent: int = 2) -> Any:
    """Build a ruamel CommentedMap from value dict with comments from info_tree (recursive).

    key_indent is the column used for comments so they align with keys at this nesting level.
    When the value is a dict whose keys are not in info_tree (e.g. backend name -> config),
    expand each value using the appropriate tree (dict_value_trees per key, or info_tree).
    """
    if not isinstance(value, dict):
        return value
    cm: CommentedMap = CommentedMap()
    for k, v in value.items():
        info = info_tree.get(k, {})
        nested = info.get("nested", {})
        dict_value_trees = info.get("dict_value_trees", {})
        if isinstance(v, dict) and v is not None:
            if nested:
                # Pass full entry when we have per-key trees (e.g. backends) so inner keys use correct tree
                recurse_tree = info if dict_value_trees else nested
                v = _to_commented_map(v, recurse_tree, key_indent=key_indent + 2)
            elif info_tree and not info:
                # Key not in tree (e.g. backend name); use parent's dict_value_trees per key, else this tree
                tree_for_value = info_tree.get("dict_value_trees", {}).get(k) or info_tree
                v = _to_commented_map(v, tree_for_value, key_indent=key_indent + 2)
        cm[k] = v
        comment = _format_comment(
            info.get("description", ""),
            info.get("default"),
            info.get("possible"),
        )
        if comment:
            # ruamel adds "# " when emitting; pass text without it
            comment_text = comment.lstrip("# ").strip()
            cm.yaml_set_comment_before_after_key(k, before=comment_text, indent=key_indent)
    return cm


def _dump_yaml_with_comments(data: Any) -> str:
    """Dump data to YAML string using ruamel (round-trip) so CommentedMap comments are emitted."""
    yaml_rt = YAML(typ="rt")
    yaml_rt.indent(mapping=2, sequence=2)
    yaml_rt.width = 4096  # prevent long lines (e.g. comments, anchors) from wrapping onto new lines
    stream = StringIO()
    yaml_rt.dump(data, stream)
    return stream.getvalue()


def _emit_section_yaml_with_comments(
    section_key: str,
    value: dict[str, Any],
    info_tree: dict[str, Any],
) -> list[str]:
    """Emit YAML for one top-level section with comments (via ruamel.yaml)."""
    commented = _to_commented_map(value, info_tree)
    section_map: CommentedMap = CommentedMap()
    section_map[section_key] = commented
    return _dump_yaml_with_comments(section_map).rstrip().splitlines()


def _customer_facing_doc(doc: str) -> str:
    """First sentence only; avoid internal/class references for customer-facing doc."""
    if not doc:
        return ""
    flat = " ".join(doc.split())
    first_sentence = flat.split(".")[0].strip()
    if first_sentence and not first_sentence.endswith("."):
        first_sentence += "."
    return first_sentence


def _collect_section_yaml(key: str, config_class: Any, values: dict[str, Any], info_tree: dict[str, Any]) -> list[str]:
    """Build markdown lines for one config section: brief doc + YAML block with comments (no tables, no class names)."""
    lines: list[str] = []
    lines.append(f"### `{key}`")
    lines.append("")
    doc = _customer_facing_doc((config_class.__doc__ or "").strip())
    if doc:
        lines.append(doc)
        lines.append("")
    lines.append("```yaml wordWrap")
    lines.extend(_emit_section_yaml_with_comments(key, values, info_tree))
    lines.append("```")
    lines.append("")
    return lines


def _get_backend_config_models(model_class: type, field_name: str) -> dict[str, type] | None:
    """Return backend name -> config class when this field is a known dict-of-backends (e.g. models.controller.backends)."""
    if field_name != "backends":
        return None
    try:
        from nhx.core.models.config import BACKEND_CONFIG_MODELS, ControllerConfig

        if model_class is ControllerConfig:
            return BACKEND_CONFIG_MODELS
    except Exception:
        pass
    return None


def _model_dump_for_yaml(model_class: type) -> dict[str, Any]:
    """Get default model as dict, suitable for YAML (enums/Paths as plain values).

    Uses ``model_construct`` rather than ``model_class()`` so post-init
    ``@model_validator`` hooks don't run. Those hooks coerce field values
    based on the host environment (e.g. ``NemoHelixConfig.validate_runtime``
    auto-demotes ``runtime: docker`` to ``runtime: none`` when the docker
    socket isn't reachable), which would leak the doc-builder's local
    environment into the static reference output and make the YAML value
    disagree with the documented default.
    """
    try:
        instance = model_class.model_construct()
        # mode='json' converts Enum to value, Path to str, so YAML has no Python tags
        return instance.model_dump(mode="json", exclude_none=False, exclude_unset=False)
    except Exception:
        return {}


def _filter_values_for_docs(model_class: type, value_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove keys marked exclude_from_docs so they do not appear in generated config reference."""
    if not isinstance(value_dict, dict):
        return value_dict
    try:
        fields = getattr(model_class, "model_fields", {})
    except Exception:
        return value_dict
    annotations = getattr(model_class, "__annotations__", {})
    out: dict[str, Any] = {}
    for k, v in value_dict.items():
        finfo = fields.get(k)
        if finfo is not None and (getattr(finfo, "json_schema_extra", None) or {}).get("exclude_from_docs") is True:
            continue
        if finfo is not None and getattr(finfo, "exclude", None) is True:
            continue
        if isinstance(v, dict) and finfo is not None:
            ann = getattr(finfo, "annotation", None) or annotations.get(k)
            nested_class = _get_nested_model_class(ann) if ann is not None else None
            if nested_class is not None:
                origin = get_origin(ann) if ann is not None else None
                # dict[K, Model]: value is a mapping; inject examples if empty so docs show structure
                if origin is dict or getattr(origin, "__name__", None) == "dict":
                    if not v:
                        try:
                            # Prefer registry of backend name -> config class when available (e.g. models backends)
                            backend_models = _get_backend_config_models(model_class, k)
                            if backend_models:
                                v = {
                                    name: _filter_values_for_docs(cls, cls().model_dump(mode="json"))
                                    for name, cls in backend_models.items()
                                }
                            else:
                                try:
                                    example_val = nested_class().model_dump(mode="json")
                                except Exception:
                                    # Model may require env or validation (e.g. SecretKeyEncryptorConfig); use defaults only
                                    example_val = nested_class.model_construct().model_dump(mode="json")
                                v = {"default": _filter_values_for_docs(nested_class, example_val)}
                        except Exception:
                            pass
                    else:
                        backend_models = _get_backend_config_models(model_class, k)
                        if backend_models:
                            v = {
                                k2: _filter_values_for_docs(backend_models.get(k2, nested_class), v2)
                                for k2, v2 in v.items()
                            }
                        else:
                            v = {k2: _filter_values_for_docs(nested_class, v2) for k2, v2 in v.items()}
                else:
                    v = _filter_values_for_docs(nested_class, v)
        out[k] = v
    return out


def _clean_dump_for_display(data: dict[str, Any]) -> dict[str, Any]:
    """Remove empty dicts/lists and None values for a shorter example; keep structure."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = _clean_dump_for_display(v)
            if not v:
                continue
        if isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out


def generate_yaml(entries: list[tuple[str, Any]]) -> str:
    """Generate example YAML file content (defaults only, no comments)."""
    example: dict[str, Any] = {}
    for key, config_class in entries:
        data = _model_dump_for_yaml(config_class)
        cleaned = _clean_dump_for_display(data)
        example[key] = cleaned if cleaned else {}
    return yaml.safe_dump(example, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_markdown(entries: list[tuple[str, Any]]) -> str:
    """Generate full markdown document: YAML-first with comments, no tables, no class names."""
    current_year = datetime.now().year
    lines: list[str] = []
    lines.append("---")
    lines.append(
        f"# SPDX-FileCopyrightText: Copyright (c) 2025-{current_year} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    )
    lines.append("# SPDX-License-Identifier: Apache-2.0")
    lines.append('title: "NeMo Helix configuration reference"')
    lines.append('description: ""')
    lines.append("---")
    lines.append("")
    lines.append(
        "This document describes the structure and defaults for the global config file for the NeMo Helix. "
        "All sections are shown in YAML format with inline comments for description, default, and possible values."
    )
    lines.append("")
    lines.append("## Configuration sections")
    lines.append("")

    for key, config_class in entries:
        values = _filter_values_for_docs(config_class, _model_dump_for_yaml(config_class))
        if not values:
            values = {}
        info_tree = _build_field_info_tree(config_class)
        lines.extend(_collect_section_yaml(key, config_class, values, info_tree))

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate config reference markdown from service config classes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory to write output file (default: repo root)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO_ROOT / "docs" / "set-up" / "config-reference.mdx",
        help="Output path for markdown file",
    )
    parser.add_argument(
        "--output-to-file",
        nargs="?",
        const=REPO_ROOT / "packages" / "nhx_platform" / "config" / "example-config.yaml",
        default=None,
        type=Path,
        metavar="PATH",
        help="Also write a standalone example YAML (defaults to packages/nhx_platform/config/example-config.yaml when flag is given with no path)",
    )
    args = parser.parse_args()

    entries = _load_config_classes()
    if not entries:
        print("No config classes loaded.", file=sys.stderr)
        return 1

    if args.output_dir != REPO_ROOT:
        args.markdown = args.output_dir / args.markdown.name

    md_content = generate_markdown(entries)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md_content, encoding="utf-8")
    print(f"Wrote {args.markdown}")

    if args.output_to_file is not None:
        yaml_content = generate_yaml(entries)
        args.output_to_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_to_file.write_text(yaml_content, encoding="utf-8")
        print(f"Wrote {args.output_to_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
