#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate Automodel YAML configs from TrainingStepConfig JSON fixtures.

Uses compile_automodel_config() from nhx-automodel. Input configs are grouped
by model in subdirectories of input_configs/ so each model is downloaded only once.

Directory layout:
    input_configs/
        llama-3.2-1b/          # one subdirectory per model
            llama_3_2_1b_lora.json
            llama_3_2_1b_full_sft.json
            ...
    output_configs/            # flat directory, one YAML per input JSON
        llama_3_2_1b_lora.yaml
        ...

Usage:
    # Single config
    python generate_configs.py input_configs/llama-3.2-1b/llama_3_2_1b_lora.json \
        -o output_configs/llama_3_2_1b_lora.yaml

    # Regenerate all configs
    python generate_configs.py --all

    # CI mode: regenerate all and fail if any output differs from committed version
    python generate_configs.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
AUTOMODEL_SRC = REPO_ROOT / "services" / "automodel" / "src"

if AUTOMODEL_SRC.is_dir():
    sys.path.insert(0, str(AUTOMODEL_SRC))
else:
    sys.path.insert(0, "/app/services/automodel/src")

from nhx.automodel.app.constants import V4_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # noqa: E402
from nhx.automodel.tasks.training.backends.config import compile_automodel_config  # noqa: E402
from nhx.automodel.tasks.training.schemas import TrainingStepConfig  # noqa: E402
from nhx.customization_common.service.context import NHXJobContext  # noqa: E402

INPUT_DIR = SCRIPT_DIR / "input_configs"
OUTPUT_DIR = SCRIPT_DIR / "output_configs"

# Files to download from HF -- config + tokenizer only, not multi-GB weights
MODEL_METADATA_PATTERNS = [
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
]


def _enum_representer(dumper: yaml.Dumper, data: Enum) -> yaml.Node:
    return dumper.represent_str(str(data.value))


def _none_representer(dumper: yaml.Dumper, data: None) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


yaml.add_multi_representer(Enum, _enum_representer)
yaml.add_representer(type(None), _none_representer)


def _download_model_metadata(model_id: str, target_dir: Path, trust_remote_code: bool = False) -> Path:
    """Download only model config and tokenizer files (not weights) from HuggingFace."""
    from huggingface_hub import snapshot_download

    allow_patterns = list(MODEL_METADATA_PATTERNS)
    if trust_remote_code:
        allow_patterns.append("*.py")

    local_dir = target_dir / model_id.replace("/", "--")
    return Path(
        snapshot_download(
            model_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
    )


def _delete_hf_cache(model_id: str) -> None:
    """Delete HuggingFace hub cache for a model to free disk space."""
    try:
        hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        cache_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub")))
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache = cache_dir / cache_name
        if model_cache.is_dir():
            shutil.rmtree(model_cache)
    except Exception:
        pass


def _replace_paths(config_dict: dict[str, Any], replacements: dict[str, str]) -> None:
    """Replace path strings in a nested config dict.

    Swaps temp workspace paths back to portable values:
    - Temp dataset paths (/tmp/.../dataset/file) -> original dataset dir
    - Local model download path -> HF model name
    - Merged dataset filenames (train.jsonl) -> original filenames (training.jsonl)
    """
    tmp_dataset_pattern = re.compile(r"/tmp/[^/]+/dataset/([^/]+)")

    # prepare_dataset() renames files during merge; map back to originals
    merged_to_original = {
        "train.jsonl": "training.jsonl",
    }

    def _replace(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_replace(item) for item in obj]
        elif isinstance(obj, str):
            result = tmp_dataset_pattern.sub(
                lambda m: f"{replacements.get('__dataset_dir__', '')}/{merged_to_original.get(m.group(1), m.group(1))}",
                obj,
            )
            for old, new in replacements.items():
                if old.startswith("__"):
                    continue
                result = result.replace(old, new)
            return result
        return obj

    replaced = _replace(config_dict)
    config_dict.clear()
    config_dict.update(replaced)


def _detect_model_flags(model_local_dir: Path) -> dict[str, bool]:
    """Auto-detect model flags from HF config.json.

    In production, these are resolved from ModelEntity (model registry).
    For contract test generation we derive them from the downloaded config.json,
    which contains the same architecture class name.
    """
    config_file = model_local_dir / "config.json"
    if not config_file.exists():
        return {}

    with open(config_file) as f:
        hf_config = json.load(f)

    architectures = hf_config.get("architectures", [])
    arch_name = architectures[0] if architectures else None

    flags: dict[str, bool] = {}

    if arch_name and arch_name in V4_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        flags["v4_compatible"] = True

    if arch_name == "NemotronHForCausalLM" and "moe" not in hf_config:
        flags["override_custom_impl"] = True

    return flags


def _apply_lora_defaults(config: TrainingStepConfig) -> None:
    """Apply production-equivalent LoRA defaults.

    In production, _translate_lora_config() (compiler.py) sets target_modules
    when None.  The test script bypasses that code path, so we replicate the
    default here.  Without this, the generated YAML contains
    ``target_modules: null`` which causes Automodel to raise:
        ValueError("Expected match_all_linear to be true or
                    target_modules/exclude_modules to be non-empty")
    """
    lora = config.training.lora
    if lora is None or lora.target_modules:
        return

    lora.target_modules = ["*proj"]


def _compile_one(input_path: Path, model_local_dir: Path) -> dict[str, Any]:
    """Compile a single input config using an already-downloaded model directory."""
    with open(input_path) as f:
        raw = json.load(f)

    model_name = raw["model"]["name"]
    dataset_dir = raw.get("dataset", {}).get("path", "")

    raw["model"]["path"] = str(model_local_dir)

    for flag, value in _detect_model_flags(model_local_dir).items():
        raw["model"].setdefault(flag, value)

    config = TrainingStepConfig.model_validate(raw)
    _apply_lora_defaults(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        job_ctx = NHXJobContext(
            workspace="default",
            job_id="config-generation",
            attempt_id="attempt-0",
            step="compile-config",
            task="task-config-generation",
            jobs_url=None,
            files_url=None,
            storage_path=Path(tmpdir),
            config_path=input_path,
        )
        compiled = compile_automodel_config(config, Path(tmpdir), job_ctx)

    # Keys we add to the recipe config that are not the recipe's. The golden
    # files describe the contract with nemo_automodel, so ours would only make
    # them churn whenever our own reporting shape changed -- and, worse, would
    # do it silently: the check reports a mismatch, the mismatch names the
    # embedding configs, and the branch below skips the whole test on that
    # basis. `_progress_reporting` did exactly that when it was added.
    for _ours in ("_resolved_chat_template", "_progress_reporting", "_recipe"):
        compiled.pop(_ours, None)
    _replace_paths(
        compiled,
        {
            str(model_local_dir): model_name,
            "__dataset_dir__": dataset_dir,
        },
    )
    if "checkpoint" in compiled and "checkpoint_dir" in compiled["checkpoint"]:
        compiled["checkpoint"]["checkpoint_dir"] = "./checkpoints"

    return compiled


def _read_model_info(config_path: Path) -> tuple[str, bool]:
    """Read model name and trust_remote_code from an input config."""
    with open(config_path) as f:
        raw = json.load(f)
    return raw["model"]["name"], raw.get("model", {}).get("trust_remote_code", False)


def _dump_yaml(config_dict: dict[str, Any]) -> str:
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def _write_config(compiled: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(_dump_yaml(compiled))


def _discover_model_groups() -> list[tuple[str, list[Path]]]:
    """Find all model groups: (group_name, [config_paths]).

    Each subdirectory of input_configs/ is a model group containing one or
    more JSON configs that share the same model.
    """
    groups = []
    for model_dir in sorted(INPUT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        configs = sorted(model_dir.glob("*.json"))
        if configs:
            groups.append((model_dir.name, configs))
    return groups


def _process_model_group(
    group_name: str,
    config_paths: list[Path],
    write: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    """Download model once, compile all configs in the group, clean up.

    Args:
        group_name: Name of the model group directory.
        config_paths: Input JSON config files for this model.
        write: If True, write output YAML files. If False, just return compiled dicts.

    Returns:
        List of (output_path, compiled_dict) pairs.
    """
    model_name, trust_remote_code = _read_model_info(config_paths[0])
    print(f"[{group_name}] Downloading model metadata for {model_name}")

    results = []
    with tempfile.TemporaryDirectory() as model_tmpdir:
        model_local_dir = _download_model_metadata(model_name, Path(model_tmpdir), trust_remote_code=trust_remote_code)

        for input_path in config_paths:
            output_path = OUTPUT_DIR / f"{input_path.stem}.yaml"
            compiled = _compile_one(input_path, model_local_dir)
            if write:
                _write_config(compiled, output_path)
            results.append((output_path, compiled))
            print(f"    {input_path.name} -> {output_path.name}")

    _delete_hf_cache(model_name)
    print(f"[{group_name}] Done ({len(config_paths)} configs, cache cleaned)")
    return results


def _generate_all() -> None:
    """Regenerate all output configs, downloading each model once."""
    groups = _discover_model_groups()
    if not groups:
        print(f"No model groups found in {INPUT_DIR}/")
        sys.exit(1)

    total = sum(len(configs) for _, configs in groups)
    print(f"Regenerating {total} configs across {len(groups)} models...")
    for group_name, config_paths in groups:
        _process_model_group(group_name, config_paths, write=True)
    print(f"Done. {total} configs written to {OUTPUT_DIR}/")


def _check_all(exclude: set[str] | None = None) -> None:
    """Regenerate all configs and fail if any differ from the committed version.

    ``exclude`` names config stems whose mismatches are known and tolerated. It
    is an argument rather than a constant here because the caller owns the list,
    and it filters *which configs are compared* rather than which failures are
    forgiven -- the distinction matters. The test used to scan this function's
    output for an excluded name and skip the whole run if it found one, so any
    mismatch that happened to list an excluded config alongside a real one
    disabled the check entirely and said nothing. Adding a key to every compiled
    config did exactly that.
    """
    exclude = exclude or set()
    groups = _discover_model_groups()
    if not groups:
        print(f"No model groups found in {INPUT_DIR}/")
        sys.exit(1)

    total = sum(len(configs) for _, configs in groups) - len(exclude)
    print(f"Checking {total} configs across {len(groups)} models...")
    if exclude:
        print(f"Excluded: {', '.join(sorted(exclude))}")
    mismatches: list[str] = []
    missing: list[str] = []

    for group_name, config_paths in groups:
        results = _process_model_group(group_name, config_paths, write=False)
        for output_path, compiled in results:
            if output_path.stem in exclude:
                continue
            generated_yaml = _dump_yaml(compiled)
            if not output_path.exists():
                missing.append(output_path.name)
            elif generated_yaml != output_path.read_text():
                mismatches.append(output_path.name)

    if missing or mismatches:
        print()
        if missing:
            print(f"{len(missing)} output config(s) missing: {', '.join(missing)}")
        if mismatches:
            print(f"{len(mismatches)} output config(s) out of date: {', '.join(mismatches)}")
        print("\nRun 'python generate_configs.py --all' to regenerate.")
        sys.exit(1)

    print(f"\nAll {total} configs are up to date.")


def _generate_single(input_path: Path, output_path: Path) -> None:
    """Generate a single config, downloading the model on the fly."""
    model_name, trust_remote_code = _read_model_info(input_path)
    print(f"Downloading model metadata for {model_name}")

    with tempfile.TemporaryDirectory() as model_tmpdir:
        model_local_dir = _download_model_metadata(model_name, Path(model_tmpdir), trust_remote_code=trust_remote_code)
        compiled = _compile_one(input_path, model_local_dir)

    _delete_hf_cache(model_name)
    _write_config(compiled, output_path)
    print(f"Config generated: {input_path.name} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Automodel config from Customizer TrainingStepConfig.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Regenerate all configs (input_configs/*/*.json -> output_configs/*.yaml)",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify all output configs match what the current code generates (for CI)",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated config stems to leave out of --check (e.g. ones not yet supported)",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input TrainingStepConfig JSON file (not needed with --all or --check)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output YAML file path")
    args = parser.parse_args()

    if args.all:
        _generate_all()
    elif args.check:
        _check_all({stem for stem in args.exclude.split(",") if stem})
    else:
        if not args.input:
            parser.error("input file is required when not using --all or --check")
        if not args.output:
            parser.error("-o/--output is required when not using --all or --check")
        if not args.input.is_file():
            print(f"Error: {args.input} not found")
            sys.exit(1)

        _generate_single(args.input, args.output)


if __name__ == "__main__":
    main()
