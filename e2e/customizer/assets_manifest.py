# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and query ``assets_manifest.json`` for customizer E2E tooling."""

import json
import os
import sys
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).with_name("assets_manifest.json")


def manifest_path() -> Path:
    return _MANIFEST_PATH


def repo_root() -> Path:
    return _MANIFEST_PATH.resolve().parents[2]


def load() -> dict[str, Any]:
    with _MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def s3_bucket(manifest: dict[str, Any] | None = None) -> str:
    return os.environ.get("S3_BUCKET", (manifest or load())["s3"]["bucket"])


def generation_config(manifest: dict[str, Any] | None = None, *, repo_root_dir: Path | None = None) -> dict[str, Any]:
    manifest = manifest or load()
    cfg = manifest["generation"]
    root = repo_root_dir or repo_root()
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return {
        "seed": int(cfg["seed"]),
        "training_size": int(cfg["training_size"]),
        "validation_size": int(cfg["validation_size"]),
        "output_dir": output_dir,
    }


def model_ignore_globs(manifest: dict[str, Any] | None = None) -> str:
    return str((manifest or load())["model_ignore_globs"])


def iter_models(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return list((manifest or load())["models"])


def local_only_formats(manifest: dict[str, Any] | None = None) -> set[str]:
    """Dataset output formats not yet published to S3 (``"s3_published": false``)."""
    formats: set[str] = set()
    for dataset in (manifest or load())["datasets"].values():
        if dataset.get("s3_published", True) is False:
            formats.update(dataset["outputs"])
    return formats


def print_publish_settings(repo_root_dir: Path) -> None:
    """Print bucket, dataset output dir, and model ignore globs for publish_assets_to_s3.sh."""
    manifest = load()
    cfg = generation_config(manifest, repo_root_dir=repo_root_dir)
    print(s3_bucket(manifest), cfg["output_dir"], model_ignore_globs(manifest))


def print_models() -> None:
    """Print s3_folder and hf_repo tab-separated, one model per line."""
    for model in iter_models():
        print(f"{model['s3_folder']}\t{model['hf_repo']}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: assets_manifest.py publish-settings <repo_root> | models")

    command = sys.argv[1]
    if command == "publish-settings":
        if len(sys.argv) != 3:
            raise SystemExit("usage: assets_manifest.py publish-settings <repo_root>")
        print_publish_settings(Path(sys.argv[2]))
        return 0
    if command == "models":
        print_models()
        return 0
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
