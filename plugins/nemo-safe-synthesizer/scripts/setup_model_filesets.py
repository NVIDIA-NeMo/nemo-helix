#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup script to create model weight filesets in the Files API.

This script creates HuggingFace-backed filesets that serve as facades for
the model weights used by Safe Synthesizer. Run this once during deployment
setup to register the model sources.

Usage:
    python setup_model_filesets.py --files-api-url http://localhost:8080

Environment Variables:
    NHX_FILES_URL: Files API base URL (alternative to --files-api-url)
    HF_TOKEN: HuggingFace token for private models (optional)
"""

import argparse
import logging
import os
import sys

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError, NemoHTTPError, NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILESETS = [
    {
        "name": "smollm3-3b",
        "description": "SmolLM3 3B model for synthesis",
        "storage": {
            "type": "huggingface",
            "repo_id": "HuggingFaceTB/SmolLM3-3B",
            "repo_type": "model",
            "revision": "main",
        },
    },
    {
        "name": "gliner-gretel-bi-large",
        "description": "GLiNER model for PII detection",
        "storage": {
            "type": "huggingface",
            "repo_id": "gretelai/gretel-gliner-bi-large-v1.0",
            "repo_type": "model",
            "revision": "main",
        },
    },
    {
        "name": "bge-base-en",
        "description": "BGE base English embeddings (GLiNER dependency)",
        "storage": {
            "type": "huggingface",
            "repo_id": "BAAI/bge-base-en-v1.5",
            "repo_type": "model",
            "revision": "main",
        },
    },
    {
        "name": "sentence-transformer-distiluse",
        "description": "Sentence Transformer for text embeddings in evaluation",
        "storage": {
            "type": "huggingface",
            "repo_id": "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "repo_type": "model",
            "revision": "main",
        },
    },
]

DEFAULT_WORKSPACE = "default"


def create_filesets(
    sdk: NeMoHelix,
    workspace: str,
    dry_run: bool = False,
) -> list[str]:
    """Create model filesets in the Files API.

    Args:
        sdk: NeMo Helix SDK client
        workspace: Workspace to create filesets in
        dry_run: If True, only print what would be created

    Returns:
        List of created fileset names
    """
    files = client_from_platform(sdk, FilesClient)
    created = []
    for fileset_config in MODEL_FILESETS:
        name = fileset_config["name"]
        full_name = f"{workspace}/{name}"

        if dry_run:
            logger.info("Would create fileset: %s", full_name)
            logger.info("  Storage: %s", fileset_config["storage"])
            created.append(name)
            continue

        try:
            files.get_fileset(name=name, workspace=workspace)
            logger.info("Fileset already exists: %s", full_name)
            created.append(name)
            continue
        except NotFoundError:
            logger.debug("Fileset does not exist yet: %s", full_name)

        try:
            storage = HuggingfaceStorageConfig.model_validate(fileset_config["storage"])
            files.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=name,
                    description=fileset_config.get("description", ""),
                    purpose="generic",
                    storage=storage,
                ),
            )
            logger.info("Created fileset: %s", full_name)
            created.append(name)
        except ConflictError:
            logger.info("Fileset already exists: %s", full_name)
            created.append(name)
        except NemoHTTPError as e:
            logger.error("Failed to create fileset %s: %s", full_name, e)
        except ValueError as e:
            logger.error("Invalid fileset config for %s: %s", full_name, e)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Create model weight filesets in the Files API")
    parser.add_argument(
        "--files-api-url",
        default=os.environ.get("NHX_FILES_URL", "http://localhost:8080"),
        help="Files API base URL",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Workspace to create filesets in (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without creating filesets")
    args = parser.parse_args()

    logger.info("Files API URL: %s", args.files_api_url)
    logger.info("Workspace: %s", args.workspace)
    if args.dry_run:
        logger.info("DRY RUN - no filesets will be created")

    sdk = NeMoHelix(base_url=args.files_api_url)
    created = create_filesets(sdk, args.workspace, dry_run=args.dry_run)

    if created:
        logger.info("\nCreated %d filesets:", len(created))
        for name in created:
            logger.info("  - %s/%s", args.workspace, name)
    else:
        logger.warning("No filesets were created")

    return 0 if created else 1


if __name__ == "__main__":
    sys.exit(main())
