# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config store population utilities."""

import logging
from pathlib import Path

import fsspec
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.guardrails.app.utils.config_utils import (
    _load_and_execute_py_config,
    get_path_to_py_configs,
)
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import RailsConfig

logger = logging.getLogger(__name__)


async def populate_config_store(
    entities_client: EntityClient,
    config_store_path: Path,
):
    """Populate entity store with default configs from mounted config store.

    Args:
        entities_client: The entity client to use for creating configs.
        config_store_path: Path to config store.
    """

    # Ensure config_store_path is an absolute path
    config_store_path = config_store_path.resolve()

    # Use fsspec to list directories in the config_store_path (str for consistent behavior across platforms)
    fs = fsspec.filesystem("file")
    config_store_str = str(config_store_path)

    # Check if the config store path exists
    if not fs.exists(config_store_str):
        logger.warning(f"Config store path does not exist: {config_store_path}. Skipping config store initialization.")
        return

    # NOTE: We assume the system workspace already exists because guardrails.startup()
    # waits for the entities service to be ready before calling this function.
    # See architecture/docs/service-startup.md for the seeding pattern.

    logger.debug("Prefilling database with default guardrail configs in the Config Store")

    for config_dir in fs.ls(config_store_str):
        config_dir_path = Path(config_dir)
        # Resolve relative entries (e.g. bare "abc") against config_store_path so isdir() checks the right place
        if not config_dir_path.is_absolute():
            config_dir_path = (config_store_path / config_dir_path).resolve()
        else:
            config_dir_path = config_dir_path.resolve()

        # Skip entries that start with '..' (configmap artifacts)
        if not fs.isdir(str(config_dir_path)) or config_dir_path.name.startswith(".."):
            continue

        config_name = config_dir_path.name

        # Check if the guardrail_config already exists
        try:
            existing_config = await entities_client.get(GuardrailConfig, name=config_name, workspace=SYSTEM_WORKSPACE)
            if existing_config:
                logger.debug(f"{config_name} config already exists, skipping")
                continue
        except EntityNotFoundError:
            pass  # Config doesn't exist, we'll create it

        url = f"file://{config_dir_path}"
        try:
            rails_config = RailsConfig.from_path(str(config_dir_path))
            if rails_config.passthrough is None:
                rails_config.passthrough = True
            # Deduplicate models loaded from disk. RailsConfig.from_path uses
            # os.walk(followlinks=True), which can read the same YAML file multiple
            # times when the directory contains symlinks (ex. a Kubernetes ConfigMap
            # mount). Keep only the first occurrence of each model type.
            seen_types: set = set()
            deduped_models = []
            for m in rails_config.models:
                if m.type not in seen_types:
                    seen_types.add(m.type)
                    deduped_models.append(m)
            if len(deduped_models) < len(rails_config.models):
                logger.debug(
                    "Duplicate model types detected in config '%s' loaded from %s "
                    "Deduplicated the models to keep only the first occurrence of each type.",
                    config_name,
                    config_dir_path,
                )
                rails_config = rails_config.model_copy(update={"models": deduped_models})
        except Exception as e:
            logger.warning(f"Failed to load config from {config_dir_path}: {e}")
            continue

        guardrail_config = GuardrailConfig(
            name=config_name,
            workspace=SYSTEM_WORKSPACE,
            description=f"{config_name} guardrail config",
            data=rails_config,
        )

        try:
            await entities_client.create(guardrail_config)
            logger.debug(f"{config_name} config-store added")
        except Exception as e:
            logger.warning(f"Failed to add {config_name} config-store: {e}")
            continue

        # Load and execute config.py files if any; do not let one config's failure abort the rest
        try:
            py_config_files = get_path_to_py_configs(url)
            for py_config_file in py_config_files:
                _load_and_execute_py_config(py_config_file)
            logger.debug(f"{config_name} config-store initialized")
        except Exception as e:
            logger.warning(f"Failed to load py config for {config_name}: {e}")
