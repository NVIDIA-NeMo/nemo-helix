# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform seed task: orchestrates built-in and plugin-contributed seeding."""

import asyncio
import logging
import sys
from dataclasses import dataclass, field

from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.discovery import discover_seed_jobs
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import CreateModelProviderRequest
from nhx.common.config import get_platform_config
from nhx.common.entities import EntityClient
from nhx.common.sdk_factory import get_async_platform_sdk
from nhx.common.service.api.health import async_wait_for_dependencies
from nhx.platform_seed.config import PlatformSeedConfig

logger = logging.getLogger(__name__)

PLATFORM_SEED_DEPENDENCIES = ["entities", "auth", "files", "models"]


@dataclass
class PlatformSeedResult:
    """Result of a platform seed run."""

    auth_ok: bool = False
    guardrails_ok: bool = False
    models_ok: bool = False
    plugin_results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


async def seed_guardrails(entity_client: EntityClient, config: PlatformSeedConfig) -> None:
    """Seed default guardrail configs and file-based configs from config store. Idempotent."""
    from nhx.guardrails.app.seeding import seed_default_configs
    from nhx.guardrails.app.utils.config_store import populate_config_store

    # Seed code-defined default guardrail configs
    await seed_default_configs(entity_client)
    # Seed file-based configs from the Config Store path
    await populate_config_store(entity_client, config.guardrails_config_store_path)

    logger.info("Guardrails config store populated")


async def seed_auth(entity_client: EntityClient, config: PlatformSeedConfig) -> None:
    """Seed auth role bindings (platform admin, wildcard default/system workspace). Idempotent."""
    from nhx.core.auth.app.seeding import run_seeding

    if not await run_seeding(entity_client):
        raise RuntimeError("Auth role binding seeding failed")
    logger.info("Auth role bindings seeded")


async def seed_model_provider(sdk: AsyncNeMoHelix) -> None:
    """Seed the default nvidia-build model provider. Idempotent."""
    models = client_from_platform(sdk, AsyncModelsClient)
    try:
        await models.create_provider(
            workspace="system",
            body=CreateModelProviderRequest(
                name="nvidia-build",
                host_url="https://integrate.api.nvidia.com",
                api_key_secret_name="ngc-api-key",
            ),
        )
        logger.info("nvidia-build model provider created")
    except ConflictError:
        logger.debug("nvidia-build model provider already exists, skipping")


async def run_plugin_seed_jobs(
    sdk: AsyncNeMoHelix,
    entity_client: EntityClient,
    config: PlatformSeedConfig,
    result: PlatformSeedResult,
) -> None:
    """Discover and run plugin seed jobs registered under ``nemo.seed``."""
    seed_jobs = discover_seed_jobs()
    if not seed_jobs:
        logger.debug("No plugin seed jobs discovered")
        return

    logger.info("Discovered %d plugin seed job(s): %s", len(seed_jobs), ", ".join(sorted(seed_jobs)))
    for name, seed_cls in seed_jobs.items():
        if not config.is_plugin_seed_enabled(name):
            logger.info(
                "Plugin seed job %r disabled via %s=false",
                name,
                config.plugin_seed_enabled_env_var(name),
            )
            continue
        try:
            job = seed_cls()
            job.sdk = sdk
            job.entities_client = entity_client
            await job.run()
            result.plugin_results[name] = True
            logger.info("Plugin seed job %r completed successfully", name)
        except Exception as exc:
            result.plugin_results[name] = False
            message = f"Plugin seed job {name!r} failed: {exc}"
            logger.exception(message)
            result.errors.append(message)


async def run_platform_seed(
    entity_client: EntityClient,
    sdk: AsyncNeMoHelix,
    config: PlatformSeedConfig,
) -> PlatformSeedResult:
    """
    Run all enabled platform seed operations.

    Idempotent: safe to run multiple times. Each seed step may be enabled or
    disabled via config. Failures in one step are logged and recorded in the
    result; other steps still run.

    Args:
        entity_client: Entity client for creating/updating entities.
        sdk: Async NeMo Helix SDK (for files API and internal calls).
        config: Seed configuration (enabled flags, paths, etc.).

    Returns:
        PlatformSeedResult with per-step success and any errors.
    """
    result = PlatformSeedResult()

    if not config.enabled:
        logger.debug("Platform seed disabled via config")
        return result

    if config.auth_enabled:
        try:
            await seed_auth(entity_client, config)
            result.auth_ok = True
        except Exception as e:
            msg = f"Auth seed failed: {e}"
            logger.exception(msg)
            result.errors.append(msg)

    if config.guardrails_enabled:
        try:
            await seed_guardrails(entity_client, config)
            result.guardrails_ok = True
        except Exception as e:
            msg = f"Guardrails seed failed: {e}"
            logger.exception(msg)
            result.errors.append(msg)

    if config.model_provider_enabled:
        try:
            await seed_model_provider(sdk)
            result.models_ok = True
        except Exception as e:
            msg = f"Models seed failed: {e}"
            logger.exception(msg)
            result.errors.append(msg)

    await run_plugin_seed_jobs(sdk, entity_client, config, result)

    return result


async def run_platform_seed_from_startup() -> bool:
    """
    Run platform seed from platform API lifespan (e.g. when seed_on_startup is enabled).

    Uses existing logging and config. Waits for dependencies if configured, then runs
    the same seed logic as the standalone task. Does not call sys.exit.

    Returns:
        True if seed completed with no errors (or was disabled), False otherwise.
    """
    config = PlatformSeedConfig()
    if not config.enabled:
        logger.info("Platform seed is disabled (NHX_PLATFORM_SEED_ENABLED=false)")
        return True

    try:
        platform_config = get_platform_config()
    except Exception as e:
        logger.error("Failed to load platform config: %s", e)
        return False

    if config.wait_for_ready_enabled:
        timeout_per_service = config.wait_for_ready_retries * config.wait_for_ready_interval_seconds
        logger.info("Waiting for dependencies: %s", PLATFORM_SEED_DEPENDENCIES)
        if not await async_wait_for_dependencies(
            platform_config,
            PLATFORM_SEED_DEPENDENCIES,
            timeout_per_service=timeout_per_service,
            poll_interval=config.wait_for_ready_interval_seconds,
        ):
            logger.error("One or more dependencies did not become ready in time")
            return False

    sdk = get_async_platform_sdk(as_service="platform-seed", internal=True)
    entity_client = EntityClient(client_from_platform(sdk, AsyncEntitiesClient))

    try:
        result = await run_platform_seed(entity_client, sdk, config)
    finally:
        await sdk.close()

    if result.errors:
        for err in result.errors:
            logger.error("%s", err)
        return False

    logger.info("Platform seed completed successfully")
    return True


def run() -> int:
    """Execute the platform seed task (CLI/Job entry point).

    Expects to be run via the platform entrypoint (nemo-helix run task --task nhx.platform_seed)
    so that logging and observability are already configured.
    """
    ok = asyncio.run(run_platform_seed_from_startup())
    return 0 if ok else 1


def main() -> None:
    """Synchronous entry point for scripts and Jobs."""
    sys.exit(run())
