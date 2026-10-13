# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API dependencies for the Guardrails service."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.guardrails.app.services.configs.registry import ConfigRegistry
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.services.rails.service import RailsService

logger = logging.getLogger(__name__)


def get_config_registry(
    entities_client: EntityClient = Depends(get_entity_client),
) -> ConfigRegistry:
    """Get the ConfigRegistry instance."""
    return ConfigRegistry(entities_client=entities_client)


ConfigRegistryDep = Annotated[ConfigRegistry, Depends(get_config_registry)]


# Dependency for RailsRegistry
@lru_cache()
def get_rails_registry() -> RailsRegistry:
    return RailsRegistry()


RailsRegistryDep = Annotated[RailsRegistry, Depends(get_rails_registry)]


# Dependency for RailsService
def get_rails_service(
    config_registry: ConfigRegistryDep,
    rails_registry: RailsRegistryDep,
) -> RailsService:
    return RailsService(config_registry=config_registry, rails_registry=rails_registry)


RailsServiceDep = Annotated[RailsService, Depends(get_rails_service)]
