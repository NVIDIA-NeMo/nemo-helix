# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the running Studio service."""

from typing import Annotated

from fastapi import Depends
from nhx.common.service.dependencies import service_instance_dependency
from nhx.studio.config import StudioConfig
from nhx.studio.service import StudioService

StudioServiceDep = Annotated[StudioService, Depends(service_instance_dependency("studio", StudioService))]


def get_studio_config(studio: StudioServiceDep) -> StudioConfig:
    return studio.service_config or StudioConfig()


StudioConfigDep = Annotated[StudioConfig, Depends(get_studio_config)]
