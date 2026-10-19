# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from data_designer.config.seed_source import SeedSource
from pydantic import field_validator

FILESET_TOO_MANY_SLASHES = "Path fileset cannot contain more than one slash (/)"
MISSING_FRAGMENT_DELIMITER = "Path should contain a fragment delimiter (#)"
EMPTY_FILESET = "Fileset cannot be empty"
EMPTY_PATH_FRAGMENT = "Path fragment cannot be empty"


class FilesetFileSeedSource(SeedSource):
    seed_type: Literal["nhx"] = "nhx"

    path: str

    @field_validator("path", mode="after")
    def validate_path(cls, value: str) -> str:
        components = value.split("#", 1)
        if len(components) == 1:
            raise ValueError(MISSING_FRAGMENT_DELIMITER)

        fileset = components[0]
        if len(fileset) == 0:
            raise ValueError(EMPTY_FILESET)

        fileset_components = fileset.split("/")
        if len(fileset_components) > 2:
            raise ValueError(FILESET_TOO_MANY_SLASHES)

        path_fragment = components[1]
        if len(path_fragment) == 0:
            raise ValueError(EMPTY_PATH_FRAGMENT)

        return value
