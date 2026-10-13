# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from nhx.guardrails.entities.values._private import RailsConfig


def _enum_to_primitive(obj):
    """Convert an Enum object to its primitive value, or return the object unchanged."""
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _normalize(data):
    """
    Recursively walk through data and replace any Enum instances with their primitive values.
    Handles dictionaries, lists, and simple values.
    """
    if isinstance(data, dict):
        return {k: _normalize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_normalize(v) for v in data]
    return _enum_to_primitive(data)


class ConfigSource(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def get_config(cls, files_url: Optional[str] = None, config_data: Optional[dict] = None) -> RailsConfig:
        raise NotImplementedError


class FileSystemConfigSource(ConfigSource):
    @classmethod
    def get_config(cls, files_url: Optional[str] = None, config_data: Optional[dict] = None) -> RailsConfig:
        # we should run this in executor
        # would this be like async then?
        # TODO: remove it once we have fsspec
        if files_url is None:
            raise ValueError("files_url is required")

        files_url = files_url.replace("file://", "")
        try:
            config = RailsConfig.from_path(files_url)
        except ValueError as e:
            raise ValueError(f"Invalid config files at {files_url}. Exception: {e}") from e

        # TODO: remove this once it becomes the default in NeMo Guardrails Toolkit
        # For the microservice, we want the passthrough mode to be the default.
        # NOTE: RAZVAN
        if config.passthrough is None:
            config.passthrough = True

        return config

    def get_config_id_py_config_files(self, file_path):
        return [full_path for full_path in Path(file_path).rglob("*.py")]


class YamlConfigSource(ConfigSource):
    @classmethod
    def get_config(cls, files_url: Optional[str] = None, config_data: Optional[dict] = None) -> RailsConfig:
        if config_data is None:
            raise ValueError("config_data is required")

        primitive_config = _normalize(config_data)
        yaml_content = yaml.safe_dump(primitive_config, default_flow_style=False, sort_keys=False)

        try:
            config = RailsConfig.from_content(yaml_content=yaml_content)

        except ValueError as e:
            raise ValueError(f"Invalid config data: {config_data}. Exception: {e}") from e

        # TODO: remove this once it becomes the default in NeMo Guardrails Toolkit
        # For the microservice, we want the passthrough mode to be the default.
        if config.passthrough is None:
            config.passthrough = True

        return config


# NOTE (Razvan)
# to be consistent with get_db for example, I think it also makes sense that registry should not care about
# the source handling logic, e.g., we prefer config_data over files_url
def get_config(files_url: Optional[str] = None, config_data: Optional[dict] = None) -> RailsConfig:
    if config_data:
        return YamlConfigSource.get_config(config_data=config_data)
    elif files_url:
        return FileSystemConfigSource.get_config(files_url=files_url)
    else:
        raise ValueError("At least one of files_url or config_data must be provided")
