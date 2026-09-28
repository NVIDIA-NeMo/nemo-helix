# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin OpenAPI generation config — read from each plugin's pyproject.toml."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PluginConfig:
    """Configuration for a plugin's OpenAPI spec generation.

    A plugin opts in by declaring a ``[tool.nemo.openapi]`` table in its own
    ``pyproject.toml``; values in that table populate the fields below.
    """

    dir: str  # directory under plugins/, e.g. "nemo-data-designer"
    service_name: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    factory_override: Optional[str] = None  # "module:callable" escape hatch
    data_designer_plugin_allowlist: Optional[List[str]] = None

    @classmethod
    def from_pyproject(cls, pyproject_path: Path) -> Optional["PluginConfig"]:
        """Return None if the plugin hasn't opted in or has no nemo.services entries."""
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        opts = data.get("tool", {}).get("nemo", {}).get("openapi")
        if opts is None:
            return None

        # A virtual workspace member isn't installed, so discover_services() can't find it.
        if data.get("tool", {}).get("uv", {}).get("package") is False:
            print(f"Skipping OpenAPI spec for plugin '{pyproject_path.parent.name}': tool.uv.package = false")
            return None

        services = (data.get("project", {}).get("entry-points", {}) or {}).get("nemo.services", {})
        if not services:
            return None

        plugin_dir = pyproject_path.parent.name
        env_vars = opts.get("env_vars")
        if env_vars is not None and not isinstance(env_vars, dict):
            raise ValueError(f"plugin '{plugin_dir}': [tool.nemo.openapi].env_vars must be a table")
        data_designer_plugin_allowlist = opts.get("data_designer_plugin_allowlist")
        if data_designer_plugin_allowlist is not None and (
            not isinstance(data_designer_plugin_allowlist, list)
            or not all(isinstance(item, str) for item in data_designer_plugin_allowlist)
        ):
            raise ValueError(
                f"plugin '{plugin_dir}': [tool.nemo.openapi].data_designer_plugin_allowlist must be a list of strings"
            )

        return cls(
            dir=plugin_dir,
            service_name=opts.get("service_name"),
            env_vars=env_vars,
            factory_override=opts.get("factory_override"),
            data_designer_plugin_allowlist=data_designer_plugin_allowlist,
        )

    def resolve_service_name(self) -> str:
        if self.service_name is not None:
            return self.service_name
        pyproject_path = Path(f"plugins/{self.dir}/pyproject.toml")
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        services = (data.get("project", {}).get("entry-points", {}) or {}).get("nemo.services", {})
        if len(services) != 1:
            raise ValueError(
                f"plugin '{self.dir}' has {len(services)} nemo.services entries — "
                f"expected exactly 1; set [tool.nemo.openapi].service_name to disambiguate"
            )
        return next(iter(services))

    def output_path(self) -> str:
        return f"plugins/{self.dir}/openapi/openapi.yaml"


def discover_plugins(plugins_root: Path = Path("plugins")) -> List[PluginConfig]:
    """Return plugins with a ``[tool.nemo.openapi]`` table, sorted by dir name."""
    configs: List[PluginConfig] = []
    for pyproject_path in sorted(plugins_root.glob("*/pyproject.toml")):
        config = PluginConfig.from_pyproject(pyproject_path)
        if config is not None:
            configs.append(config)
    return configs
