# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import sys
from typing import Callable, List, Union

with open(f"{pathlib.Path(__file__).parent}/resources/VERSION") as fd:
    __version__ = fd.readline().strip()

this = sys.modules[__name__]

sys.modules["garak"] = this

from garak import _config, _plugins  # noqa: E402


class PluginCacheNoValidate(_plugins.PluginCache):
    """Override PluginCache to avoid validation against full garak installation"""

    def _valid_loaded_cache(self, *args, **kwargs):
        return True

    def _load_plugin_cache(self):
        # Read the packaged cache directly. Upstream copies it into the per-user
        # cache dir with a non-atomic shutil.copy2 and then reads that copy, so
        # concurrent first imports (e.g. parallel OpenAPI generation on a fresh
        # CI runner) can read a truncated file. Without validation the copy is
        # never rebuilt, so it adds nothing but the race.
        with open(self._plugin_cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)


# Prepopulate the singleton cache so that PluginCache doesn't call its
# own _valid_loaded_cache
_plugins.PluginCache._plugin_cache_dict = PluginCacheNoValidate()._load_plugin_cache()

PLUGIN_TYPES = _plugins.PLUGIN_TYPES


def parse_plugin_spec(spec: str, category: str, probe_tag_filter: str = "") -> tuple[List[str], List[str]]:
    plugins = _config.parse_plugin_spec(spec, category)
    plugin_names = plugins[0]
    unknown_plugins = plugins[1]

    if probe_tag_filter is not None and len(probe_tag_filter) > 1:
        plugins_to_skip = []
        for plugin_name in plugin_names:
            info = _plugins.plugin_info(plugin_name)
            if not any([tag.startswith(probe_tag_filter) for tag in info.get("tags", [])]):
                plugins_to_skip.append(plugin_name)  # using list.remove doesn't update for-loop position

        for plugin_to_skip in plugins_to_skip:
            plugin_names.remove(plugin_to_skip)

    return (plugin_names, unknown_plugins)


def plugin_info(plugin: Union[Callable, str]) -> dict:
    return _plugins.plugin_info(plugin)


# Necessary to cover `from garak.* import *` cases
sys.modules["garak._plugins"] = _plugins
sys.modules["garak._config"] = _config
