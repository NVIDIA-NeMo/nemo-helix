# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from garakapi import PLUGIN_TYPES, parse_plugin_spec, plugin_info


def test_PLUGIN_TYPES():
    assert isinstance(PLUGIN_TYPES, tuple)


def test_parse_plugin_spec():
    assert isinstance(parse_plugin_spec("grandma,encoding", "probes"), tuple)


def test_parse_plugin_spec_tag_filter():
    assert isinstance(parse_plugin_spec("grandma,encoding", "probes", probe_tag_filter="owasp:llm06"), tuple)

    unfiltered = parse_plugin_spec("all", "probes")
    filtered = parse_plugin_spec("all", "probes", probe_tag_filter="owasp:llm06")

    # Try not to assume too much specifically about probes and their tags apart
    # from some have the above tag and some do not.
    assert len(unfiltered[0]) > len(filtered[0])


def test_parse_plugin_spec_non_probe_default_tag_filter():
    detectors = parse_plugin_spec("always.Fail,always.Pass", "detectors")
    assert isinstance(detectors, tuple)
    assert len(detectors[0]) == 2


def test_plugin_info():
    assert isinstance(plugin_info("probes.grandma.Slurs"), dict)


def test_load_plugin_cache_reads_packaged_cache_without_user_copy(tmp_path, monkeypatch):
    import json

    from garakapi import PluginCacheNoValidate, _plugins

    user_cache = tmp_path / "resources" / "plugin_cache.json"
    monkeypatch.setattr(_plugins.PluginCache, "_user_plugin_cache_filename", user_cache)

    loaded = PluginCacheNoValidate()._load_plugin_cache()

    with open(_plugins.PluginCache._plugin_cache_filename, encoding="utf-8") as f:
        assert loaded == json.load(f)
    # Concurrent importers raced on this copy; nothing may write it any more.
    assert not user_cache.exists()
