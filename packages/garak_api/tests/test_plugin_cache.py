# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest
from garakapi import PLUGIN_TYPES, PluginCacheNoValidate, _plugins, parse_plugin_spec, plugin_info


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


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    source = tmp_path / "package" / "plugin_cache.json"
    source.parent.mkdir()
    source.write_text('{"probes": {"new": {}}}', encoding="utf-8")
    source.chmod(0o640)
    os.utime(source, (20, 20))
    destination = tmp_path / "cache" / "resources" / "plugin_cache.json"
    monkeypatch.setattr(_plugins.PluginCache, "_plugin_cache_filename", source)
    monkeypatch.setattr(_plugins.PluginCache, "_user_plugin_cache_filename", destination)
    return object.__new__(PluginCacheNoValidate), source, destination


@pytest.mark.parametrize("existing_cache", [False, True])
def test_cache_publication_is_atomic(isolated_cache, monkeypatch, existing_cache):
    cache, source, destination = isolated_cache
    old_cache = {"probes": {"old": {}}}
    if existing_cache:
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(old_cache), encoding="utf-8")
        os.utime(destination, (10, 10))

    copying = Barrier(3)
    finish_copying = Event()
    temporary_paths = []
    copy2 = shutil.copy2

    def paused_copy(source_path, destination_path):
        temporary_paths.append(Path(destination_path))
        # Hold both writers after truncation and a partial write, exposing the
        # original race deterministically without relying on process timing.
        with open(destination_path, "w", encoding="utf-8") as stream:
            stream.write('{"probes":')
            stream.flush()
            copying.wait(timeout=10)
            assert finish_copying.wait(timeout=10)
        return copy2(source_path, destination_path)

    monkeypatch.setattr(_plugins.shutil, "copy2", paused_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writers = [pool.submit(cache._load_plugin_cache) for _ in range(2)]
        try:
            copying.wait(timeout=10)
            assert len(set(temporary_paths)) == 2
            assert all(path.parent == destination.parent for path in temporary_paths)
            if existing_cache:
                # Readers during a refresh still see the complete old cache.
                with destination.open(encoding="utf-8") as stream:
                    assert json.load(stream) == old_cache
            else:
                assert not destination.exists()
        finally:
            finish_copying.set()
        assert [writer.result(timeout=10) for writer in writers] == [{"probes": {"new": {}}}] * 2

    assert json.loads(destination.read_text(encoding="utf-8")) == {"probes": {"new": {}}}
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(source.stat().st_mode)
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("existing_cache", [False, True])
@pytest.mark.parametrize("failure", ["copy", "replace"])
def test_cache_publication_failure_cleans_up(isolated_cache, monkeypatch, existing_cache, failure):
    cache, _, destination = isolated_cache
    old_content = '{"probes": {"old": {}}}'
    if existing_cache:
        destination.parent.mkdir(parents=True)
        destination.write_text(old_content, encoding="utf-8")
        os.utime(destination, (10, 10))

    def fail_copy(source_path, destination_path):
        Path(destination_path).write_text('{"probes":', encoding="utf-8")
        raise OSError("copy failed")

    def fail_replace(source_path, destination_path):
        raise OSError("replace failed")

    if failure == "copy":
        monkeypatch.setattr(_plugins.shutil, "copy2", fail_copy)
    else:
        monkeypatch.setattr(_plugins.os, "replace", fail_replace)

    with pytest.raises(OSError, match=f"{failure} failed"):
        cache._load_plugin_cache()

    if existing_cache:
        assert destination.read_text(encoding="utf-8") == old_content
        assert destination.stat().st_mtime == 10
        assert list(destination.parent.iterdir()) == [destination]
    else:
        assert not destination.exists()
        assert list(destination.parent.iterdir()) == []


@pytest.mark.parametrize("mtime", [20, 30])
def test_fresh_cache_is_not_replaced(isolated_cache, monkeypatch, mtime):
    cache, _, destination = isolated_cache
    destination.parent.mkdir(parents=True)
    destination.write_text('{"probes": {"user": {}}}', encoding="utf-8")
    os.utime(destination, (mtime, mtime))

    def unexpected_copy(*args, **kwargs):
        pytest.fail("An up-to-date cache should not be copied")

    monkeypatch.setattr(_plugins.shutil, "copy2", unexpected_copy)
    assert cache._load_plugin_cache() == {"probes": {"user": {}}}
    assert destination.stat().st_mtime == mtime
