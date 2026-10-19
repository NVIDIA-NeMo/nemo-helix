# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OPA bundle generation."""

import gzip
import io
import json
import tarfile

import pytest


@pytest.mark.asyncio
async def test_authorization_data_merges_plugin_authz_contributions(monkeypatch):
    """Plugin authz contributions are included before validation and bundle build.

    The bundle derives contributions via ``discover_plugin_authz`` (routes-derived model),
    so the stub returns a clean ``PluginAuthzResult`` rather than a raw contribution dict.
    """
    from nemo_helix_plugin.authz import AuthzContribution, AuthzEndpointMethod
    from nemo_helix_plugin.authz_discovery import PluginAuthzResult
    from nhx.core.auth.app.bundle import _build_authorization_data_internal

    plugin_path = "/apis/example-plugin/v2/workspaces/{workspace}/jobs"
    result = PluginAuthzResult(
        key="example-plugin",
        contribution=AuthzContribution(
            permissions={"example-plugin.jobs.read": "Read example plugin jobs"},
            endpoints={
                plugin_path: {
                    "get": AuthzEndpointMethod(
                        permissions=["example-plugin.jobs.read"],
                        scopes=["example-plugin:read", "platform:read"],
                    )
                }
            },
        ),
        problems=[],
        warnings=[],
        mount_name="example-plugin",
    )

    monkeypatch.setattr(
        "nemo_helix_plugin.authz_discovery.discover_plugin_authz",
        lambda: [result],
    )

    data = await _build_authorization_data_internal(entities_client=None)

    assert data["authz"]["endpoints"][plugin_path]["get"]["permissions"] == ["example-plugin.jobs.read"]
    assert "example-plugin.jobs.read" in data["authz"]["roles"]["Viewer"]["permissions"]


@pytest.mark.asyncio
async def test_bundle_generation():
    """Test that bundle can be generated without a database."""
    from nhx.core.auth.app.bundle import clear_bundle_cache, get_opa_bundle_with_etag

    # Clear any cached bundle
    clear_bundle_cache()

    # Generate bundle without database
    bundle_bytes, etag = await get_opa_bundle_with_etag(entities_client=None)

    # Verify bundle is valid
    assert bundle_bytes is not None
    assert len(bundle_bytes) > 0
    assert etag is not None
    assert len(etag) == 32  # MD5 hash is 32 hex chars

    # Verify bundle is valid gzip
    bundle_io = io.BytesIO(bundle_bytes)
    with gzip.GzipFile(fileobj=bundle_io, mode="rb") as gz:
        tar_bytes = gz.read()

    # Verify bundle is valid tarfile
    tar_io = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=tar_io, mode="r") as tar:
        members = tar.getnames()

        # Should contain data.json and manifest
        assert "data.json" in members
        assert ".manifest" in members

        # Should contain at least one .rego policy file
        rego_files = [m for m in members if m.endswith(".rego")]
        assert len(rego_files) > 0

        # Verify data.json structure
        data_file = tar.extractfile("data.json")
        assert data_file is not None
        data = json.load(data_file)

        # Should have authz key
        assert "authz" in data
        assert "roles" in data["authz"]
        assert "workspaces" in data["authz"]


@pytest.mark.asyncio
async def test_bundle_caching():
    """Test that bundle is cached correctly."""
    from nhx.core.auth.app.bundle import clear_bundle_cache, get_opa_bundle_with_etag

    # Clear cache
    clear_bundle_cache()

    # First call should generate bundle
    bundle1, etag1 = await get_opa_bundle_with_etag(entities_client=None)

    # Second call should return cached bundle
    bundle2, etag2 = await get_opa_bundle_with_etag(entities_client=None)

    # Should be the same
    assert bundle1 == bundle2
    assert etag1 == etag2


@pytest.mark.asyncio
async def test_bundle_etag_stability():
    """Test that bundle E-Tag is stable for same data."""
    from nhx.core.auth.app.bundle import clear_bundle_cache, get_opa_bundle_with_etag

    # Clear cache
    clear_bundle_cache()

    # Generate bundle
    _, etag1 = await get_opa_bundle_with_etag(entities_client=None)

    # Clear cache and regenerate
    clear_bundle_cache()
    _, etag2 = await get_opa_bundle_with_etag(entities_client=None)

    # E-Tag should be the same for same data
    assert etag1 == etag2


# --- Plugin authz fail-mode (authz.on_invalid_plugin) ---


def _problem_result():
    """A plugin result with one valid route and one unruled (deny) route + a problem."""
    from nemo_helix_plugin.authz import AuthzContribution, AuthzEndpointMethod
    from nemo_helix_plugin.authz_discovery import PluginAuthzResult

    contribution = AuthzContribution(
        permissions={"p.read": "Read"},
        endpoints={
            "/apis/p/v2/ok": {"get": AuthzEndpointMethod(permissions=["p.read"])},
            "/apis/p/v2/bad": {"get": AuthzEndpointMethod(permissions=[], deny=True)},
        },
    )
    return PluginAuthzResult(key="p", contribution=contribution, problems=["/apis/p/v2/bad (GET) has no @path_rule"])


def _patch_failmode(monkeypatch, results, on_invalid):
    from types import SimpleNamespace

    import nhx.core.auth.app.bundle as bundle

    monkeypatch.setattr("nemo_helix_plugin.authz_discovery.discover_plugin_authz", lambda: results)
    monkeypatch.setattr(
        bundle,
        "get_service_config",
        lambda _cls: SimpleNamespace(on_invalid_plugin=on_invalid),
    )
    return bundle


def test_on_invalid_plugin_deny_route_keeps_valid_routes(monkeypatch):
    bundle = _patch_failmode(monkeypatch, [_problem_result()], "deny_route")
    merged = bundle.merge_plugin_authz_contributions({"authz": {}})
    endpoints = merged["authz"]["endpoints"]
    assert "deny" not in endpoints["/apis/p/v2/ok"]["get"]  # valid route preserved
    assert endpoints["/apis/p/v2/bad"]["get"]["deny"] is True  # only the bad route denied
    assert "p" in bundle.get_degraded_plugins()


def test_on_invalid_plugin_quarantine_denies_whole_plugin(monkeypatch):
    bundle = _patch_failmode(monkeypatch, [_problem_result()], "quarantine")
    merged = bundle.merge_plugin_authz_contributions({"authz": {}})
    endpoints = merged["authz"]["endpoints"]
    # The previously-valid route is now denied too — the whole plugin is quarantined.
    assert endpoints["/apis/p/v2/ok"]["get"]["deny"] is True
    assert endpoints["/apis/p/v2/bad"]["get"]["deny"] is True
    # quarantine also fences the whole namespace, so a route the runner mounts that
    # derivation never saw (quarantine only rewrites the routes it did see) can't fall through.
    assert merged["authz"]["config"]["denied_plugin_prefixes"] == ["/apis/p"]


def test_on_invalid_plugin_hard_fail_raises(monkeypatch):
    bundle = _patch_failmode(monkeypatch, [_problem_result()], "hard_fail")
    with pytest.raises(RuntimeError, match="hard_fail"):
        bundle.merge_plugin_authz_contributions({"authz": {}})


def test_clean_plugin_merges_without_degraded(monkeypatch):
    from nemo_helix_plugin.authz import AuthzContribution, AuthzEndpointMethod
    from nemo_helix_plugin.authz_discovery import PluginAuthzResult

    clean = PluginAuthzResult(
        key="c",
        contribution=AuthzContribution(
            permissions={"c.read": "Read"},
            endpoints={"/apis/c/v2/x": {"get": AuthzEndpointMethod(permissions=["c.read"])}},
        ),
        problems=[],
    )
    bundle = _patch_failmode(monkeypatch, [clean], "deny_route")
    merged = bundle.merge_plugin_authz_contributions({"authz": {}})
    assert "/apis/c/v2/x" in merged["authz"]["endpoints"]
    assert bundle.get_degraded_plugins() == {}


def test_degraded_plugin_with_no_routes_is_namespace_fenced(monkeypatch):
    """A plugin that couldn't be enumerated (empty contribution) fences its whole namespace,
    so any route it still mounts can't fall through the service: no-match bypass."""
    from nemo_helix_plugin.authz import AuthzContribution
    from nemo_helix_plugin.authz_discovery import PluginAuthzResult

    degraded = PluginAuthzResult(
        key="bad",
        contribution=AuthzContribution(),  # no endpoints — could not enumerate
        problems=["failed to load plugin: RuntimeError('boom')"],
    )
    bundle = _patch_failmode(monkeypatch, [degraded], "deny_route")
    merged = bundle.merge_plugin_authz_contributions({"authz": {}})
    assert merged["authz"]["config"]["denied_plugin_prefixes"] == ["/apis/bad"]
    assert "bad" in bundle.get_degraded_plugins()


def test_degraded_plugin_fences_both_key_and_mount_name(monkeypatch):
    """When a degraded plugin's declared mount name diverges from its entry-point key (the
    name==key invariant is only warned, not enforced), the fence must cover both /apis/<key>
    and /apis/<name> — the runner mounts the plugin's real routes at /apis/<name>."""
    from nemo_helix_plugin.authz import AuthzContribution
    from nemo_helix_plugin.authz_discovery import PluginAuthzResult

    degraded = PluginAuthzResult(
        key="bad",
        contribution=AuthzContribution(),  # no endpoints — could not enumerate
        problems=["failed to load plugin: RuntimeError('boom')"],
        mount_name="bad-actual",
    )
    bundle = _patch_failmode(monkeypatch, [degraded], "deny_route")
    merged = bundle.merge_plugin_authz_contributions({"authz": {}})
    assert merged["authz"]["config"]["denied_plugin_prefixes"] == ["/apis/bad", "/apis/bad-actual"]
