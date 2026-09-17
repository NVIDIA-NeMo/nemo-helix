# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_platform_plugin.discovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from fastapi import APIRouter
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.discovery import (
    _ALL_SURFACE_GROUPS,
    AGENT_CLI_GROUP,
    CUSTOMIZATION_CONTRIBUTORS_GROUP,
    discover,
    discover_agent_cli,
    discover_cli,
    discover_customization_contributors,
    discover_entry_points,
    discover_functions,
    discover_jobs,
    discover_manifests,
    discover_sdk,
    discover_seed_jobs,
    discover_services,
)
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.sdk import NemoPluginSDKResources
from nemo_platform_plugin.seed import NemoSeedJob
from nemo_platform_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Cache management — must clear between tests since discover() uses lru_cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_discovery_cache():
    discover_entry_points.cache_clear()
    discover.cache_clear()
    discover_manifests.cache_clear()
    discover_customization_contributors.cache_clear()
    yield
    discover_entry_points.cache_clear()
    discover.cache_clear()
    discover_manifests.cache_clear()
    discover_customization_contributors.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalPluginService(NemoService):
    name = "test"
    dependencies = []

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(APIRouter(), tag="Test")]


def _make_service_cls(name: str = "test") -> type[_MinimalPluginService]:
    return type(f"_Plugin_{name}", (_MinimalPluginService,), {"name": name, "dependencies": []})


class _MinimalPluginCLI(NemoCLI):
    name = "test"

    def get_cli(self) -> typer.Typer:
        return typer.Typer(name="test")


def _make_cli_cls() -> type[_MinimalPluginCLI]:
    return _MinimalPluginCLI


def _make_sdk_resource(_owner: object) -> object:
    return object()


class _MinimalPluginJob(NemoJob):
    name = "test-job"
    description = "A test job"

    def run(self, config: dict) -> dict:
        return {"result": config.get("value", "default")}


def _make_job_cls(name: str = "test-job") -> type[_MinimalPluginJob]:
    return type(f"_Job_{name}", (_MinimalPluginJob,), {"name": name})


class _MinimalSeedJob(NemoSeedJob):
    name = "test-seed"

    async def run(self) -> None:
        return None


def _make_seed_cls(name: str = "test-seed") -> type[_MinimalSeedJob]:
    return type(f"_Seed_{name}", (_MinimalSeedJob,), {"name": name})


class _MinimalSpec(BaseModel):
    value: str = "default"


class _MinimalPluginFunction(NemoFunction[_MinimalSpec]):
    name = "test-fn"
    spec_schema = _MinimalSpec

    async def run(self, spec: _MinimalSpec) -> dict:
        return {"value": spec.value}


def _make_function_cls(name: str = "test-fn") -> type[_MinimalPluginFunction]:
    return type(f"_Fn_{name}", (_MinimalPluginFunction,), {"name": name})


def _make_ep(name: str, value: object, *, version: str = "1.0.0", description: str = "") -> MagicMock:
    dist = MagicMock()
    dist.metadata.get = lambda key, default="": {"Version": version, "Summary": description}.get(key, default)
    ep = MagicMock()
    ep.name = name
    ep.value = f"nmp.{name}:obj"
    ep.load.return_value = value
    ep.dist = dist
    return ep


def _eps_by_group(mapping: dict[str, list[MagicMock]]):
    def _side_effect(group: str) -> list[MagicMock]:
        return mapping.get(group, [])

    return _side_effect


# ---------------------------------------------------------------------------
# discover_entry_points — metadata only
# ---------------------------------------------------------------------------


class TestDiscoverEntryPoints:
    def test_returns_empty_when_no_entry_points(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]):
            result = discover_entry_points("nemo.anything")
        assert result == {}

    def test_returns_entry_points_without_loading(self) -> None:
        ep = _make_ep("alpha", object())
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_entry_points("nemo.cli")
        assert result == {"alpha": ep}
        ep.load.assert_not_called()

    def test_passes_group_to_entry_points(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_entry_points("nemo.jobs")
        mock_eps.assert_called_once_with(group="nemo.jobs")

    def test_surface_allowlist_filters_entry_points(self, monkeypatch) -> None:
        alpha = _make_ep("alpha", object())
        beta = _make_ep("beta", object())
        monkeypatch.setenv("NEMO_PLUGIN_CLI_ALLOWLIST", "alpha")

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[alpha, beta]):
            result = discover_entry_points("nemo.cli")

        assert result == {"alpha": alpha}

    def test_empty_surface_allowlist_disables_entry_points(self, monkeypatch) -> None:
        alpha = _make_ep("alpha", object())
        monkeypatch.setenv("NEMO_PLUGIN_SERVICES_ALLOWLIST", "")

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[alpha]):
            result = discover_entry_points("nemo.services")

        assert result == {}

    def test_wildcard_surface_allowlist_allows_all_entry_points(self, monkeypatch) -> None:
        alpha = _make_ep("alpha", object())
        monkeypatch.setenv("NEMO_PLUGIN_CLI_ALLOWLIST", "*")

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[alpha]):
            result = discover_entry_points("nemo.cli")

        assert result == {"alpha": alpha}

    def test_surface_allowlist_overrides_global_allowlist(self, monkeypatch) -> None:
        alpha = _make_ep("alpha", object())
        beta = _make_ep("beta", object())
        monkeypatch.setenv("NEMO_PLUGIN_ALLOWLIST", "alpha")
        monkeypatch.setenv("NEMO_PLUGIN_CLI_ALLOWLIST", "beta")

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[alpha, beta]):
            result = discover_entry_points("nemo.cli")

        assert result == {"beta": beta}

    def test_allowlist_filters_dot_scoped_entry_points_by_plugin_name(self, monkeypatch) -> None:
        alpha_job = _make_ep("alpha.job", object())
        beta_job = _make_ep("beta.job", object())
        monkeypatch.setenv("NEMO_PLUGIN_JOBS_ALLOWLIST", "alpha")

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[alpha_job, beta_job]):
            result = discover_entry_points("nemo.jobs")

        assert result == {"alpha.job": alpha_job}

    def test_orders_entry_points_by_name_regardless_of_discovery_order(self) -> None:
        # importlib.metadata.entry_points() order reflects distribution discovery
        # order on sys.path, which differs across environments (e.g. macOS vs
        # Linux). Callers like typer.core.TyperGroup preserve registration order
        # rather than sorting, so discover_entry_points() must sort itself to
        # keep CLI/doc generation reproducible across environments.
        unsloth = _make_ep("unsloth.jobs", object())
        automodel = _make_ep("automodel.jobs", object())
        rl = _make_ep("rl.jobs", object())

        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[unsloth, automodel, rl]):
            result = discover_entry_points("nemo.jobs")

        assert list(result.keys()) == ["automodel.jobs", "rl.jobs", "unsloth.jobs"]


# ---------------------------------------------------------------------------
# discover — generic
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_returns_empty_when_no_entry_points(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]):
            result = discover("nemo.anything")
        assert result == {}

    def test_loads_values(self) -> None:
        ep = _make_ep("alpha", {"key": "value"})
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover("nemo.custom")
        assert result == {"alpha": {"key": "value"}}

    def test_passes_group_to_entry_points(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover("nemo.jobs")
        mock_eps.assert_called_once_with(group="nemo.jobs")

    def test_failing_entry_point_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("boom")
        good = _make_ep("good", "ok")
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover("nemo.x")
        assert result == {"good": "ok"}

    def test_failure_does_not_affect_subsequent_entries(self) -> None:
        eps = [_make_ep("first", 1), _make_ep("broken", None), _make_ep("last", 3)]
        eps[1].load.side_effect = ImportError("missing")
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=eps):
            result = discover("nemo.x")
        assert "broken" not in result
        assert result["first"] == 1
        assert result["last"] == 3

    def test_loads_multiple_values(self) -> None:
        eps = [_make_ep("alpha", 1), _make_ep("beta", 2)]
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=eps):
            result = discover("nemo.x")
        assert result == {"alpha": 1, "beta": 2}

    def test_failed_entry_point_logs_warning_without_traceback(self, caplog) -> None:
        bad = _make_ep("missing-plugin", None)
        bad.load.side_effect = ModuleNotFoundError("No module named 'missing_plugin'")
        with caplog.at_level("WARNING", logger="nemo_platform_plugin.discovery"):
            with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad]):
                discover("nemo.skills")
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 1
        assert "missing-plugin" in warning_records[0].message
        assert warning_records[0].exc_info is None or warning_records[0].exc_info[0] is None

    def test_failed_entry_point_logs_traceback_at_debug(self, caplog) -> None:
        bad = _make_ep("missing-plugin", None)
        bad.load.side_effect = ModuleNotFoundError("No module named 'missing_plugin'")
        with caplog.at_level("DEBUG", logger="nemo_platform_plugin.discovery"):
            with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad]):
                discover("nemo.skills")
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG" and "missing-plugin" in r.message]
        assert len(debug_records) >= 1
        tb_record = debug_records[-1]
        assert tb_record.exc_info is not None
        assert tb_record.exc_info[0] is ModuleNotFoundError


# ---------------------------------------------------------------------------
# discover_services
# ---------------------------------------------------------------------------


class TestDiscoverServices:
    def test_uses_nemo_services_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_services()
        mock_eps.assert_called_once_with(group="nemo.services")

    def test_loads_service_class(self) -> None:
        cls = _make_service_cls("alpha")
        ep = _make_ep("alpha", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_services()
        assert result["alpha"] is cls

    def test_returned_class_is_instantiable(self) -> None:
        cls = _make_service_cls("alpha")
        ep = _make_ep("alpha", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_services()
        instance = result["alpha"]()
        assert isinstance(instance, NemoService)

    def test_loads_multiple_services(self) -> None:
        classes = {n: _make_service_cls(n) for n in ("alpha", "beta")}
        eps = [_make_ep(n, c) for n, c in classes.items()]
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=eps):
            result = discover_services()
        assert set(result.keys()) == {"alpha", "beta"}

    def test_failing_service_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")
        good = _make_ep("good", _make_service_cls("good"))
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_services()
        assert "bad" not in result
        assert "good" in result


# ---------------------------------------------------------------------------
# discover_cli
# ---------------------------------------------------------------------------


class TestDiscoverCLI:
    def test_uses_nemo_cli_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_cli()
        mock_eps.assert_called_once_with(group="nemo.cli")

    def test_loads_cli_class(self) -> None:
        cls = _make_cli_cls()
        ep = _make_ep("alpha", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_cli()
        assert result["alpha"] is cls

    def test_returned_class_is_instantiable_and_returns_typer(self) -> None:
        cls = _make_cli_cls()
        ep = _make_ep("alpha", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_cli()
        assert isinstance(result["alpha"]().get_cli(), typer.Typer)

    def test_failing_cli_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")
        good = _make_ep("good", _make_cli_cls())
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_cli()
        assert "bad" not in result
        assert "good" in result


# ---------------------------------------------------------------------------
# discover_agent_cli
# ---------------------------------------------------------------------------


class TestDiscoverAgentCLI:
    def test_uses_agent_cli_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_agent_cli()
        mock_eps.assert_called_once_with(group=AGENT_CLI_GROUP)

    def test_loads_cli_class_under_agent_name(self) -> None:
        class _AnalystCLI(_MinimalPluginCLI):
            name = "analyst"

        ep = _make_ep("analyst", _AnalystCLI)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_agent_cli()

        assert result["analyst"] is _AnalystCLI
        assert isinstance(result["analyst"]().get_cli(), typer.Typer)


# ---------------------------------------------------------------------------
# discover_jobs
# ---------------------------------------------------------------------------


class TestDiscoverJobs:
    def test_uses_nemo_jobs_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_jobs()
        mock_eps.assert_called_once_with(group="nemo.jobs")

    def test_loads_job_class(self) -> None:
        cls = _make_job_cls("example.say-hello")
        ep = _make_ep("example.say-hello", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_jobs()
        assert result["example.say-hello"] is cls

    def test_returned_class_is_instantiable_and_runnable(self) -> None:
        cls = _make_job_cls("example.say-hello")
        ep = _make_ep("example.say-hello", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_jobs()
        output = result["example.say-hello"]().run({"value": "hi"})
        assert output == {"result": "hi"}

    def test_failing_job_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")
        good = _make_ep("good", _make_job_cls("good"))
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_jobs()
        assert "bad" not in result
        assert "good" in result


# ---------------------------------------------------------------------------
# discover_functions
# ---------------------------------------------------------------------------


class TestDiscoverFunctions:
    def test_uses_nemo_functions_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_functions()
        mock_eps.assert_called_once_with(group="nemo.functions")

    def test_loads_function_class(self) -> None:
        cls = _make_function_cls("greet")
        ep = _make_ep("example.greet", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_functions()
        assert result["example.greet"] is cls

    @pytest.mark.asyncio
    async def test_returned_class_is_instantiable_and_runnable(self) -> None:
        cls = _make_function_cls("greet")
        ep = _make_ep("example.greet", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_functions()
        instance = result["example.greet"]()
        output = await instance.run(_MinimalSpec(value="hi"))
        assert output == {"value": "hi"}

    def test_failing_function_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")
        good = _make_ep("example.good", _make_function_cls("good"))
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_functions()
        assert "bad" not in result
        assert "example.good" in result

    def test_logs_warning_on_name_mismatch(self, caplog) -> None:
        # Class declares ``name = "wrong"`` but entry-point key suffix
        # is ``"greet"`` — mismatch must be visible at startup.
        cls = type("_FnMismatch", (_MinimalPluginFunction,), {"name": "wrong"})
        ep = _make_ep("example.greet", cls)
        with caplog.at_level("WARNING", logger="nemo_platform_plugin.discovery"):
            with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
                result = discover_functions()
        assert result["example.greet"] is cls
        assert any("must match the function-name part" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# discover_seed_jobs
# ---------------------------------------------------------------------------


class TestDiscoverSeedJobs:
    def test_uses_nemo_seed_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_seed_jobs()
        mock_eps.assert_called_once_with(group="nemo.seed")

    def test_loads_seed_job_class(self) -> None:
        cls = _make_seed_cls("example")
        ep = _make_ep("example", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_seed_jobs()
        assert result["example"] is cls

    def test_returned_class_is_instantiable(self) -> None:
        cls = _make_seed_cls("example")
        ep = _make_ep("example", cls)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_seed_jobs()
        assert isinstance(result["example"](), NemoSeedJob)

    def test_failing_seed_job_is_skipped(self) -> None:
        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")
        good = _make_ep("good", _make_seed_cls("good"))
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_seed_jobs()
        assert "bad" not in result
        assert "good" in result


class TestDiscoverSDK:
    def test_uses_nemo_sdk_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_sdk()
        mock_eps.assert_called_once_with(group="nemo.sdk")

    def test_accepts_sync_only_container(self) -> None:
        container = NemoPluginSDKResources(sync_resource=_make_sdk_resource)
        ep = _make_ep("example", container)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_sdk()
        assert result["example"] is container

    def test_accepts_async_only_container(self) -> None:
        container = NemoPluginSDKResources(async_resource=_make_sdk_resource)
        ep = _make_ep("example", container)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_sdk()
        assert result["example"] is container

    def test_rejects_container_without_sync_or_async_resource(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            NemoPluginSDKResources()

    def test_skips_entry_that_is_not_a_container(self) -> None:
        bad = _make_ep("bad", SimpleNamespace(sync_resource=object))
        good = _make_ep("good", NemoPluginSDKResources(sync_resource=_make_sdk_resource))
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            result = discover_sdk()
        assert "bad" not in result
        assert "good" in result


# ---------------------------------------------------------------------------
# discover_manifests
# ---------------------------------------------------------------------------


class TestDiscoverManifests:
    def test_returns_empty_when_no_entry_points(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({})):
            result = discover_manifests()
        assert result == {}

    def test_scans_all_surface_groups(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({})) as mock_eps:
            discover_manifests()
        called_groups = {c.kwargs["group"] for c in mock_eps.call_args_list}
        assert called_groups == set(_ALL_SURFACE_GROUPS)

    def test_does_not_load_entry_point_values(self) -> None:
        ep = _make_ep("alpha", object())
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.services": [ep]})):
            discover_manifests()
        ep.load.assert_not_called()

    def test_builds_manifest_from_distribution_metadata(self) -> None:
        ep = _make_ep("example", None, version="2.3.4", description="An example plugin")
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.services": [ep]})):
            result = discover_manifests()
        assert result["example"].version == "2.3.4"
        assert result["example"].description == "An example plugin"

    def test_deduplicates_across_groups(self) -> None:
        ep_svc = _make_ep("alpha", None, version="1.0.0")
        ep_cli = _make_ep("alpha", None, version="1.0.0")
        with patch(
            "nemo_platform_plugin.discovery.entry_points",
            side_effect=_eps_by_group({"nemo.services": [ep_svc], "nemo.cli": [ep_cli]}),
        ):
            result = discover_manifests()
        assert list(result.keys()) == ["alpha"]

    def test_dist_none_produces_empty_metadata(self) -> None:
        ep = _make_ep("alpha", None)
        ep.dist = None
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.services": [ep]})):
            result = discover_manifests()
        assert result["alpha"].version == ""
        assert result["alpha"].description == ""

    def test_discovers_from_any_surface_group(self) -> None:
        ep = _make_ep("cli-only", None, version="0.5.0")
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.cli": [ep]})):
            result = discover_manifests()
        assert "cli-only" in result

    def test_job_only_plugins_use_plugin_name_not_job_name(self) -> None:
        ep = _make_ep("example.say-hello", None, version="1.2.3", description="Job only plugin")
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.jobs": [ep]})):
            result = discover_manifests()
        assert list(result.keys()) == ["example"]
        assert result["example"].version == "1.2.3"

    def test_function_only_plugins_use_plugin_name_not_function_name(self) -> None:
        ep = _make_ep("example.greet", None, version="1.2.3", description="Function only plugin")
        with patch("nemo_platform_plugin.discovery.entry_points", side_effect=_eps_by_group({"nemo.functions": [ep]})):
            result = discover_manifests()
        assert list(result.keys()) == ["example"]
        assert result["example"].version == "1.2.3"


class TestDiscoverCustomizationContributors:
    def test_group_in_all_surface_groups(self) -> None:
        assert CUSTOMIZATION_CONTRIBUTORS_GROUP in _ALL_SURFACE_GROUPS

    def test_uses_customization_contributors_group(self) -> None:
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[]) as mock_eps:
            discover_customization_contributors()
        mock_eps.assert_called_once_with(group=CUSTOMIZATION_CONTRIBUTORS_GROUP)

    def test_instantiates_contributor_class(self) -> None:
        class _Contributor:
            name = "fake"
            dependencies = ["jobs"]

            def get_routers(self) -> list[RouterSpec]:
                return []

            def get_cli(self) -> None:
                return None

            def get_sdk_resources(self):
                return None

        ep = _make_ep("fake", _Contributor)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_customization_contributors()
        assert isinstance(result["fake"], _Contributor)

    def test_contributor_without_a_cli_summary_loads(self) -> None:
        """Contributing CLI help is optional; a backend without it must still load."""

        class _Contributor:
            name = "legacy"
            dependencies = ["jobs"]

            def get_routers(self) -> list[RouterSpec]:
                return []

            def get_cli(self) -> None:
                return None

            def get_sdk_resources(self):
                return None

        assert not hasattr(_Contributor, "get_cli_summary")
        ep = _make_ep("legacy", _Contributor)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            result = discover_customization_contributors()
        assert isinstance(result["legacy"], _Contributor)

    def test_failing_contributor_raises(self) -> None:
        from nemo_platform_plugin.customization_contributor import CustomizationContributorDiscoveryError

        bad = _make_ep("bad", None)
        bad.load.side_effect = RuntimeError("broken")

        class _Contributor:
            name = "good"
            dependencies = ["jobs"]

            def get_routers(self) -> list[RouterSpec]:
                return []

            def get_cli(self) -> None:
                return None

            def get_sdk_resources(self):
                return None

        good = _make_ep("good", _Contributor)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[bad, good]):
            with pytest.raises(CustomizationContributorDiscoveryError, match="Failed to load"):
                discover_customization_contributors()

    def test_name_mismatch_raises(self) -> None:
        from nemo_platform_plugin.customization_contributor import CustomizationContributorDiscoveryError

        class _Contributor:
            name = "wrong"
            dependencies = ["jobs"]

            def get_routers(self) -> list[RouterSpec]:
                return []

            def get_cli(self) -> None:
                return None

            def get_sdk_resources(self):
                return None

        ep = _make_ep("expected", _Contributor)
        with patch("nemo_platform_plugin.discovery.entry_points", return_value=[ep]):
            with pytest.raises(CustomizationContributorDiscoveryError, match="differs from class name"):
                discover_customization_contributors()
