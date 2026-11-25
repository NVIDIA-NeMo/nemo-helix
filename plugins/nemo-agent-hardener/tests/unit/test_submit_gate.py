# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The war-game compile gate: a bad input is rejected at submit, before a job queues or a run record exists."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _doubles import SecretsStore
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.model_config import ModelChoice, WarGameModels
from nemo_helix_plugin.client.errors import InternalServerError
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.jobs import executor as executor_module
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.execution_profiles import DockerJobExecutionProfile, DockerJobExecutionProfileConfig

_DECLARED = "SUBMIT_GATE_VICTIM_KEY"


class _Entities:
    """Serves one saved manifest (or none) and records the run records compile pre-creates."""

    def __init__(self, manifest: AgentHardenerManifest | None) -> None:
        self._manifest = manifest
        self.created: list[Any] = []

    async def get(self, _entity_type: Any, *, name: str, workspace: str) -> AgentHardenerManifest:
        if self._manifest is None:
            raise NemoEntityNotFoundError(f"{workspace}/{name}")
        return self._manifest

    async def create(self, entity: Any) -> Any:
        self.created.append(entity)
        return entity.model_copy(update={"name": "run-1"})


@pytest.fixture(autouse=True)
def _host_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AgentHardenerConfig(
        venv_path=tmp_path / "venv", garak_venv_path=tmp_path / "garak", operator_env_file=tmp_path / "absent.env"
    )
    monkeypatch.setattr(AgentHardenerConfig, "get", lambda: config)
    monkeypatch.delenv(_DECLARED, raising=False)


async def _compile(
    manifest: AgentHardenerManifest | None,
    store: SecretsStore | None = None,
    entities: _Entities | None = None,
    **spec: Any,
) -> dict[str, Any]:
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default",
        spec=run_module.WarGameSpec(manifest_id="m1", driver="service", **spec),
        entity_client=entities or _Entities(manifest),
        job_name="job-1",
        async_sdk=(store or SecretsStore(set())).client(),
    )
    (step,) = list(job_spec["steps"])
    return step["config"]


async def _rejection(manifest: AgentHardenerManifest | None, store: SecretsStore | None = None, **spec: Any) -> str:
    """Compile expecting a submit-time rejection; a rejected war-game must leave no run record behind."""
    entities = _Entities(manifest)
    with pytest.raises(HelixJobCompilationError) as exc:
        await _compile(manifest, store, entities, **spec)
    assert entities.created == []
    return str(exc.value)


def _manifest(**fields: Any) -> AgentHardenerManifest:
    return AgentHardenerManifest(name="m1", workspace="default", **fields)


async def test_an_unknown_manifest_is_rejected_by_name() -> None:
    assert "Manifest 'm1' not found in workspace 'default'" in await _rejection(None)


async def test_missing_secret_refs_are_named_without_reading_any_value() -> None:
    store = SecretsStore({"default/deployed"})
    manifest = _manifest(secret_refs={"GITHUB_TOKEN": "team/gh"}, deployed_secret_refs={"OTHER": "default/deployed"})

    message = await _rejection(manifest, store)

    assert "team/gh (GITHUB_TOKEN)" in message
    assert "deployed" not in message
    assert not any(path.endswith("/access") for path in store.paths)


async def test_an_unreachable_secrets_store_is_not_reported_as_missing_secrets() -> None:
    with pytest.raises(InternalServerError):
        await _compile(_manifest(secret_refs={"K": "default/k"}), SecretsStore(set(), status=503))


async def test_a_declared_secret_no_source_provides_is_rejected_with_the_fix() -> None:
    message = await _rejection(_manifest(secrets=[_DECLARED]))

    assert f"missing required secrets for the victim agent: {_DECLARED}" in message
    assert f"--secrets {_DECLARED}=<workspace>/submit-gate-victim-key" in message


async def test_declared_secrets_count_every_source_the_job_assembles(tmp_path: Path) -> None:
    env_file = tmp_path / "victim.env"
    env_file.write_text(f"{_DECLARED}=from-file\n", encoding="utf-8")
    manifest = _manifest(
        secrets=[_DECLARED, "FROM_REF", "INFERENCE_API_KEY", "NIM_API_KEY"],
        secret_refs={"FROM_REF": "default/ref"},
    )
    models = WarGameModels(attack=ModelChoice(api_key_secret="attack-key"))

    config = await _compile(
        manifest, SecretsStore({"default/ref", "default/attack-key"}), env_file=str(env_file), models=models
    )

    assert config["run_name"] == "run-1"


@pytest.mark.parametrize(
    "execution_profiles",
    [[DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())]],
)
async def test_a_container_job_leaves_declared_secrets_to_the_task_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pod's environment is invisible here, so only Secrets-store refs can be checked at submit.
    config = AgentHardenerConfig(
        venv_path=tmp_path / "venv", container_jobs=True, openshell_gateway_endpoint="http://openshell:8080"
    )
    monkeypatch.setattr(AgentHardenerConfig, "get", lambda: config)
    monkeypatch.setattr(executor_module, "get_qualified_image", lambda name: f"reg.example/{name}:t")

    step_config = await _compile(_manifest(secrets=[_DECLARED], victim_image="reg/victim:1"))

    assert step_config[run_module._MODELS_PREFLIGHTED] is True


async def test_an_unavailable_model_is_rejected_listing_what_the_key_reaches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "get_nemo_client", lambda **_k: object())
    monkeypatch.setattr(_common, "resolve_model_key", lambda *_a, **_k: "k")
    monkeypatch.setattr(
        run_module,
        "validate_choice",
        lambda *_a: SimpleNamespace(ok=False, reason="unknown_model", available=["real/a"], detail=""),
    )
    models = WarGameModels(analysis=ModelChoice(model="typo", base_url="https://x/v1"))

    message = await _rejection(_manifest(), models=models)

    assert "'typo' is not available" in message and "real/a" in message
    assert "Check the model name" in message


async def test_compile_probes_the_merged_models_and_tells_the_job_not_to_repeat_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(run_module, "get_nemo_client", lambda **_k: object())
    monkeypatch.setattr(_common, "resolve_model_key", lambda *_a, **_k: "k")
    monkeypatch.setattr(
        run_module, "validate_choice", lambda model, *_a: probes.append(model) or SimpleNamespace(ok=True)
    )
    stored = WarGameModels(analysis=ModelChoice(model="stored", base_url="https://x/v1"))
    override = WarGameModels(analysis=ModelChoice(model="override"))

    config = await _compile(_manifest(models=stored), models=override)

    assert probes == ["override"]
    assert config[run_module._MODELS_PREFLIGHTED] is True
