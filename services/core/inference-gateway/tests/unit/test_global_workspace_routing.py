# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing against the global workspace (ASTD-526)."""

from __future__ import annotations

from datetime import datetime

from nemo_platform.types.inference import ModelProvider
from nemo_platform.types.inference.virtual_model import VirtualModel
from nmp.common.entities.global_workspace import workspace_lookup_order
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelEntityInfo, ModelProviderInfo
from nmp.core.inference_gateway.api.v2.openai import resolve_vm_for_model
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache


def _make_vm(workspace: str, name: str) -> VirtualModel:
    return VirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        default_model_entity=f"{workspace}/{name}",
    )


def test_lookup_order_is_local_then_global() -> None:
    assert workspace_lookup_order("team-a") == ("team-a", "default")


def test_lookup_order_does_not_repeat_global() -> None:
    assert workspace_lookup_order("default") == ("default",)


class TestVirtualModelCache:
    def test_global_vm_resolves_from_another_workspace(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert cache.get("team-a", "shared-llm") is not None

    def test_local_vm_shadows_global(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "llm"), _make_vm("team-a", "llm")])
        resolved = cache.get("team-a", "llm")
        assert resolved is not None
        assert resolved.workspace == "team-a"

    def test_workspace_scoped_vm_does_not_leak_to_other_workspaces(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("team-a", "private-llm")])
        assert cache.get("team-b", "private-llm") is None

    def test_unknown_name_still_misses(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert cache.get("team-a", "nope") is None


class TestResolveVMForModel:
    def test_global_vm_resolves_for_plain_name(self) -> None:
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "shared-llm")])
        assert resolve_vm_for_model(cache, "team-a", "shared-llm") is not None

    def test_lora_composite_resolves_through_global_base(self) -> None:
        """An adapter fine-tuned in a workspace routes via the shared base model's VM."""
        cache = VirtualModelCache()
        cache.rebuild([_make_vm("default", "base-llm")])
        resolved = resolve_vm_for_model(cache, "team-a", "base-llm&adapters/team-a/my-adapter")
        assert resolved is not None
        assert resolved.workspace == "default"


class TestModelCache:
    def _provider_info(self, workspace: str, name: str) -> ModelProviderInfo:
        return ModelProviderInfo(
            model_provider=ModelProvider(
                workspace=workspace,
                name=name,
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )

    def test_global_provider_resolves_from_another_workspace(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("default", "shared-provider"))
        assert cache.get_from_provider("team-a", "shared-provider") is not None

    def test_local_provider_shadows_global(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("default", "prov"))
        cache.update_model_info(self._provider_info("team-a", "prov"))
        resolved = cache.get_from_provider("team-a", "prov")
        assert resolved is not None
        assert resolved.model_provider.workspace == "team-a"

    def test_private_provider_does_not_leak(self) -> None:
        cache = ModelCache()
        cache.update_model_info(self._provider_info("team-a", "prov"))
        assert cache.get_from_provider("team-b", "prov") is None

    def test_global_model_entity_resolves_from_another_workspace(self) -> None:
        cache = ModelCache()
        cache.model_entity_info_map[("default", "shared-llm")] = ModelEntityInfo(workspace="default", name="shared-llm")
        assert cache.get_from_model_entity("team-a", "shared-llm") is not None

    def test_local_model_entity_shadows_global(self) -> None:
        cache = ModelCache()
        cache.model_entity_info_map[("default", "llm")] = ModelEntityInfo(workspace="default", name="llm")
        cache.model_entity_info_map[("team-a", "llm")] = ModelEntityInfo(workspace="team-a", name="llm")
        resolved = cache.get_from_model_entity("team-a", "llm")
        assert resolved is not None
        assert resolved.workspace == "team-a"

    def test_private_model_entity_does_not_leak(self) -> None:
        cache = ModelCache()
        cache.model_entity_info_map[("team-a", "llm")] = ModelEntityInfo(workspace="team-a", name="llm")
        assert cache.get_from_model_entity("team-b", "llm") is None
