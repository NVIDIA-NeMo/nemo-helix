# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import data_designer.config as dd
from data_designer.config.default_model_settings import get_default_providers
from data_designer.engine.model_provider import ModelProvider as NDDModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry, resolve_model_provider_registry
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoTransportError, NotFoundError, PermissionDeniedError
from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient
from nemo_platform_plugin.models.types import ModelProvider as NMPModelProvider

logger = logging.getLogger(__name__)

# TODO: Clients are now all using the DataDesigner interface object, which accepts list[NDDModelProvider]
# as input, _not_ a ModelProviderRegistry. We should restructure the utility functions in this module
# around returning that directly instead of forcing clients to make a registry only to access `registry.providers`.
# When picking up that work, consider whether returning an empty list and/or None is still valid, or if we
# should _always_ return a non-empty list (or raise).


class InvalidModelProviderReferenceError(Exception): ...


def parse_provider_reference(ref: str, default_workspace: str) -> tuple[str, str]:
    match ref.split("/"):
        case [provider_name]:
            return default_workspace, provider_name
        case [workspace, provider_name]:
            return workspace, provider_name
        case _:
            raise InvalidModelProviderReferenceError(f"Invalid model provider reference: {ref!r}")


_NO_OP = "no-op"


def make_noop_provider() -> dd.ModelProvider:
    return NDDModelProvider(name=_NO_OP, endpoint=_NO_OP)


def make_null_registry() -> ModelProviderRegistry:
    # While relatively useless in practice, a DataDesignerConfig that does not use LLMs in any columns
    # is semantically valid. The library requires a non-empty ModelProviderRegistry, so in this scenario
    # we can provide this dummy null registry.
    return ModelProviderRegistry(
        providers=[make_noop_provider()],
    )


def _make_local_model_provider_registry() -> ModelProviderRegistry | None:
    providers = get_default_providers()
    if len(providers) > 0:
        return resolve_model_provider_registry(providers)


@dataclass
class ModelProviderCollection:
    sdk: AsyncNeMoPlatform
    default_workspace: str

    # key = user-supplied provider name
    providers: dict[str, tuple[NDDModelProvider, NMPModelProvider]] = field(default_factory=dict)
    config_errors: list[str] = field(default_factory=list)
    internal_errors: list[str] = field(default_factory=list)
    inaccessible_providers: set[str] = field(default_factory=set)

    async def add(self, model_config: dd.ModelConfig) -> None:
        user_supplied_provider_name = model_config.provider
        try:
            workspace, provider_name = parse_provider_reference(user_supplied_provider_name, self.default_workspace)
        except InvalidModelProviderReferenceError:
            self.config_errors.append(
                f"Malformed model provider {user_supplied_provider_name!r} for model config with alias {model_config.alias!r}"
            )
            return

        await self._add_provider(
            user_supplied_provider_name=user_supplied_provider_name,
            workspace=workspace,
            provider_name=provider_name,
        )

        self._ensure_model_is_enabled(user_supplied_provider_name, model_config)

    async def _add_provider(
        self, user_supplied_provider_name: str, workspace: str, provider_name: str
    ) -> tuple[NDDModelProvider, NMPModelProvider] | None:
        # Don't attempt to add a provider we already saw and failed to retrieve
        if user_supplied_provider_name in self.inaccessible_providers:
            return

        # If we already saw and added this provider, early return
        if user_supplied_provider_name in self.providers:
            return self.providers[user_supplied_provider_name]

        nmp_provider = await self._get_nmp_provider(user_supplied_provider_name, workspace, provider_name)
        if nmp_provider is None:
            return

        models = client_from_platform(self.sdk, AsyncModelsClient)
        ndd_provider = NDDModelProvider(
            name=user_supplied_provider_name,
            endpoint=models.get_provider_route_openai_url(nmp_provider),
            extra_headers={k: v for k, v in self.sdk.default_headers.items() if isinstance(v, str)},
        )
        providers = (ndd_provider, nmp_provider)
        self.providers[user_supplied_provider_name] = providers
        return providers

    async def _get_nmp_provider(
        self, user_supplied_provider_name: str, workspace: str, provider_name: str
    ) -> NMPModelProvider | None:
        try:
            return await get_nmp_provider_async(self.sdk, workspace, provider_name)
        except (NotFoundError, PermissionDeniedError):
            self.config_errors.append(
                f"Cannot access provider {user_supplied_provider_name!r}. Check that it exists and you have access to it."
            )
            self.inaccessible_providers.add(user_supplied_provider_name)
        except NemoTransportError as e:
            logger.debug(
                "Error connecting while retrieving model provider",
                extra={"provider_name": provider_name, "workspace": workspace},
                exc_info=True,
            )
            self.internal_errors.append(
                "Could not connect to Models or Inference Gateway while resolving "
                f"provider {user_supplied_provider_name!r}: {e}"
            )
            self.inaccessible_providers.add(user_supplied_provider_name)
        except Exception as e:
            logger.exception(
                "Error retrieving model provider", extra={"provider_name": provider_name, "workspace": workspace}
            )
            self.internal_errors.append(f"Error retrieving model provider {user_supplied_provider_name!r}: {e}")
            self.inaccessible_providers.add(user_supplied_provider_name)

    def _ensure_model_is_enabled(self, user_supplied_provider_name: str, model_config: dd.ModelConfig) -> None:
        if user_supplied_provider_name not in self.providers:
            return

        nmp_provider = self.providers[user_supplied_provider_name][1]
        if nmp_provider.enabled_models and model_config.model not in nmp_provider.enabled_models:
            self.config_errors.append(
                f"Model {model_config.model!r} is not enabled for provider {user_supplied_provider_name!r}"
            )

    def get_model_provider_registry(self) -> ModelProviderRegistry | None:
        if len(self.config_errors) > 0:
            raise NDDInvalidConfigError(f"Errors in model configs: {self.config_errors}")

        if len(self.internal_errors) > 0:
            raise NDDInternalError(f"Unexpected errors occurred retrieving model providers: {self.internal_errors}")

        if len(self.providers.values()) > 0:
            registry_providers = [providers_tuple[0] for providers_tuple in self.providers.values()]
            return ModelProviderRegistry(providers=registry_providers)


async def make_model_provider_registry(
    model_configs: list[dd.ModelConfig],
    *,
    sdk: AsyncNeMoPlatform,
    default_workspace: str,
) -> ModelProviderRegistry | None:
    """Creates a ModelProviderRegistry that can be passed to the Data Designer library
    to handle the user request, with all providers pointing to Inference Gateway.

    Returns:
        A model provider registry, or None if model_configs is empty.

    Raises:
        NDDInvalidConfigError or NDDInternalError
    """
    collection = ModelProviderCollection(sdk, default_workspace)
    for model_config in model_configs:
        await collection.add(model_config)

    return collection.get_model_provider_registry()


def get_nmp_provider(sdk: NeMoPlatform, workspace: str, provider_name: str) -> NMPModelProvider:
    models = client_from_platform(sdk, ModelsClient)
    return models.get_provider(workspace=workspace, name=provider_name).data()


async def get_nmp_provider_async(sdk: AsyncNeMoPlatform, workspace: str, provider_name: str) -> NMPModelProvider:
    models = client_from_platform(sdk, AsyncModelsClient)
    response = await models.get_provider(workspace=workspace, name=provider_name)
    return response.data()
