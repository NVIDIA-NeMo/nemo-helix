# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Guardrails plugin e2e tests."""

from collections.abc import Callable, Iterator

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.client import NemoClient

from e2e.guardrails.utils import (
    ChatOutcome,
    ConfigMode,
    GuardrailsChatTestCase,
    RailType,
    content_safety_config,
    create_guarded_virtual_model,
    create_guardrail_config,
    delete_guardrail_config,
    setup_mock_provider,
    unique_name,
)


@pytest.fixture
def guardrails_chat_test_case(
    sdk: NeMoHelix,
    client: NemoClient,
    workspace: str,
) -> Iterator[Callable[..., GuardrailsChatTestCase]]:
    created_configs: list[tuple[str, str]] = []

    def _factory(
        *,
        config_mode: ConfigMode,
        outcome: ChatOutcome,
        rail_types: tuple[RailType, ...],
        streaming: bool = False,
    ) -> GuardrailsChatTestCase:
        test_case = GuardrailsChatTestCase(
            client=client,
            workspace=workspace,
            virtual_model_name=unique_name("gr-vm"),
            backend_model_name=unique_name("main-model"),
            content_safety_model_name=unique_name("cs-model"),
            config_name=unique_name("gr-config"),
            config_mode=config_mode,
            outcome=outcome,
            rail_types=rail_types,
        )
        config_data = content_safety_config(
            content_safety_model_ref=test_case.content_safety_model_ref,
            rail_types=rail_types,
            streaming=streaming,
        )
        setup_mock_provider(sdk, test_case)
        create_guarded_virtual_model(client=client, test_case=test_case, config_data=config_data)
        if config_mode == "referenced":
            created_configs.append((workspace, test_case.config_name))
        return test_case

    yield _factory

    for config_workspace, config_name in created_configs:
        try:
            delete_guardrail_config(client, workspace=config_workspace, name=config_name)
        except Exception:
            pass


@pytest.fixture
def guardrails_check_test_case(
    sdk: NeMoHelix,
    client: NemoClient,
    workspace: str,
) -> Iterator[Callable[..., tuple[GuardrailsChatTestCase, dict]]]:
    created_configs: list[tuple[str, str]] = []

    def _factory(
        *,
        config_mode: ConfigMode,
        outcome: ChatOutcome,
        rail_types: tuple[RailType, ...],
    ) -> tuple[GuardrailsChatTestCase, dict]:
        test_case = GuardrailsChatTestCase(
            client=client,
            workspace=workspace,
            virtual_model_name=unique_name("gr-vm"),
            backend_model_name=unique_name("main-model"),
            content_safety_model_name=unique_name("cs-model"),
            config_name=unique_name("gr-config"),
            config_mode=config_mode,
            outcome=outcome,
            rail_types=rail_types,
        )
        config_data = content_safety_config(
            content_safety_model_ref=test_case.content_safety_model_ref,
            rail_types=rail_types,
            streaming=False,
        )

        setup_mock_provider(sdk, test_case)

        if config_mode == "referenced":
            create_guardrail_config(
                client,
                workspace=workspace,
                name=test_case.config_name,
                description="E2E content-safety Guardrails checks config",
                config_data=config_data,
            )
            created_configs.append((workspace, test_case.config_name))

        return test_case, config_data

    yield _factory

    for config_workspace, config_name in created_configs:
        try:
            delete_guardrail_config(client, workspace=config_workspace, name=config_name)
        except Exception:
            pass
