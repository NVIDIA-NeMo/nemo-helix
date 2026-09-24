# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Guardrails plugin e2e tests."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.guardrail.types import CreateGuardrailConfigRequest

from e2e.guardrails.utils import (
    ChatOutcome,
    ConfigMode,
    GuardrailsChatTestCase,
    RailType,
    content_safety_config,
    create_guarded_virtual_model,
    guardrail_client_from_sdk,
    setup_mock_provider,
    unique_name,
)


@pytest.fixture
def guardrails_chat_test_case(
    sdk: NeMoHelix,
    workspace: str,
) -> Iterator[Callable[..., GuardrailsChatTestCase]]:
    created_configs: list[tuple[str, str]] = []
    guardrail_client = guardrail_client_from_sdk(sdk)

    def _factory(
        *,
        config_mode: ConfigMode,
        outcome: ChatOutcome,
        rail_types: tuple[RailType, ...],
        streaming: bool = False,
    ) -> GuardrailsChatTestCase:
        test_case = GuardrailsChatTestCase(
            sdk=sdk,
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
        create_guarded_virtual_model(sdk=sdk, test_case=test_case, config_data=config_data)
        if config_mode == "referenced":
            created_configs.append((workspace, test_case.config_name))
        return test_case

    yield _factory

    for config_workspace, config_name in created_configs:
        try:
            guardrail_client.delete_guardrail_config(workspace=config_workspace, name=config_name)
        except Exception:
            pass


@pytest.fixture
def guardrails_check_test_case(
    sdk: NeMoHelix,
    workspace: str,
) -> Iterator[Callable[..., tuple[GuardrailsChatTestCase, dict[str, Any]]]]:
    created_configs: list[tuple[str, str]] = []
    guardrail_client = guardrail_client_from_sdk(sdk)

    def _factory(
        *,
        config_mode: ConfigMode,
        outcome: ChatOutcome,
        rail_types: tuple[RailType, ...],
    ) -> tuple[GuardrailsChatTestCase, dict[str, Any]]:
        test_case = GuardrailsChatTestCase(
            sdk=sdk,
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
            guardrail_client.create_guardrail_config(
                workspace=workspace,
                body=CreateGuardrailConfigRequest(
                    name=test_case.config_name,
                    description="E2E content-safety Guardrails checks config",
                    data=config_data,
                ),
            )
            created_configs.append((workspace, test_case.config_name))

        return test_case, config_data

    yield _factory

    for config_workspace, config_name in created_configs:
        try:
            guardrail_client.delete_guardrail_config(workspace=config_workspace, name=config_name)
        except Exception:
            pass
