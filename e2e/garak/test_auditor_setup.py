# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""E2E tests for auditor setup flow

Adds and lists configs, targets and their dependencies.
"""

from nemo_helix_plugin.garak.client import GarakClient
from nemo_helix_plugin.garak.types import CreateAuditConfigRequest, CreateAuditTargetRequest
from nemo_helix_plugin.client.client import NemoClient
from nhx.testing import short_unique_name


def test_create_audit_config(client: NemoClient, workspace: str) -> None:
    """Create an audit config (script e2e-multi style: system, run, plugins, reporting)."""
    auditor = GarakClient.from_client(client)
    config_name = short_unique_name("e2e-auditor-config")

    config = auditor.create_audit_config(
        workspace=workspace,
        body=CreateAuditConfigRequest(
            name=config_name,
            system={"parallel_attempts": 20, "lite": True},
            run={"generations": 7},
            plugins={"probe_spec": "grandma,encoding.InjectBase2048,encoding.InjectROT13"},
            reporting={},
        ),
    ).data()
    assert config.name == config_name
    assert config.workspace == workspace

    retrieved = auditor.get_audit_config(name=config_name, workspace=workspace).data()
    assert retrieved.name == config_name
    assert retrieved.system["parallel_attempts"] == 20
    assert retrieved.run["generations"] == 7


def test_create_audit_target(
    client: NemoClient,
    workspace: str,
) -> None:
    """Create project and audit target pointing at IGW mock provider (script minimal target)."""
    auditor = GarakClient.from_client(client)
    target_name = short_unique_name("e2e-auditor-target")
    provider_name = short_unique_name("e2e-auditor-provider")

    target = auditor.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(
            name=target_name,
            type="nim",
            model="nvdev/mistralai/mistral-7b-instruct-v0.3",
            options={
                "nim": {
                    "skip_seq_start": "<think>",
                    "skip_seq_end": "</think>",
                    "max_tokens": 4000,
                    "nhx_uri_spec": {
                        "inference_gateway": {
                            "workspace": workspace,
                            "provider": provider_name,
                        }
                    },
                }
            },
        ),
    ).data()
    assert target.name == target_name
    assert target.workspace == workspace
    assert target.type == "nim"

    retrieved = auditor.get_audit_target(name=target_name, workspace=workspace).data()
    assert retrieved.name == target_name
    nim_opts = retrieved.options or {}
    nhx_spec = (nim_opts.get("nim") or {}).get("nhx_uri_spec") or {}
    igw = nhx_spec.get("inference_gateway") or {}
    assert igw.get("provider") == provider_name


def test_create_audit_config_and_target_then_list(
    client: NemoClient,
    workspace: str,
) -> None:
    """Create config and target (script flow), list them"""
    auditor = GarakClient.from_client(client)
    config_name = short_unique_name("e2e-auditor-list-config")
    target_name = short_unique_name("e2e-auditor-list-target")
    provider_name = short_unique_name("e2e-auditor-list-provider")

    auditor.create_audit_config(
        workspace=workspace,
        body=CreateAuditConfigRequest(
            name=config_name,
            system={"parallel_attempts": 10, "lite": True},
            run={"generations": 3},
            plugins={},
            reporting={},
        ),
    )
    auditor.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(
            name=target_name,
            type="nim",
            model="nvdev/mistralai/mistral-7b-instruct-v0.3",
            options={
                "nim": {
                    "nhx_uri_spec": {
                        "inference_gateway": {"workspace": workspace, "provider": provider_name},
                    }
                }
            },
        ),
    )
    config_names = [c.name for c in auditor.list_audit_configs(workspace=workspace).items()]
    assert config_name in config_names

    target_names = [t.name for t in auditor.list_audit_targets(workspace=workspace).items()]
    assert target_name in target_names
