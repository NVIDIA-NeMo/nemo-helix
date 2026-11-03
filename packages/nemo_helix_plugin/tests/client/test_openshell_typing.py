# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for provider RPCs and nested protobuf fields; checked by ty.

Functions are intentionally not collected by pytest. The optional SDK is needed
for these contracts, as it is for the OpenShell backend tests.
"""

# ruff: noqa: E402 -- Check the optional SDK before imports.

from typing import assert_type

import pytest

pytest.importorskip("openshell")

from nemo_helix_plugin.openshell.isolation import restricted_platform_policy
from nemo_helix_plugin.openshell.provider_client import ProviderGatewayClient
from openshell._proto import datamodel_pb2 as dm
from openshell._proto import openshell_pb2 as pb
from openshell._proto.sandbox_pb2 import SandboxPolicy


async def _check_provider_types(client: ProviderGatewayClient) -> None:
    response = await client.get_provider(pb.GetProviderRequest())
    assert_type(response, pb.ProviderResponse)
    assert_type(response.provider, dm.Provider)
    assert_type(response.provider.metadata.labels["owner"], str)
    assert_type(response.provider.credentials["API_TOKEN"], str)
    assert_type(await client.create_provider(pb.CreateProviderRequest()), pb.ProviderResponse)
    assert_type(await client.update_provider(pb.UpdateProviderRequest()), pb.ProviderResponse)
    assert_type(await client.delete_provider(pb.DeleteProviderRequest()), pb.DeleteProviderResponse)
    listing = await client.list_providers(pb.ListProvidersRequest())
    assert_type(listing, pb.ListProvidersResponse)
    assert_type(listing.providers[0], dm.Provider)
    assert_type(listing.next_page_token, str)
    profile = await client.get_provider_profile(pb.GetProviderProfileRequest())
    assert_type(profile, pb.ProviderProfileResponse)
    assert_type(profile.profile.endpoints[0].host, str)
    assert_type(await client.get_sandbox(pb.GetSandboxRequest()), pb.SandboxResponse)
    assert_type(
        await client.detach_sandbox_provider(pb.DetachSandboxProviderRequest()), pb.DetachSandboxProviderResponse
    )


def _check_policy_type(policy: SandboxPolicy) -> None:
    assert_type(
        restricted_platform_policy(policy, host="helix.example", port=443, workspace="team", binaries=[]),
        SandboxPolicy,
    )
