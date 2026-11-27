# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mounts the UserSim plugin clients on the platform SDK as ``client.user_sim``."""

from nemo_helix_plugin.sdk import NemoPluginSDKResources
from nemo_user_sim_plugin.client import AsyncUserSimClient, UserSimClient

user_sim_sdk_resources = NemoPluginSDKResources(
    sync_resource=UserSimClient.from_client,
    async_resource=AsyncUserSimClient.from_client,
)
