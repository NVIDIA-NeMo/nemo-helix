# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_helix_plugin.auth.profiles import endpoints
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method


class _ProfileMethods:
    get_me = method(endpoints.get_me)
    update_my_profile = method(endpoints.update_my_profile)
    list_my_preferences = method(endpoints.list_my_preferences)
    get_my_preference = method(endpoints.get_my_preference)
    replace_my_preference = method(endpoints.replace_my_preference)
    patch_my_preference = method(endpoints.patch_my_preference)
    delete_my_preference = method(endpoints.delete_my_preference)
    list_preference_defaults = method(endpoints.list_preference_defaults)
    get_preference_defaults = method(endpoints.get_preference_defaults)
    replace_preference_defaults = method(endpoints.replace_preference_defaults)
    patch_preference_defaults = method(endpoints.patch_preference_defaults)
    delete_preference_defaults = method(endpoints.delete_preference_defaults)
    resolve_profiles = method(endpoints.resolve_profiles)


class ProfilesClient(_ProfileMethods, NemoClient):
    """Sync client for the caller's profile, preferences, and principal display resolution."""


class AsyncProfilesClient(_ProfileMethods, AsyncNemoClient):
    """Async client for the caller's profile, preferences, and principal display resolution."""
