# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed clients for the UserSim plugin API, used by the CLI and mounted on the SDK as ``client.user_sim``."""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_user_sim_plugin.types import endpoints


class _UserSimMethods:
    health = method(endpoints.health)
    probes = method(endpoints.probes)
    suggest_probes = method(endpoints.suggest_probes)
    check_models = method(endpoints.check_models)
    create_job = method(endpoints.create_job)
    list_jobs = method(endpoints.list_jobs)
    get_job = method(endpoints.get_job)
    get_job_status = method(endpoints.get_job_status)
    download_result = method(endpoints.download_result)
    delete_job = method(endpoints.delete_job)
    cancel_job = method(endpoints.cancel_job)


class UserSimClient(_UserSimMethods, NemoClient):
    """Sync client for the UserSim plugin API."""


class AsyncUserSimClient(_UserSimMethods, AsyncNemoClient):
    """Async client for the UserSim plugin API."""
