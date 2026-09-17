# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Agent Hardener service.

Wraps the endpoint functions from ``agent_hardener.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from nemo_platform_plugin.agent_hardener import endpoints
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method


class _AgentHardenerMethods:
    healthz = method(endpoints.healthz)

    create_war_game_job = method(endpoints.create_war_game_job)
    list_war_game_jobs = method(endpoints.list_war_game_jobs)
    get_war_game_job = method(endpoints.get_war_game_job)
    delete_war_game_job = method(endpoints.delete_war_game_job)
    cancel_war_game_job = method(endpoints.cancel_war_game_job)
    get_war_game_job_status = method(endpoints.get_war_game_job_status)
    list_war_game_job_logs = method(endpoints.list_war_game_job_logs)
    list_war_game_job_results = method(endpoints.list_war_game_job_results)
    get_war_game_job_result = method(endpoints.get_war_game_job_result)
    download_war_game_job_result = method(endpoints.download_war_game_job_result)

    get_manifest = method(endpoints.get_manifest)
    list_manifests = method(endpoints.list_manifests)
    create_manifest = method(endpoints.create_manifest)
    inspect_project = method(endpoints.inspect_project)
    inspect_agent = method(endpoints.inspect_agent)
    update_manifest = method(endpoints.update_manifest)
    refresh_manifest = method(endpoints.refresh_manifest)
    delete_manifest = method(endpoints.delete_manifest)
    get_model_config_defaults = method(endpoints.get_model_config_defaults)
    validate_model = method(endpoints.validate_model)

    get_run = method(endpoints.get_run)
    list_runs = method(endpoints.list_runs)
    delete_run = method(endpoints.delete_run)
    apply_mitigation = method(endpoints.apply_mitigation)
    compose_defense = method(endpoints.compose_defense)
    ingest_event = method(endpoints.ingest_event)
    get_events = method(endpoints.get_events)

    create_synth_benign_job = method(endpoints.create_synth_benign_job)
    list_synth_benign_jobs = method(endpoints.list_synth_benign_jobs)
    get_synth_benign_job = method(endpoints.get_synth_benign_job)
    delete_synth_benign_job = method(endpoints.delete_synth_benign_job)
    cancel_synth_benign_job = method(endpoints.cancel_synth_benign_job)
    get_synth_benign_job_status = method(endpoints.get_synth_benign_job_status)
    list_synth_benign_job_logs = method(endpoints.list_synth_benign_job_logs)
    list_synth_benign_job_results = method(endpoints.list_synth_benign_job_results)
    get_synth_benign_job_result = method(endpoints.get_synth_benign_job_result)
    download_synth_benign_job_result = method(endpoints.download_synth_benign_job_result)


class AgentHardenerClient(_AgentHardenerMethods, NemoClient):
    """Sync client for the Agent Hardener service API."""


class AsyncAgentHardenerClient(_AgentHardenerMethods, AsyncNemoClient):
    """Async client for the Agent Hardener service API."""
