# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Agents service.

Wraps the endpoint functions from ``agents.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from __future__ import annotations

from nemo_helix_plugin.agents import endpoints
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method


class _AgentsMethods:
    create_sample_agent = method(endpoints.create_sample_agent)
    list_post_run_actions = method(endpoints.list_post_run_actions)
    get_agent = method(endpoints.get_agent)
    list_agents = method(endpoints.list_agents)
    register_agent = method(endpoints.register_agent)
    delete_agent = method(endpoints.delete_agent)
    get_deployment = method(endpoints.get_deployment)
    list_deployments = method(endpoints.list_deployments)
    create_deployment = method(endpoints.create_deployment)
    delete_deployment = method(endpoints.delete_deployment)
    list_deployment_modes = method(endpoints.list_deployment_modes)
    get_deployment_logs = method(endpoints.get_deployment_logs)
    stream_deployment_logs = method(endpoints.stream_deployment_logs)
    create_session = method(endpoints.create_session)
    list_sessions = method(endpoints.list_sessions)
    get_session = method(endpoints.get_session)
    close_session = method(endpoints.close_session)
    delete_session = method(endpoints.delete_session)
    create_environment_spec = method(endpoints.create_environment_spec)
    list_environment_specs = method(endpoints.list_environment_specs)
    get_environment_spec = method(endpoints.get_environment_spec)
    delete_environment_spec = method(endpoints.delete_environment_spec)
    create_environment = method(endpoints.create_environment)
    list_environments = method(endpoints.list_environments)
    get_environment = method(endpoints.get_environment)
    delete_environment = method(endpoints.delete_environment)
    create_compute_spec = method(endpoints.create_compute_spec)
    list_compute_specs = method(endpoints.list_compute_specs)
    get_compute_spec = method(endpoints.get_compute_spec)
    delete_compute_spec = method(endpoints.delete_compute_spec)
    invoke_agent = method(endpoints.invoke_agent)
    invoke_deployment = method(endpoints.invoke_deployment)
    create_execute_job = method(endpoints.create_execute_job)
    get_execute_job = method(endpoints.get_execute_job)
    get_execute_job_by_id = method(endpoints.get_execute_job_by_id)
    get_execute_job_by_name = method(endpoints.get_execute_job_by_name)
    list_execute_job_results = method(endpoints.list_execute_job_results)
    list_execute_job_results_by_id = method(endpoints.list_execute_job_results_by_id)
    list_execute_job_results_by_name = method(endpoints.list_execute_job_results_by_name)
    create_agent_job = method(endpoints.create_agent_job)
    list_agent_jobs = method(endpoints.list_agent_jobs)
    get_agent_job = method(endpoints.get_agent_job)
    get_agent_job_by_id = method(endpoints.get_agent_job_by_id)
    get_agent_job_by_name = method(endpoints.get_agent_job_by_name)
    delete_agent_job = method(endpoints.delete_agent_job)
    delete_agent_job_by_id = method(endpoints.delete_agent_job_by_id)
    delete_agent_job_by_name = method(endpoints.delete_agent_job_by_name)
    cancel_agent_job = method(endpoints.cancel_agent_job)
    cancel_agent_job_by_id = method(endpoints.cancel_agent_job_by_id)
    cancel_agent_job_by_name = method(endpoints.cancel_agent_job_by_name)
    list_agent_job_logs = method(endpoints.list_agent_job_logs)
    list_agent_job_logs_by_id = method(endpoints.list_agent_job_logs_by_id)
    list_agent_job_logs_by_name = method(endpoints.list_agent_job_logs_by_name)
    list_agent_job_results = method(endpoints.list_agent_job_results)
    list_agent_job_results_by_id = method(endpoints.list_agent_job_results_by_id)
    list_agent_job_results_by_name = method(endpoints.list_agent_job_results_by_name)
    get_agent_job_status = method(endpoints.get_agent_job_status)
    get_agent_job_status_by_id = method(endpoints.get_agent_job_status_by_id)
    get_agent_job_status_by_name = method(endpoints.get_agent_job_status_by_name)
    get_agent_job_result = method(endpoints.get_agent_job_result)
    get_agent_job_result_by_id = method(endpoints.get_agent_job_result_by_id)
    get_agent_job_result_by_name = method(endpoints.get_agent_job_result_by_name)
    download_agent_job_result = method(endpoints.download_agent_job_result)
    download_agent_job_result_by_id = method(endpoints.download_agent_job_result_by_id)
    download_agent_job_result_by_name = method(endpoints.download_agent_job_result_by_name)


class AgentsClient(_AgentsMethods, NemoClient):
    """Sync client for the Agents service API."""


class AsyncAgentsClient(_AgentsMethods, AsyncNemoClient):
    """Async client for the Agents service API."""
