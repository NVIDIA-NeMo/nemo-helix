# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Data Designer service.

Wraps the endpoint functions from ``data_designer.endpoints`` as direct
methods using the ``method()`` descriptor, following the files/models pattern.

The Data Designer plugin exposes streaming preview endpoints and
job-submission collections (``/jobs/create`` and retrieval job groups). The high-level SDK resource
(``DataDesignerResource``) adds frame-collection and job-polling convenience
on top of these raw HTTP calls; callers that need that convenience should
continue using the plugin's resource layer, which will construct these
typed clients internally.
"""

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.data_designer import endpoints


class _DataDesignerMethods:
    preview = method(endpoints.preview)
    retrieval_preview = method(endpoints.retrieval_preview)
    create_job = method(endpoints.create_job)
    list_jobs = method(endpoints.list_jobs)
    get_job = method(endpoints.get_job)
    delete_job = method(endpoints.delete_job)
    cancel_job = method(endpoints.cancel_job)
    get_job_status = method(endpoints.get_job_status)
    get_job_logs = method(endpoints.get_job_logs)
    list_job_results = method(endpoints.list_job_results)
    get_job_result = method(endpoints.get_job_result)
    download_job_result = method(endpoints.download_job_result)


class DataDesignerClient(_DataDesignerMethods, NemoClient):
    """Sync client for the Data Designer service API."""


class AsyncDataDesignerClient(_DataDesignerMethods, AsyncNemoClient):
    """Async client for the Data Designer service API."""
