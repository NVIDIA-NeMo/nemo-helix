# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-gated wiring for the Harbor/OpenSandbox POC.

The dispatcher reuses agent-evaluate's archive preparation and scoring. The
temporary scaled-evals environment is supplied by the POC worker image; its
service entrypoints are not installed or started.
"""

import logging
import os

from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig

logger = logging.getLogger(__name__)

POC_ENABLED = "NHX_EVALS_HARBOR_OPENSANDBOX_POC"
POC_WORKER_IMAGE = "NHX_EVALS_HARBOR_POC_WORKER_IMAGE"
POC_ENVIRONMENT = "scaled_evals.harbor_opensandbox_environment:NemoOpenSandboxEnvironment"
POC_COMMAND = ["nemo_evals.tasks.harbor_task_dispatch"]


def enabled() -> bool:
    """Enable this development path only when the operator explicitly selects it."""
    return os.environ.get(POC_ENABLED) == "1"


def worker_environment() -> list[EnvironmentVariable]:
    """Forward public connection settings; Jobs injects its Kubernetes Secret separately."""
    domain = os.environ["OPENSANDBOX_DOMAIN"]
    result = [
        EnvironmentVariable(name=POC_ENABLED, value="1"),
        EnvironmentVariable(name="OPENSANDBOX_DOMAIN", value=domain),
        EnvironmentVariable(name="OPENSANDBOX_PROTOCOL", value=os.environ.get("OPENSANDBOX_PROTOCOL", "http")),
    ]
    return result


def configure_runtime(config: HarborRuntimeConfig, ctx: JobContext) -> HarborRuntimeConfig:
    """Select NeMo's real environment, serial trials, labelled ownership, and bounded sandbox lifetime."""
    if not enabled():
        return config
    environment = HarborEnvironmentConfig(
        import_path=POC_ENVIRONMENT,
        kwargs={
            "domain": os.environ["OPENSANDBOX_DOMAIN"],
            "protocol": os.environ.get("OPENSANDBOX_PROTOCOL", "http"),
            "use_server_proxy": True,
            "ready_timeout_sec": 300,
            "request_timeout_sec": 120,
            "sandbox_timeout_sec": 900,
            "trusted_allowed_hosts": [],
            "metadata": {
                "nemo-poc": "harbor-evals",
                "nemo-job": ctx.job_id or "local-poc",
                "nemo-attempt": os.environ.get("NEMO_JOB_ATTEMPT_ID", "local"),
                "nemo-owner": os.environ.get("HOSTNAME", "local"),
            },
        },
    )
    logger.warning("Harbor POC selects %s via %s", environment.import_path, os.environ["OPENSANDBOX_DOMAIN"])
    return config.model_copy(update={"environment": environment, "n_concurrent_trials": 1, "max_retries": 0})
