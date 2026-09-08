# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entrypoint for evaluation execution jobs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from nemo_platform_plugin.sdk_provider import get_async_task_sdk, get_task_sdk
from nemo_platform_plugin.tasks.dispatcher import run_task
from nemo_scaled_evals_plugin.jobs.evaluation_execution import EvaluationExecutionJob


def main() -> int:
    """Dispatch the evaluation execution job."""
    _ensure_in_cluster_kubeconfig()
    return run_task(
        EvaluationExecutionJob,
        sdk=get_task_sdk("scaled-evals"),
        async_sdk=get_async_task_sdk("scaled-evals"),
    )


def _ensure_in_cluster_kubeconfig() -> None:
    """Create the kubeconfig expected by the sandbox adapter inside a Job pod."""
    kubeconfig = os.getenv("KUBECONFIG")
    host = os.getenv("KUBERNETES_SERVICE_HOST")
    if not kubeconfig or not host or Path(kubeconfig).exists():
        return
    port = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "incluster",
                "cluster": {
                    "server": f"https://{host}:{port}",
                    "certificate-authority": str(service_account / "ca.crt"),
                },
            }
        ],
        "contexts": [
            {
                "name": "incluster",
                "context": {
                    "cluster": "incluster",
                    "user": "incluster",
                    "namespace": os.getenv("POD_NAMESPACE", "nemo-platform-scaled-evals"),
                },
            }
        ],
        "current-context": "incluster",
        "users": [
            {
                "name": "incluster",
                "user": {"tokenFile": str(service_account / "token")},
            }
        ],
    }
    path = Path(kubeconfig)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))


if __name__ == "__main__":
    sys.exit(main())
