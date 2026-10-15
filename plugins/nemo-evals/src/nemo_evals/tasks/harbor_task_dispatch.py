# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POC dispatcher: materialize published tasks, run Harbor/OpenSandbox, score, and upload evidence."""

import os
import sys

from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.harbor_opensandbox_poc import enabled
from nemo_evals.tasks.runner import run_async_task_main


def main() -> int:
    """Run the existing job contract through the explicitly enabled OpenSandbox path."""
    if not enabled():
        raise RuntimeError("The Harbor/OpenSandbox POC dispatcher must be enabled by the deployment")
    # The Jobs Kubernetes backend injects these names through secretKeyRef.
    for suffix in ("DOMAIN", "API_KEY"):
        source = f"OPEN_SANDBOX_{suffix}"
        if source in os.environ:
            os.environ.setdefault(f"OPENSANDBOX_{suffix}", os.environ[source])
    return run_async_task_main(AsyncAgentEvalJob, service_name="evals")


if __name__ == "__main__":
    sys.exit(main())
