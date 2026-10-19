# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task Job entrypoint: run one Harbor task of a taskset run on OpenSandbox, score it, and upload its bundle."""

import os
import sys

from nemo_evals.jobs.harbor_task_dispatch import HarborTaskAgentEvalJob
from nemo_evals.tasks.runner import run_async_task_main


def alias_sandbox_environment() -> None:
    """Expose Jobs' injected OPEN_SANDBOX_* variables under the OpenSandbox SDK's OPENSANDBOX_* names."""
    for suffix in ("DOMAIN", "API_KEY"):
        source = f"OPEN_SANDBOX_{suffix}"
        if source in os.environ:
            os.environ.setdefault(f"OPENSANDBOX_{suffix}", os.environ[source])


def main() -> int:
    """Run the task Job's step config, a ``HarborTaskJobSpec``."""
    alias_sandbox_environment()
    return run_async_task_main(HarborTaskAgentEvalJob, service_name="evals")


if __name__ == "__main__":
    sys.exit(main())
