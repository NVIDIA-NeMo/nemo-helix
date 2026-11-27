# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint for ``user-sim.evaluate`` (``python -m nemo_user_sim_plugin.tasks.evaluate``)."""

import signal
import sys
from types import FrameType

from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.tasks.dispatcher import build_ctx_from_env, run_task_with_client
from nemo_user_sim_plugin.jobs.evaluate import EvaluateJob


def _shutdown(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _shutdown)
    client = get_task_nemo_client("user-sim")
    sys.exit(run_task_with_client(EvaluateJob, client=client, ctx=build_ctx_from_env(client)))
