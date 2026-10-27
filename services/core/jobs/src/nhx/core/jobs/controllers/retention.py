# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.controller import Controller, HeartbeatMixin
from nhx.core.jobs.controllers.reconciler import JobReconciler


class StorageRetentionSweeper(HeartbeatMixin, Controller):
    """Runs the reconciler's storage retention sweep on its own loop and thread.

    The sweep lists every stopped step, which takes minutes on a platform with long job history. On the
    reconcile thread it held up reconciling, and the loop's liveness check reported the whole jobs
    controller unhealthy for the length of the sweep. The sweep reports progress for each page and step
    as it goes, so a sweep that is working is never mistaken for a stalled loop.
    """

    def __init__(self, reconciler: JobReconciler) -> None:
        self._reconciler = reconciler

    def step(self) -> None:
        self._reconciler.sweep_stopped_steps(on_progress=self.emit_heartbeat)
