# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level progress reporting for training tasks.

Provides progress reporting to the Jobs service using the NeMo Platform SDK.
``JobsServiceProgressReporter`` handles high-level phase reporting for the
training runner; backends subclass it (or instantiate it directly) supplying
their own ``service_name`` so the task SDK resolves the right credentials.

The Jobs service MERGES ``status_details`` key-wise rather than replacing the
blob -- ``JobDispatcher._update_status_details_object``, applied both to the task
and to the copy propagated up to the job. A field therefore survives every later
update that does not restate it. The merge is shallow: a key that is sent
replaces the stored value wholesale.

That is what lets the callback restate every scalar on every report rather than
only the ones a given step observed -- restating costs a handful of numbers and
makes a dropped report self-healing, while a key left unmentioned is left alone.
The accumulated series are the expensive term and are sent only when they change.

For training-specific metrics (loss, validation, checkpoints) see the
``TrainingProgressCallback`` which composes this reporter.
"""

import logging
import os
from collections.abc import Mapping
from typing import Any, cast

import httpx
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.jobs.types import PlatformJobTaskUpdate
from nmp.common.client_factory import get_task_nemo_client
from nmp.customization_common.service.context import NMPJobContext

logger = logging.getLogger(__name__)

# The SDK defaults to a 60s read/write timeout and retries twice, so one wedged
# report can hold the training thread for roughly three minutes. That trade is
# wrong here: a healthy update costs 46ms plus 0.31ms/KB, and a lost one costs
# almost nothing, because every report carries the whole series -- the next one
# supersedes it. Fail fast and let training continue.
#
# Retries stay at the SDK default. They are cheap once the timeout is short, and
# the final flush from close() is the one report with no successor to repair it.
_REPORT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class JobsServiceProgressReporter:
    """Reports high-level progress to the Jobs service."""

    def __init__(self, job_ctx: NMPJobContext, service_name: str):
        self._job_ctx = job_ctx
        # with_options rather than handing get_task_nemo_client a client: that function
        # skips its workload-identity branch entirely when passed one, so
        # supplying a client just to set a timeout would quietly change how the
        # task authenticates.
        self._client = get_task_nemo_client(service_name)
        self._jobs = JobsClient.from_client(self._client).with_options(timeout=_REPORT_TIMEOUT)
        self._is_main_rank = int(os.environ.get("RANK", "0")) == 0
        self._max_steps = 0
        self._num_epochs = 0

        # Gate on real job context, not bare truthiness: from_env() fills missing
        # identifiers with non-empty sentinel defaults, which would otherwise
        # enable reporting (and failing SDK calls) outside a real job run.
        self._enabled = self._is_main_rank and self._job_ctx.is_configured

    def configure_progress_tracking(self, max_steps: int, num_epochs: int) -> None:
        """Configure progress tracking at the start of training."""
        self._max_steps = max_steps
        self._num_epochs = num_epochs

    def _calculate_percentage_done(self, step: int | None) -> int:
        if step is None or self._max_steps <= 0:
            return 0
        # Clamp to 100: step can exceed max_steps (e.g. resumed/over-run), and
        # downstream progress consumers expect a bounded percentage.
        return min(100, int((step / self._max_steps) * 100))

    def update_task(
        self,
        status: str = "active",
        status_details: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return

        if not self._is_main_rank:
            return

        try:
            self._jobs.update_job_step_task(
                name=self._job_ctx.normalized_task,
                workspace=self._job_ctx.workspace,
                job=self._job_ctx.job_id,
                step=self._job_ctx.step,
                body=PlatformJobTaskUpdate(
                    status=PlatformJobStatus(status),
                    status_details=status_details or {},
                    error_details=error_details or {},
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to update task progress: {e}")

    def fetch_current_metrics(self) -> dict[str, list[dict[str, float | int]]] | None:
        """Read back every stored metric series, to seed a new process.

        The only read this reporter makes, and it happens once per process. The
        accumulator lives in memory, so a process taking over a task would report
        from empty and replace what the task had stored; reading it back is what
        keeps the curves continuous. Predates the per-metric series -- the two
        loss curves were seeded the same way.

        Because the server merges, no write path needs to know what is already
        stored -- a report that omits a field leaves it standing.

        Deliberately not restricted to a known set of names: backends decide what
        they accumulate, and seeding only ``train_loss`` would silently restart
        every other curve from empty. Non-list values are
        dropped so a malformed blob cannot poison the accumulator, and each list
        is copied so the caller's accumulator does not alias the response.

        Returns ``{}`` when there is nothing to seed from and ``None`` when the
        read itself failed, which are not the same thing and must not look the
        same to the caller. The merge is wholesale per key, so a caller that
        treats a failed read as "nothing stored" sends its own partial
        accumulator under ``metrics`` and destroys the history it could not read
        -- see :class:`~nmp.customization_common.training.callbacks.TrainingProgressCallback`.

        Only a 404 means "nothing stored": every other failure is a transport or
        server problem, and a distributed launch racing a briefly unreachable
        Jobs service hits those, not the 404.

        A blob that reads back malformed is a successful read of unusable data,
        so it seeds nothing and is *not* an error: overwriting it is the repair.
        Neither branch may raise -- this runs from the callback's constructor,
        which backends build outside any try, so raising here kills the training
        process rather than costing it its seed.
        """
        if not self._enabled:
            return {}

        try:
            task = self._jobs.get_job_step_task(
                name=self._job_ctx.normalized_task,
                workspace=self._job_ctx.workspace,
                job=self._job_ctx.job_id,
                step=self._job_ctx.step,
            ).data()
            stored = task.status_details or {}
        except NotFoundError:
            # Expected on a first run, where the task has no stored details yet.
            logger.info("No stored status details to seed from: task not found")
            return {}
        except Exception as e:
            logger.warning(f"Could not read stored metric series to seed from: {e}")
            return None

        if not isinstance(stored, Mapping):
            logger.warning(f"Stored status details are not an object ({type(stored).__name__}); seeding nothing")
            return {}

        metrics = stored.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            logger.warning(f"Stored metrics are not an object ({type(metrics).__name__}); seeding nothing")
            return {}

        return cast(
            dict[str, list[dict[str, float | int]]],
            {name: list(points) for name, points in metrics.items() if isinstance(points, list)},
        )

    def report_running(self, phase: str, **details: Any) -> None:
        if "step" in details and "percentage_done" not in details and self._max_steps > 0:
            details["percentage_done"] = self._calculate_percentage_done(details["step"])

        status_details = {"phase": phase, **details}
        self.update_task(status="active", status_details=status_details)

    def report_completed(self, message: str = "Completed") -> None:
        self.update_task(status="completed", status_details={"message": message, "phase": "completed"})

    def report_error(self, error: str | dict[str, Any]) -> None:
        error_details = {"message": error} if isinstance(error, str) else error
        self.update_task(status="error", error_details=error_details)

    def close(self) -> None:
        self._client.close()
