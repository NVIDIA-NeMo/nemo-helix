# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level Safe Synthesizer job SDK helpers."""

from __future__ import annotations

import json
import logging
import time
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLog, HelixJobStatusResponse
from nemo_helix_plugin.jobs.types import JobLogsQueryParams
from nemo_safe_synthesizer.config.external_results import SafeSynthesizerSummary
from typing_extensions import Self

logger = logging.getLogger(__name__)


class ReportHtml:
    """Container for a Safe Synthesizer HTML report."""

    def __init__(self, html: str):
        self.raw_html = html
        self.as_data_uri = f"data:text/html;base64,{b64encode(self.raw_html.encode()).decode()}"

    def save(self, path: str | Path) -> None:
        """Save the evaluation report to a file."""
        Path(path).write_text(self.raw_html, encoding="utf-8")

    def display_report_in_notebook(self, width: str = "100%", height: int = 1000) -> None:
        """Display the evaluation report in a Jupyter notebook."""
        try:
            from IPython.display import IFrame, display
        except ImportError:
            logger.warning("IPython is required to display reports in notebooks. Report will not be displayed.")
            return

        display(IFrame(self.as_data_uri, width=width, height=height))

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read an evaluation report from a file."""
        return cls(Path(path).read_text(encoding="utf-8"))


class SafeSynthesizerJob:
    """Convenience wrapper for a Safe Synthesizer platform job."""

    def __init__(self, job_name: str, client: NeMoHelix, workspace: str = "default"):
        self.job_name = job_name
        self._client = client
        self._workspace = workspace
        self._jobs = client_from_platform(client, JobsClient)

    def fetch_status(self) -> str:
        """Fetch the current job status."""
        return self.fetch_status_info().status

    def fetch_status_info(self) -> HelixJobStatusResponse:
        """Fetch the current job status response."""
        return self._jobs.get_job_status(name=self.job_name, workspace=self._workspace).data()

    def wait_for_completion(
        self, poll_interval: int = 10, verbose: bool = True, log_timeout: float | None = None
    ) -> None:
        """Block until the job reaches a terminal state."""
        last_page_cursor: str | None = None
        seen_log_keys: set[str] = set()
        previous_status_info = None
        current_status_info = self.fetch_status_info()
        while current_status_info.status not in ["completed", "error", "cancelled"]:
            if verbose:
                logging_level = None
                try:
                    httpx_logger = logging.getLogger("httpx")
                    logging_level = httpx_logger.level
                    httpx_logger.setLevel("ERROR")
                    new_logs, last_page_cursor = self._fetch_logs_incremental(
                        page_cursor=last_page_cursor, timeout=log_timeout
                    )
                    for new_log in new_logs:
                        log_key = f"{new_log.timestamp}:{hash(new_log.message)}"
                        if log_key not in seen_log_keys:
                            print(new_log.message.strip())
                            seen_log_keys.add(log_key)
                except (NemoClientError, httpx.HTTPError) as e:
                    logger.warning("Error fetching logs while waiting for job completion: %s", e)
                finally:
                    if logging_level is not None:
                        logging.getLogger("httpx").setLevel(logging_level)
            current_status_info = self.fetch_status_info()
            if current_status_info != previous_status_info:
                if verbose:
                    print(
                        f"Job status changed to status: '{current_status_info.status}',",
                        f"status_details: {current_status_info.status_details},",
                        f"error_details: {current_status_info.error_details}",
                    )
                previous_status_info = current_status_info
            time.sleep(poll_interval)
        if current_status_info.status in ["error", "cancelled"]:
            raise RuntimeError(
                f"Job '{self.job_name}' ended with status '{current_status_info.status}'. "
                f"Details: {current_status_info.status_details}. "
                f"Error: {current_status_info.error_details}. "
                "Check job logs with job.print_logs() for more details."
            )

    def fetch_summary(self) -> SafeSynthesizerSummary:
        """Fetch the machine-readable job summary."""
        data = self._jobs.download_job_result(name="summary", job=self.job_name, workspace=self._workspace).read()
        return SafeSynthesizerSummary.model_validate(json.loads(data.decode("utf-8")))

    def fetch_report(self) -> ReportHtml:
        """Fetch the evaluation report as HTML."""
        data = self._jobs.download_job_result(
            name="evaluation-report", job=self.job_name, workspace=self._workspace
        ).read()
        return ReportHtml(html=data.decode("utf-8"))

    def display_report_in_notebook(self, width: str = "100%", height: int = 1000) -> None:
        """Display the evaluation report in a Jupyter notebook."""
        self.fetch_report().display_report_in_notebook(width=width, height=height)

    def save_report(self, path: str | Path) -> None:
        """Save the evaluation report to a file."""
        self.fetch_report().save(path)

    def fetch_data(self) -> pd.DataFrame:
        """Fetch generated synthetic data as a pandas DataFrame."""
        data = self._jobs.download_job_result(
            name="synthetic-data", job=self.job_name, workspace=self._workspace
        ).read()
        return pd.read_csv(BytesIO(data))

    def _fetch_logs_incremental(
        self, page_cursor: str | None = None, timeout: float | None = None
    ) -> tuple[list[HelixJobLog], str | None]:
        """Fetch logs incrementally starting from a page cursor."""
        timeout = 300.0 if timeout is None else timeout
        all_logs: list[HelixJobLog] = []
        current_cursor: str | None = page_cursor
        last_cursor_with_data: str | None = page_cursor

        while True:
            logs_query: JobLogsQueryParams = {}
            if current_cursor is not None:
                logs_query["page_cursor"] = current_cursor
            page = (
                self._jobs.with_options(timeout=timeout)
                .list_job_logs(name=self.job_name, workspace=self._workspace, query_params=logs_query)
                .page()
            )

            if page.items:
                all_logs.extend(page.items)
                if current_cursor is not None:
                    last_cursor_with_data = current_cursor

            if page.metadata["next_page"] is None:
                return all_logs, last_cursor_with_data
            current_cursor = page.metadata["next_page"]

    def fetch_logs(self, timeout: float | None = None) -> Iterator[HelixJobLog]:
        """Fetch job logs as an iterator over log objects."""
        timeout = 300.0 if timeout is None else timeout
        yield from (
            self._jobs.with_options(timeout=timeout)
            .list_job_logs(name=self.job_name, workspace=self._workspace)
            .items()
        )

    def print_logs(self, timeout: float | None = None) -> None:
        """Print job logs to stdout."""
        for log in self.fetch_logs(timeout=timeout):
            print(log.message.strip())
