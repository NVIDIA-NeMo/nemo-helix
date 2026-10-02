# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from nemo_data_designer_plugin.jobs.task_results import ANALYSIS_RESULT_NAME, ARTIFACTS_RESULT_NAME
from nemo_helix_plugin.job_results import JobResults, ResultRef

ANALYSIS_FILENAME = "analysis.json"


class DataDesignerResultManager:
    def __init__(self, results: JobResults, artifacts_path: Path | str):
        self._results = results
        self._artifacts_path = Path(artifacts_path)

    def save_artifacts(self) -> ResultRef:
        return self._results.save(
            name=ARTIFACTS_RESULT_NAME,
            local_path=self._artifacts_path,
        )

    def save_analysis(self, profile: DatasetProfilerResults) -> ResultRef:
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_path = Path(tmpdir) / "analysis.json"
            analysis_path.write_text(profile.model_dump_json(indent=4), encoding="utf-8")
            return self._results.save(
                name=ANALYSIS_RESULT_NAME,
                local_path=analysis_path,
            )
