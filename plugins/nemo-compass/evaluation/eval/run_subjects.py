# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the evaluation for each subject (CI produce loop: tau2 -> ingest -> analyze).

Runs from the nemo-compass plugin directory. SUBJECTS is a JSON array of subject
names. (The analyze job doesn't use this script — it calls
`uv run python -m evaluation analyze` directly.)
"""

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from time import sleep

LOCAL_URL = "http://localhost:8080"  # the in-job CI stack
PLATFORM_ROOT = Path(__file__).resolve().parents[4]


def build_overrides(num_tasks: str, num_trials: str) -> list[str]:
    overrides: list[str] = []
    if num_tasks:
        overrides += ["--set", f"num_tasks={num_tasks}"]
    if num_trials:
        overrides += ["--set", f"num_trials={num_trials}"]
    return overrides


def subjects_from_env(env: Mapping[str, str]) -> list[str]:
    return json.loads(env["SUBJECTS"])


def _evaluation(*args: str) -> None:
    subprocess.run(
        ["uv", "run", "--project", str(PLATFORM_ROOT), "python", "-m", "evaluation", *args],
        check=True,
    )


def analyze_with_retry(subject: str, summary: str, retries: int = 2, delay: float = 90.0) -> None:
    """Run `evaluation analyze` with a rate-limit cool-down retry.

    Compass shares its gateway key with the tau2 sims, so right after a
    large sim burst the key's rate window can still be hot and Compass's
    first calls can 429. A cool-down retry absorbs that transient.
    """
    for attempt in range(retries + 1):
        try:
            # --live: this is the produce loop analyzing the traces it just made
            # on the in-job stack — there is no minted state to restore yet.
            _evaluation("analyze", subject, "--live", "--base", LOCAL_URL, "--summary-md", summary)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            print(f"analyze failed (attempt {attempt + 1}/{retries + 1}); cooling down {int(delay)}s", flush=True)
            sleep(delay)


def main() -> None:
    env = os.environ
    subjects = subjects_from_env(env)
    overrides = build_overrides(env.get("NUM_TASKS", ""), env.get("NUM_TRIALS", ""))
    summary = env["GITHUB_STEP_SUMMARY"]
    for subject in subjects:
        print(f"::group::{subject}", flush=True)
        _evaluation("doctor", subject)
        _evaluation("run", subject, "--base", LOCAL_URL, *overrides)
        analyze_with_retry(subject, summary)
        print(Path(f"evaluation/tmp/insights_{subject}.yaml").read_text(encoding="utf-8"))
        print("::endgroup::", flush=True)


if __name__ == "__main__":
    main()
