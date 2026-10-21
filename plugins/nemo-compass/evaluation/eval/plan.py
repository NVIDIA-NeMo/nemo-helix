# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve workflow mode + subjects (stdlib only; runs on the bare runner).

Precedence per value: pull_request hardcodes it -> workflow_dispatch input -> default.
Branch validation (Task-6 style) re-adds a temporary push trigger by wiring dispatch-input
fallbacks in the workflow env — no vars.EVALUATION_* mechanism lives here anymore.
"""

import json
import os


def resolve_mode(event: str, input_mode: str) -> str:
    if event == "pull_request":
        return "analyze"
    return input_mode or "analyze"


def resolve_subjects(event: str, input_subjects: str) -> list[str]:
    raw = "tau2-airline" if event == "pull_request" else (input_subjects or "tau2-airline")
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    env = os.environ
    mode = resolve_mode(env.get("GITHUB_EVENT_NAME", ""), env.get("INPUT_MODE", ""))
    subjects = resolve_subjects(env.get("GITHUB_EVENT_NAME", ""), env.get("INPUT_SUBJECTS", ""))
    with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
        fh.write(f"mode={mode}\nsubjects={json.dumps(subjects)}\n")
    with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
        fh.write(f"### evaluation-insights: mode={mode} subjects={','.join(subjects)}\n")
    print(f"mode={mode} subjects={subjects}")


if __name__ == "__main__":
    main()
