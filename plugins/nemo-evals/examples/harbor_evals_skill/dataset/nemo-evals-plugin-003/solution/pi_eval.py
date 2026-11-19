#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the published Pi Harbor task with Codex through the Inference Gateway."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

PIN = re.compile(r"([\w.-]+)/[\w.-]+#[0-9a-f]{64}")
RUN_LINE = re.compile(r"^Run (run-[0-9a-f]{16}) (?:accepted:|was already submitted\b)", re.MULTILINE)
BUILD_TIMEOUT = 20 * 60
RUN_TIMEOUT = 20 * 60
REQUEST_TIMEOUT = 60
POLL_SECONDS = 5


def nemo_output(command: list[str], *, timeout: float) -> str:
    # The CLI's URL environment overrides take precedence even with an explicit context.
    env = None
    if command[1] == "--context":
        env = {key: value for key, value in os.environ.items() if key not in {"NHX_BASE_URL", "NEMO_BASE_URL"}}
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True, timeout=timeout, env=env).stdout


def matching_run(response: dict, *, run_id: str, taskset_ref: str) -> dict:
    run = response.get("run") if isinstance(response, dict) else None
    if not isinstance(run, dict) or run.get("run_id") != run_id or run.get("taskset_ref") != taskset_ref:
        raise ValueError(f"expected run {run_id} for {taskset_ref}")
    return run


def measured_reward(response: dict, *, run_id: str, taskset_ref: str) -> int:
    run = matching_run(response, run_id=run_id, taskset_ref=taskset_ref)
    if run.get("phase") != "completed" or run.get("result_state") != "ready":
        raise ValueError(f"{run_id} has no finalized result for a completed run")
    try:
        (task,) = response["tasks"]
        if task.get("state", task.get("status")) not in {"succeeded", "completed"}:
            raise ValueError("task Job did not complete")
        result = task["result"]
        summary = result["summary"]
        if summary["trial_count"] != 1 or summary["error_count"] != 0:
            raise ValueError("expected one trial with zero errors")
        (reward,) = (score for score in result["scores"]["scores"] if score["name"] == "harbor_reward.reward")
        mean = reward["mean"]
        if reward["count"] != 1 or reward["nan_count"] != 0 or type(mean) not in (float, int) or mean not in (0, 1):
            raise ValueError("expected one measured binary Harbor reward")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{run_id} has no single error-free scored trial: {exc}") from exc
    return int(mean)


def wait_for_result(nemo: list[str], *, run_id: str, taskset_ref: str, workspace: str) -> int:
    lookup = [run_id, "--workspace", workspace, "--json"]
    deadline = time.monotonic() + RUN_TIMEOUT
    while (remaining := deadline - time.monotonic()) > 0:
        response = json.loads(nemo_output([*nemo, "runs", "status", *lookup], timeout=min(REQUEST_TIMEOUT, remaining)))
        run = matching_run(response, run_id=run_id, taskset_ref=taskset_ref)
        phase, state = run.get("phase"), run.get("result_state")
        if phase == "completed" and state == "ready":
            results = json.loads(nemo_output([*nemo, "runs", "results", *lookup], timeout=REQUEST_TIMEOUT))
            return measured_reward(results, run_id=run_id, taskset_ref=taskset_ref)
        if phase in ("failed", "cancelled", "expired") or state in ("failed", "none"):
            raise ValueError(f"{run_id} ended with phase {phase} and result state {state}")
        time.sleep(min(POLL_SECONDS, max(deadline - time.monotonic(), 0)))
    raise ValueError(f"timed out waiting for {run_id} to finish and finalize its result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taskset", required=True, help="Pinned Taskset ref (workspace/name#digest) on the run platform"
    )
    parser.add_argument(
        "--model-ref", required=True, help="Closed-book Inference Gateway VirtualModel (workspace/name)"
    )
    parser.add_argument("--context", help="NeMo CLI context. If omitted, NHX_BASE_URL can override the configured URL.")
    parser.add_argument("--sandbox-profile", default="harbor-standard")
    parser.add_argument("--inference-key-secret", help="Secret reference for OPENAI_API_KEY (workspace/name)")
    parser.add_argument("--idempotency-key", help="Reuse the same key to resume an identical submission")
    args = parser.parse_args(argv)
    taskset = PIN.fullmatch(args.taskset)
    if taskset is None:
        parser.error("--taskset must be a qualified content digest ref (workspace/name#<64 hex digits>)")
    workspace = taskset.group(1)
    nemo = ["nemo", *(["--context", args.context] if args.context else []), "evals"]
    target = {
        "kind": "harbor",
        "source": {"name": "codex", "model": {"ref": args.model_ref}},
        "agent_kwargs": {"version": "0.162.1", "disable_web_search": True},
        "closed_book": True,
        "n_attempts": 1,
    }
    if args.inference_key_secret:
        target["env_secrets"] = {"OPENAI_API_KEY": args.inference_key_secret}
    key = args.idempotency_key or str(uuid.uuid4())
    request = {
        "spec": {"tasks": args.taskset, "target": target},
        "execution": {"sandbox": {"profile": args.sandbox_profile}},
        "idempotency_key": key,
    }
    try:
        nemo_output([*nemo, "tasksets", "build-images", args.taskset, "--wait"], timeout=BUILD_TIMEOUT)
        with tempfile.TemporaryDirectory(prefix="pi-eval-") as scratch:
            spec = Path(scratch) / "request.json"
            spec.write_text(json.dumps(request), encoding="utf-8")
            try:
                output = nemo_output(
                    [
                        *nemo,
                        "runs",
                        "submit",
                        "--taskset",
                        args.taskset,
                        "--workspace",
                        workspace,
                        "--spec-file",
                        str(spec),
                    ],
                    timeout=REQUEST_TIMEOUT,
                )
                match = RUN_LINE.search(output)
                if match is None:
                    raise ValueError("runs submit did not return a run ID")
            except (ValueError, subprocess.SubprocessError) as exc:
                raise ValueError(
                    f"submission did not confirm a run ({exc}). Retry with --idempotency-key {key}"
                ) from exc
        run_id = match.group(1)
        reward = wait_for_result(nemo, run_id=run_id, taskset_ref=args.taskset, workspace=workspace)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"pi_eval: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": run_id, "taskset_ref": args.taskset, "reward": reward}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
