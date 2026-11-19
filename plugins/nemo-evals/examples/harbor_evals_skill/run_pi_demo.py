# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish task 003 and compare Codex with and without the two NeMo Evals skills."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from codex_evidence import CODEX_VERSION, observe_codex, rollout_events
from nemo_helix_plugin.skill_archive import EXCLUDED_TOP_LEVEL_DIRS
from nhx_evals_sdk.agent_eval.persistence import read_trials
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.values.evidence import EVIDENCE_LOGS

EXAMPLE = Path(__file__).resolve().parent
REPO = EXAMPLE.parents[3]
TASK = "nemo-evals-plugin-003"
SKILLS = {
    "nemo-evals-plugin": REPO / "plugins/nemo-evals/src/nemo_evals/skills/evaluator-plugin",
    "nemo-evals": REPO / ".agents/skills/nemo-evals",
}
REWARDS = ("reward", "task_instruction", "taskset_persisted", "script_correct")
RUN_PATTERN = r"run-[0-9a-f]{16}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inputs() -> dict[str, str]:
    workspace = os.getenv("WORKSPACE", "default")
    connection = os.getenv("NEMO_BASE_URL") or os.getenv("NHX_BASE_URL") or os.getenv("NHX_CURRENT_CONTEXT", "")
    values = {
        "workspace": workspace,
        "context": os.getenv("CONTEXT", ""),
        "connection_fingerprint": hashlib.sha256(connection.encode()).hexdigest(),
        "model": os.getenv("MODEL_REF", f"{workspace}/azure-openai-gpt-6-luna"),
        "profile": os.getenv("SANDBOX_PROFILE", "harbor-standard"),
        "inference_key_secret": os.getenv("INFERENCE_KEY_SECRET", ""),
        "PI_JUDGE_PLATFORM_URL": os.getenv("PI_JUDGE_PLATFORM_URL", "").rstrip("/"),
        "PI_JUDGE_MODEL_REF": os.getenv("PI_JUDGE_MODEL_REF", ""),
    }
    url = urlsplit(values["PI_JUDGE_PLATFORM_URL"])
    require(
        url.scheme in {"http", "https"}
        and bool(url.hostname)
        and url.username is None
        and not (url.path or url.query or url.fragment),
        "Set PI_JUDGE_PLATFORM_URL to the judge deployment's API origin as seen by the sandbox.",
    )
    require(
        bool(re.fullmatch(r"[\w.-]+/[\w.-]+", values["PI_JUDGE_MODEL_REF"])),
        "Set PI_JUDGE_MODEL_REF to the judge's workspace/model ref.",
    )
    return values


def materialize_dataset(directory: Path, settings: dict[str, str]) -> Path:
    dataset = directory / "dataset"
    task = dataset / TASK
    shutil.copytree(EXAMPLE / "dataset" / TASK, task, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    manifest = task / "task.toml"
    source = manifest.read_text(encoding="utf-8")
    for key in ("PI_JUDGE_PLATFORM_URL", "PI_JUDGE_MODEL_REF"):
        placeholder = f'{key} = "${{{key}:-}}"'
        require(source.count(placeholder) == 1, f"Expected one {key} placeholder in {manifest}")
        source = source.replace(placeholder, f"{key} = {json.dumps(settings[key])}")
    verifier = tomllib.loads(source)["verifier"]
    require(verifier["environment_mode"] == "shared", "Expected the shared verifier for task 003.")
    for key in ("PI_JUDGE_PLATFORM_URL", "PI_JUDGE_MODEL_REF"):
        require(verifier["env"].get(key) == settings[key], f"Expected {key} in the verifier environment.")
    manifest.write_text(source, encoding="utf-8")
    return dataset


def request(arm: str, settings: dict[str, str], skill_refs: list[str]) -> dict[str, Any]:
    source = {"name": "oracle"}
    if arm != "oracle":
        source = {"name": "codex", "model": {"ref": settings["model"]}}
    target: dict[str, Any] = {"kind": "harbor", "source": source, "n_attempts": 1, "skills": skill_refs}
    if arm != "oracle":
        target["agent_kwargs"] = {"version": CODEX_VERSION, "disable_web_search": True}
        if secret := settings["inference_key_secret"]:
            target["env_secrets"] = {"OPENAI_API_KEY": secret}
    return {
        "idempotency_key": str(uuid4()),
        "spec": {"target": target},
        "execution": {"dispatcher_profile": "default", "sandbox": {"profile": settings["profile"]}},
    }


def check_rewards(results: dict[str, Any], arm: str) -> dict[str, float]:
    run = results["run"]
    require(run["phase"] == "completed" and run["result_state"] == "ready", f"{arm}: run results are incomplete")
    (task,) = results["tasks"]
    result = task["result"]
    summary = result["summary"]
    require(summary["trial_count"] == 1 and summary["error_count"] == 0, f"{arm}: expected one error-free trial")
    rewards = {}
    for key in REWARDS:
        matches = [score for score in result["scores"]["scores"] if score["name"] == f"harbor_reward.{key}"]
        require(len(matches) == 1, f"{arm}: expected one aggregate for {key}")
        entry = matches[0]
        require(
            entry["count"] == 1
            and entry["nan_count"] == 0
            and type(entry["mean"]) in (int, float)
            and entry["mean"] in (0, 1),
            f"{arm}: expected a single binary score for {key}",
        )
        rewards[key] = entry["mean"]
    require(rewards["reward"] == all(rewards[key] for key in REWARDS[1:]), f"{arm}: inconsistent primary reward")
    return rewards


def verify_evidence(bundle: Path, run_dir: Path, arm: str, rewards: dict[str, float]) -> dict[str, Any]:
    (trial,) = read_trials(bundle)
    require(
        trial.status == AgentEvalTrialStatus.COMPLETED and trial.error is None and trial.task_id == TASK,
        f"{arm}: expected a completed task 003 trial in the bundle",
    )
    metadata = trial.metadata
    require(
        metadata.get("harbor_primary_reward_key") == "reward"
        and metadata.get("reward") == rewards["reward"]
        and metadata.get("reward_details") == rewards,
        f"{arm}: trial and run rewards differ",
    )
    descriptor = trial.evidence.descriptors.get(EVIDENCE_LOGS) if trial.evidence else None
    if not descriptor or not descriptor.ref:
        raise ValueError(f"{arm}: missing Harbor agent logs")
    agent = Path(descriptor.ref).resolve()
    require(
        agent.name == "agent" and agent.is_relative_to(bundle.resolve()) and agent.is_dir(),
        f"{arm}: agent logs must be inside the bundle",
    )
    artifacts = agent.parent / "artifacts"
    verifier = agent.parent / "verifier"
    stdout = verifier / "test-stdout.txt"
    require(stdout.is_file() and (artifacts / "startup.log").is_file(), f"{arm}: missing verifier or startup log")
    inventory = read_json(verifier / "skill-inventory.json")
    expected = {}
    if arm == "with-skills":
        for name, source in SKILLS.items():
            source = source.resolve(strict=True)
            expected[name] = {
                path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.rglob("*")
                if path.is_file() and path.relative_to(source).parts[0] not in EXCLUDED_TOP_LEVEL_DIRS
            }
    require(inventory == {"staged": expected, "registered": expected}, f"{arm}: unexpected skill inventory")
    evidence: dict[str, Any] = {"verifier_stdout": str(stdout.relative_to(run_dir))}
    if arm == "oracle":
        require(all(rewards.values()), "Oracle must pass all four reward checks before launching Codex.")
        submission = verifier / "submission.json"
        archive = verifier / "pi-digit.tar.gz"
        require(
            hashlib.sha256(archive.read_bytes()).hexdigest() == read_json(submission)["archive_sha256"],
            "Oracle archive digest differs from the persisted submission.",
        )
        evidence.update(submission=str(submission.relative_to(run_dir)), archive=str(archive.relative_to(run_dir)))
    else:
        log = (agent.parent / "trial.log").read_text(encoding="utf-8")
        require("Codex is already available at the requested version" in log, f"{arm}: Codex install was not skipped")
        events = list(rollout_events(agent))
        observed = {}
        for name, source in SKILLS.items():
            observation = observe_codex(events, name, (source / "SKILL.md").read_text(encoding="utf-8"))
            require(
                observation.skill_listed == (name in expected), f"{arm}: unexpected launch catalog entry for {name}"
            )
            observed[name] = {"listed": observation.skill_listed, "activated": observation.activated}
        evidence["skills"] = observed
    return evidence


def run_demo(run_dir: Path, settings: dict[str, str]) -> None:
    run_dir = run_dir.resolve()
    workspace = settings["workspace"]
    nemo = Path(sys.executable).with_name("nemo")
    command = [str(nemo) if nemo.is_file() else "nemo"]
    if settings["context"]:
        command += ["--context", settings["context"]]

    def cli(*args: str, capture: bool = False, scoped: bool = True) -> str:
        invocation = [*command, *args]
        if scoped:
            invocation += ["--workspace", workspace]
        completed = subprocess.run(invocation, check=True, text=True, stdout=subprocess.PIPE if capture else None)
        return completed.stdout or ""

    settings_file = run_dir / "inputs.json"
    if settings_file.exists():
        require(read_json(settings_file) == settings, "Use the original settings or a new RUN_DIR.")
    else:
        save_json(settings_file, settings)

    taskset_file = run_dir / "taskset.json"
    if not taskset_file.exists():
        with tempfile.TemporaryDirectory(prefix="harbor-pi-source-") as scratch:
            dataset = materialize_dataset(Path(scratch), settings)
            publication = json.loads(
                cli(
                    "evals", "tasksets", "create", "--harbor-dir", str(dataset), "--tasks", TASK, "--json", capture=True
                )
            )
        save_json(taskset_file, publication)
    publication = read_json(taskset_file)
    taskset = publication["taskset_ref"]
    require(
        bool(re.fullmatch(rf"{re.escape(workspace)}/[^/#]+#[0-9a-f]{{64}}", taskset))
        and len(publication["tasks"]) == publication["total"] == 1
        and bool(re.fullmatch(rf"{re.escape(workspace)}/[^/#]+#[0-9a-f]{{64}}", publication["tasks"][0])),
        "Expected one published task in a versioned Taskset.",
    )
    cli("evals", "tasksets", "build-images", taskset, "--wait", scoped=False)

    def submit(arm: str, refs: list[str]) -> str:
        directory = run_dir / arm
        directory.mkdir(exist_ok=True)
        request_file = directory / "request.json"
        if not request_file.exists():
            save_json(request_file, request(arm, settings, refs))
        id_file = directory / "run.json"
        if not id_file.exists():
            output = cli(
                "evals", "runs", "submit", "--taskset", taskset, "--spec-file", str(request_file), capture=True
            )
            print(output, end="", flush=True)
            match = re.search(rf"^Run ({RUN_PATTERN})\b", output, re.MULTILINE)
            if match is None:
                raise ValueError(f"{arm}: no run ID in submit output")
            save_json(id_file, {"run_id": match[1]})
        return read_json(id_file)["run_id"]

    def collect(arm: str, run_id: str) -> dict[str, Any]:
        directory = run_dir / arm
        cli("evals", "runs", "status", run_id, "--watch")
        results = json.loads(cli("evals", "runs", "results", run_id, "--json", capture=True))
        save_json(directory / "results.json", results)
        (task,) = json.loads(cli("evals", "runs", "tasks", run_id, "--json", capture=True))["data"]
        job = task["job"]
        details = json.loads(cli("jobs", "get", job, "-f", "json", capture=True))["spec"]["environment"]
        image = details["agent"]["image_ref"]
        environment = {
            "agent_image_ref": image,
            "agent_preinstalled": details.get("agent_preinstalled"),
            "verifier": details.get("verifier"),
        }
        save_json(directory / "job-environment.json", environment)
        archive = directory / "agent-eval-results.tar.gz"
        bundle = directory / "agent-eval-results"
        if not bundle.exists():
            with tempfile.TemporaryDirectory(prefix=".bundle-", dir=directory) as scratch:
                unpacked = Path(scratch)
                if not archive.exists():
                    download = unpacked / archive.name
                    cli(
                        "jobs",
                        "results",
                        "download",
                        "agent-eval-results",
                        "--job",
                        job,
                        "--output-file",
                        str(download),
                    )
                    download.replace(archive)
                with tarfile.open(archive) as package:
                    members = package.getmembers()
                    require(
                        all(member.isfile() or member.isdir() for member in members), f"{arm}: unsafe bundle member"
                    )
                    package.extractall(unpacked, members=members, filter="data")
                (unpacked / bundle.name).replace(bundle)
        require(
            bool(re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image))
            and environment["agent_preinstalled"] is False
            and environment["verifier"] is None,
            f"{arm}: expected a pinned image, agent_preinstalled=false, and no separate verifier image",
        )
        rewards = check_rewards(results, arm)
        return {
            "run_id": run_id,
            "job": job,
            "agent_image_ref": image,
            "reward": rewards,
            "bundle": str(bundle.relative_to(run_dir)),
            **verify_evidence(bundle, run_dir, arm, rewards),
        }

    oracle = collect("oracle", submit("oracle", []))
    skill_file = run_dir / "skills.json"
    if not skill_file.exists():
        refs = []
        for name, source in SKILLS.items():
            skill = json.loads(
                cli(
                    "agents",
                    "skills",
                    "publish",
                    name,
                    "--from-dir",
                    str(source.resolve(strict=True)),
                    "--taskset",
                    taskset,
                    "--output-format",
                    "json",
                    capture=True,
                )
            )
            digest = skill["content_hash"]
            require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"{name}: expected a pinned revision")
            refs.append(f"{workspace}/{name}#{digest}")
        save_json(skill_file, {"skill_refs": refs})
    refs = read_json(skill_file)["skill_refs"]
    baseline_id = submit("baseline", [])
    treatment_id = submit("with-skills", refs)
    arms = {"oracle": oracle}
    errors = []
    for arm, run_id in (("baseline", baseline_id), ("with-skills", treatment_id)):
        try:
            arms[arm] = collect(arm, run_id)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            errors.append(f"{arm}: {exc}")
    require(not errors, "\n".join(errors))
    require(len({arm["agent_image_ref"] for arm in arms.values()}) == 1, "Runs used different effective image digests.")
    comparison = run_dir / "comparison.json"
    save_json(comparison, {"taskset_ref": taskset, "skill_refs": refs, "arms": arms})
    print(f"Comparison saved to {comparison}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Set PI_JUDGE_PLATFORM_URL and PI_JUDGE_MODEL_REF.")
    parser.parse_args()
    os.umask(0o077)
    try:
        settings = inputs()
        if directory := os.getenv("RUN_DIR"):
            run_dir = Path(directory).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            parent = REPO / "temp"
            parent.mkdir(exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix="harbor-pi-demo-", dir=parent))
        print(f"Comparison directory: {run_dir}", flush=True)
        with (run_dir / ".launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_demo(run_dir, settings)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
