# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test that the SDK runs Harbor's hello-world example natively.

Unlike ``test_harbor_runtime.py`` (which adapts a fabricated job dir with no
Docker), this drives a real Harbor job over the bundled ``hello_world_dataset``
using the deterministic oracle agent through the one-call ``run_harbor_eval``
entry point, then checks the scores. It needs both ``harbor`` installed and a
working Docker daemon, and is skipped otherwise.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborRuntimeConfig,
    run_harbor_eval,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.skip_in_ci]

_DATASET_DIR = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "hello_world_dataset"
_FABRIC_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "harbor" / "fabric_agent" / "fabric_hello_world_dataset"
)
_TASK_NAME = "harbor/hello-world"

# Minimal BaseAgent for the resume probe. Satisfies the bundled hello-world verifier,
# which passes iff /app/hello.txt contains exactly "Hello, world!".
_RESUME_PROBE_AGENT = """\
from harbor import BaseAgent


class WrappedAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "resume-probe"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        await environment.exec("printf 'Hello, world!' > /app/hello.txt")
"""


_ENV_PROBE_AGENT = """\
from harbor import BaseAgent


class WrappedAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "env-probe"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        await environment.exec(
            'test "$PROBE_TOKEN" = "probe-value" && printf "Hello, world!" > /app/hello.txt',
            env=self.extra_env,
        )
"""


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.asyncio
async def test_sdk_runs_harbor_hello_world_natively(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    # The whole caller side: a config and one call — no JobConfig, no harbor import.
    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, agent_name="oracle")
    result = await run_harbor_eval(config, _DATASET_DIR)

    # The oracle agent writes hello.txt, so the verifier reward is 1.0 and the
    # trial completes cleanly through the SDK's Harbor adapter.
    assert len(result.trials) == 1
    trial = result.trials[0]
    assert trial.task_id == _TASK_NAME
    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert trial.metadata["reward"] == 1.0

    rewards = {score.task_id: score.outputs[0].value for score in result.scores if score.outputs}
    assert rewards == {_TASK_NAME: 1.0}

    # Harbor really wrote a per-trial result.json under the job dir.
    trial_results = list(jobs_dir.glob("*/*/result.json"))
    assert trial_results, "Harbor did not write a per-trial result.json"
    assert json.loads(trial_results[0].read_text())["task_name"] == _TASK_NAME


@pytest.mark.asyncio
async def test_harbor_resumes_a_partial_job_with_a_custom_agent_dir(tmp_path: Path) -> None:
    """Regression for AALGO-430 — a real Harbor resume with ``agent_dir`` set.

    This is the case every faked-``Job`` test misses, and the reason the bug went
    unnoticed: the scoped agent import path used to carry a fresh uuid per run, so
    Harbor's ``JobConfig`` comparison failed on the second call and it raised
    ``FileExistsError`` rather than resuming. Now the path is content-addressed, so
    an unchanged agent resumes and only the missing trial is re-run.

    Runs two attempts and drops one, so "resumed" is distinguishable from "discarded
    and re-run from scratch" — with a single attempt the two are observationally
    identical and the test would pass either way.
    """
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    # A loose wrapper file next to the dataset — the shape `agent_dir` exists for,
    # and the shape the Experimentalist always uses.
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "harbor_wrapper.py").write_text(_RESUME_PROBE_AGENT, encoding="utf-8")

    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name="resume-job",  # pinned: the cache and Harbor's resume both need it
        agent_import_path="harbor_wrapper:WrappedAgent",
        agent_dir=agent_dir,
        # Two attempts so one can be dropped and one kept. With a single attempt,
        # discarding the whole job dir and re-running is observationally identical to
        # resuming, and the assertions below could not tell them apart.
        n_attempts=2,
    )

    first = await run_harbor_eval(config, _DATASET_DIR)
    assert [trial.status for trial in first.trials] == [AgentEvalTrialStatus.COMPLETED] * 2

    job_dir = jobs_dir / "resume-job"
    trial_dirs = sorted(path.parent for path in job_dir.glob("*/result.json"))
    assert len(trial_dirs) == 2, "the first run must have written both attempts"
    survivor, dropped = trial_dirs
    survivor_result = (survivor / "result.json").read_text(encoding="utf-8")

    # Drop one attempt's result so the job is under-covered, leaving the job dir (and
    # its config.json) in place — the exact state that used to raise FileExistsError.
    (dropped / "result.json").unlink()

    second = await run_harbor_eval(config, _DATASET_DIR)

    # The point of the test: the completed attempt was *resumed*, not re-run. Harbor
    # suffixes each trial dir with a shortuuid, so a discarded job dir would come back
    # under a different name, and a re-executed trial would rewrite result.json.
    assert survivor.is_dir(), "the completed attempt's trial dir must survive the rerun"
    assert (survivor / "result.json").read_text(encoding="utf-8") == survivor_result, (
        "the completed attempt must be reused untouched, not re-executed"
    )
    assert not dropped.is_dir(), "the result-less attempt must be cleared and re-run"

    assert [trial.status for trial in second.trials] == [AgentEvalTrialStatus.COMPLETED] * 2
    assert {trial.task_id for trial in second.trials} == {_TASK_NAME}
    assert [trial.metadata["reward"] for trial in second.trials] == [1.0, 1.0]


@pytest.mark.asyncio
async def test_agent_env_from_host_reach_the_agent_and_persist_as_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host variable named in ``agent_env_from_host`` reaches the agent, and only its name reaches disk.

    Harbor resolves the ``${NAME}`` template when it constructs the agent and serializes the template
    back into the job dir's ``config.json``. If either half broke, the platform's ``env_secrets`` route
    would silently run the agent without its credential or persist that credential in plaintext.
    """
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    monkeypatch.setenv("PROBE_TOKEN", "probe-value")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "harbor_wrapper.py").write_text(_ENV_PROBE_AGENT, encoding="utf-8")
    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name="env-probe",
        agent_import_path="harbor_wrapper:WrappedAgent",
        agent_dir=agent_dir,
        agent_env_from_host=["PROBE_TOKEN"],
    )

    result = await run_harbor_eval(config, _DATASET_DIR)

    assert [trial.status for trial in result.trials] == [AgentEvalTrialStatus.COMPLETED]
    assert [(score.metric_type, score.outputs[0].value) for score in result.scores] == [("harbor_reward", 1.0)]
    persisted = json.loads((jobs_dir / "env-probe" / "config.json").read_text(encoding="utf-8"))
    assert persisted["agents"][0]["env"] == {"PROBE_TOKEN": "${PROBE_TOKEN}"}
    on_disk = [path for path in (jobs_dir / "env-probe").rglob("*") if path.is_file()]
    assert not any("probe-value" in path.read_text(encoding="utf-8", errors="ignore") for path in on_disk)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="needs NVIDIA_API_KEY for build.nvidia.com")
async def test_nemo_fabric_agent_runs_deepagents_on_nemotron_inside_harbor(tmp_path: Path) -> None:
    """The documented Fabric-inside-Harbor recipe reaches a passing reward with the key kept off disk.

    Covers the whole bridge at once: `NemoFabricAgent` resolves the `nvidia` provider's credential
    variable and endpoint, Fabric installs the deepagents harness in the task container, the model
    writes the file the verifier checks, and Harbor persists only the `${NVIDIA_API_KEY}` template.
    """
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name="fabric-deepagents",
        agent_import_path="nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent",
        agent_kwargs={
            "fabric_adapter_id": "nvidia.fabric.langchain.deepagents",
            "fabric_package": "nemo-fabric[deepagents]==0.3.0b1",
            "fabric_workspace": "/app",
        },
        agent_model_name=os.environ.get("NEMO_FABRIC_TEST_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        agent_env_from_host=["NVIDIA_API_KEY"],
        agent_setup_timeout_multiplier=8.0,
        agent_timeout_multiplier=5.0,
    )

    result = await run_harbor_eval(config, _FABRIC_DATASET_DIR)

    assert [trial.status for trial in result.trials] == [AgentEvalTrialStatus.COMPLETED]
    assert [(score.metric_type, score.outputs[0].value) for score in result.scores] == [("harbor_reward", 1.0)]
    persisted = json.loads((jobs_dir / "fabric-deepagents" / "config.json").read_text(encoding="utf-8"))
    assert persisted["agents"][0]["env"] == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
    key = os.environ["NVIDIA_API_KEY"]
    on_disk = [path for path in (jobs_dir / "fabric-deepagents").rglob("*") if path.is_file()]
    assert not any(key in path.read_text(encoding="utf-8", errors="ignore") for path in on_disk)
