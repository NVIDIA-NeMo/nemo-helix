# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills injected into a native Harbor run: validation, AgentConfig, and cache identity."""

import json
import shutil
from pathlib import Path

import pytest
from harbor_fixtures import write_harbor_trial_result
from nhx_evals_sdk.agent_eval.runtimes.harbor import runtime as harbor_runtime
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import (
    CACHE_STAMP_FILENAME,
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    _build_native_job,
    _cache_is_stale,
    _cache_stamp,
    _write_cache_stamp,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.skills import skill_tree_digest
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.metrics.runner_rewards import HarborRewardMetric
from pydantic import ValidationError

_HELLO_WORLD_DATASET = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "hello_world_dataset"


def _skill(parent: Path, name: str, body: str = "# skill\n") -> Path:
    path = parent / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(body, encoding="utf-8")
    return path


# --- validation ---------------------------------------------------------------------------------


def test_unset_skills_leave_today_behaviour(tmp_path: Path) -> None:
    config = HarborRuntimeConfig(jobs_dir=tmp_path, agent_kwargs={"skills_dir": "/opt/skills"})
    assert config.skills is None


def test_relative_skill_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must be absolute"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=[Path("org/name")])


def test_missing_skill_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="existing directory"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=[tmp_path / "absent"])


def test_symlinked_skill_dir_is_rejected(tmp_path: Path) -> None:
    real = _skill(tmp_path / "real", "writer")
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValidationError, match="not a symlink"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=[link])


@pytest.mark.parametrize("shape", ["missing", "directory", "symlink"])
def test_skill_without_a_regular_root_skill_md_is_rejected(tmp_path: Path, shape: str) -> None:
    skill = tmp_path / "writer"
    skill.mkdir()
    if shape == "directory":
        (skill / "SKILL.md").mkdir()
    elif shape == "symlink":
        target = tmp_path / "elsewhere.md"
        target.write_text("# skill\n", encoding="utf-8")
        (skill / "SKILL.md").symlink_to(target)
    with pytest.raises(ValidationError, match="no regular SKILL.md"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=[skill])


def test_duplicate_skill_names_are_rejected(tmp_path: Path) -> None:
    first = _skill(tmp_path / "v1", "writer")
    second = _skill(tmp_path / "v2", "writer")
    with pytest.raises(ValidationError, match="duplicate skill name 'writer'"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=[first, second])


@pytest.mark.parametrize("skills", ["empty", "one"])
def test_agent_skills_dir_kwarg_is_rejected_with_managed_skills(tmp_path: Path, skills: str) -> None:
    managed = [] if skills == "empty" else [_skill(tmp_path, "writer")]
    with pytest.raises(ValidationError, match="skills_dir"):
        HarborRuntimeConfig(jobs_dir=tmp_path, skills=managed, agent_kwargs={"skills_dir": "/opt/skills"})


@pytest.mark.asyncio
async def test_run_rechecks_skills_set_after_validation(tmp_path: Path) -> None:
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", job_name="j").model_copy(
        update={"skills": [Path("relative")]}
    )
    with pytest.raises(ValueError, match="must be absolute"):
        await HarborAgentTaskRunner(config=config).run_tasks([], None)


# --- digest -------------------------------------------------------------------------------------


def test_digest_covers_bytes_exec_bit_and_hidden_files_but_not_location(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "a", "writer")
    (skill / "scripts").mkdir()
    script = skill / "scripts" / "run.sh"
    script.write_text("echo hi\n", encoding="utf-8")
    baseline = skill_tree_digest(skill)

    assert skill_tree_digest(shutil.copytree(skill, tmp_path / "b" / "writer")) == baseline
    script.chmod(script.stat().st_mode | 0o100)
    executable = skill_tree_digest(skill)
    assert executable != baseline
    (skill / ".hidden").write_text("x", encoding="utf-8")
    assert skill_tree_digest(skill) != executable
    assert len(baseline) == 64 and baseline == baseline.lower()


def test_digest_rejects_symlinks_inside_a_skill(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "writer")
    (skill / "link.md").symlink_to(skill / "SKILL.md")
    with pytest.raises(ValueError, match="contains a symlink: link.md"):
        skill_tree_digest(skill)


# --- AgentConfig and task skills_dir ------------------------------------------------------------


def _dataset(tmp_path: Path, *, task_skills_dir: str | None = None) -> Path:
    dataset = shutil.copytree(_HELLO_WORLD_DATASET, tmp_path / "dataset")
    task = dataset / "hello-world"
    if task_skills_dir is not None:
        toml = (task / "task.toml").read_text(encoding="utf-8")
        (task / "task.toml").write_text(
            toml.replace("[environment]\n", f'[environment]\nskills_dir = "{task_skills_dir}"\n'), encoding="utf-8"
        )
    return dataset


class _RecordingJob:
    created: list[object] = []

    @classmethod
    async def create(cls, job_config: object) -> "_RecordingJob":
        cls.created.append(job_config)
        return cls()

    async def run(self) -> None:
        return None

    def add_hook(self, *_args: object, **_kwargs: object) -> None:
        return None


@pytest.fixture
def recorded_jobs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    created: list[object] = []
    monkeypatch.setattr(_RecordingJob, "created", created)
    monkeypatch.setattr(harbor_runtime, "_create_harbor_job", _RecordingJob.create)
    monkeypatch.setattr(harbor_runtime.SetupSlots, "attach", lambda self, job: None)
    return created


@pytest.mark.asyncio
@pytest.mark.parametrize("skills", ["unset", "empty", "two"])
async def test_skills_reach_harbor_agent_config(tmp_path: Path, recorded_jobs: list[object], skills: str) -> None:
    managed = {
        "unset": None,
        "empty": [],
        "two": [_skill(tmp_path / "s", "writer"), _skill(tmp_path / "s", "reviewer")],
    }[skills]
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", skills=managed, agent_kwargs={"mode": "x"})
    _, run_job = _build_native_job(config, _dataset(tmp_path), None, job_name="j", env_templates={})
    await run_job()

    (job_config,) = recorded_jobs
    (agent_config,) = job_config.agents  # ty: ignore[unresolved-attribute]
    assert agent_config.skills == ([] if managed is None else [str(path) for path in managed])
    assert agent_config.kwargs == {"mode": "x"}


@pytest.mark.asyncio
@pytest.mark.parametrize("skills", ["empty", "one"])
async def test_task_skills_dir_is_rejected_before_the_job_dir_is_discarded(
    tmp_path: Path, recorded_jobs: list[object], skills: str
) -> None:
    managed = [] if skills == "empty" else [_skill(tmp_path / "s", "writer")]
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", skills=managed, force_rerun=True)
    job_dir, run_job = _build_native_job(
        config, _dataset(tmp_path, task_skills_dir="/skills"), None, job_name="j", env_templates={}
    )
    job_dir.mkdir(parents=True)
    (job_dir / "keep").write_text("completed work", encoding="utf-8")

    with pytest.raises(ValueError, match="'hello-world' sets environment.skills_dir"):
        await run_job()
    assert (job_dir / "keep").exists()
    assert recorded_jobs == []


@pytest.mark.asyncio
async def test_task_skills_dir_is_allowed_when_skills_are_unmanaged(
    tmp_path: Path, recorded_jobs: list[object]
) -> None:
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs")
    _, run_job = _build_native_job(
        config, _dataset(tmp_path, task_skills_dir="/skills"), None, job_name="j", env_templates={}
    )
    await run_job()
    assert len(recorded_jobs) == 1


# --- cache identity ------------------------------------------------------


def _seed(tmp_path: Path, skills: list[Path] | None) -> tuple[HarborRuntimeConfig, Path, AgentEvalTask]:
    """A complete, stamped job dir for ``skills``, as a finished run would leave it."""
    dataset = tmp_path / "dataset"
    task_dir = dataset / "t"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[task]\nname = "t"\n', encoding="utf-8")
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "cached"
    write_harbor_trial_result(job_dir / "t__aaa", task_name="t", rewards={"reward": 1.0})
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="cached", skills=skills)
    task = AgentEvalTask(
        id="t",
        intent="x",
        inputs={"instruction": "x"},
        metrics=[HarborRewardMetric()],
        metadata={"harbor_dataset_path": str(dataset), "harbor_task_dir": str(task_dir)},
    )
    _write_cache_stamp(job_dir, _cache_stamp(config, dataset, [task]))
    return config, job_dir, task


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record each attempted Harbor run and its force_rerun, without Harbor."""
    calls: list[bool] = []

    def fake(config, _dataset_path, _task_names, *, job_name=None, force_rerun=None, env_templates, job_scopes=()):
        async def run_job() -> None:
            calls.append(bool(force_rerun))

        return config.jobs_dir / (job_name or "job"), run_job

    monkeypatch.setattr(harbor_runtime, "_build_native_job", fake)
    return calls


def _stale(config: HarborRuntimeConfig, job_dir: Path, task: AgentEvalTask) -> bool:
    return _cache_is_stale(job_dir, _cache_stamp(config, Path(str(task.metadata["harbor_dataset_path"])), [task]))


@pytest.mark.asyncio
async def test_identical_skills_hit_the_cache(tmp_path: Path, run_calls: list[bool]) -> None:
    config, _job_dir, task = _seed(tmp_path, [_skill(tmp_path / "s", "writer")])
    await HarborAgentTaskRunner(config=config).run_tasks([task])
    assert run_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["bytes", "exec_bit", "added_file"])
async def test_changed_skill_contents_miss_the_cache(tmp_path: Path, run_calls: list[bool], change: str) -> None:
    skill = _skill(tmp_path / "s", "writer")
    script = skill / "run.sh"
    script.write_text("echo hi\n", encoding="utf-8")
    config, _job_dir, task = _seed(tmp_path, [skill])
    if change == "bytes":
        (skill / "SKILL.md").write_text("# skill v2\n", encoding="utf-8")
    elif change == "exec_bit":
        script.chmod(script.stat().st_mode | 0o100)
    else:
        (skill / "notes.md").write_text("more\n", encoding="utf-8")

    await HarborAgentTaskRunner(config=config).run_tasks([task])
    assert run_calls == [True], "results from other skill contents must be discarded, not resumed"


def test_relocated_identical_skill_still_hits(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "s", "writer")
    config, job_dir, task = _seed(tmp_path, [skill])
    moved = shutil.copytree(skill, tmp_path / "elsewhere" / "writer")
    assert _stale(config.model_copy(update={"skills": [moved]}), job_dir, task) is False


def test_renamed_skill_misses(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "s", "writer")
    config, job_dir, task = _seed(tmp_path, [skill])
    renamed = shutil.copytree(skill, tmp_path / "elsewhere" / "editor")
    assert _stale(config.model_copy(update={"skills": [renamed]}), job_dir, task) is True


@pytest.mark.parametrize(("seeded", "requested"), [("unset", "empty"), ("empty", "unset"), ("unset", "one")])
def test_managed_and_unmanaged_runs_do_not_share_a_cache(tmp_path: Path, seeded: str, requested: str) -> None:
    lists: dict[str, list[Path] | None] = {"unset": None, "empty": [], "one": [_skill(tmp_path / "s", "writer")]}
    config, job_dir, task = _seed(tmp_path, lists[seeded])
    assert _stale(config.model_copy(update={"skills": lists[requested]}), job_dir, task) is True


def test_stamp_written_before_skills_still_hits_an_unmanaged_run(tmp_path: Path) -> None:
    config, job_dir, task = _seed(tmp_path, None)
    stamp_path = job_dir / CACHE_STAMP_FILENAME
    stored = json.loads(stamp_path.read_text(encoding="utf-8"))
    del stored["skills"]
    stamp_path.write_text(json.dumps(stored), encoding="utf-8")
    assert _stale(config, job_dir, task) is False


def test_raw_skill_paths_stay_out_of_the_stamp(tmp_path: Path) -> None:
    skill = _skill(tmp_path / "s", "writer")
    _config, job_dir, _task = _seed(tmp_path, [skill])
    stamp = (job_dir / CACHE_STAMP_FILENAME).read_text(encoding="utf-8")
    assert str(skill) not in stamp
    assert json.loads(stamp)["skills"] == [{"name": "writer", "digest": skill_tree_digest(skill)}]
