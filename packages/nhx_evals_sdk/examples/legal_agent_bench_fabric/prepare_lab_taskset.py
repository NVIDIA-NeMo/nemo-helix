# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a native AgentEval taskset for Harvey Labs' Legal Agent Benchmark (LAB).

Self-contained: downloads the pinned public LAB source (verifying its SHA-256), reads each raw task
(`tasks/**/task.json` = title + instructions + rubric criteria, plus a `documents/` dir), and turns it
into a typed `AgentEvalTask`:

* `id`        = flattened LAB task id
* `intent`    = task title (grader-only; never shown to the agent)
* `inputs`    = instruction (LAB instructions + skill manuals) + `files` seeded into the workspace:
                the input **documents** under `documents/` and LAB's three **skills** under `skills/<name>/`
* `reference` = `{"criteria": [...], "task_title": ...}` (grader-only)
* `metrics`   = `[LabRubricMetric(...)]` — scores with LAB's own `evaluation/score_rubric`

**Skills are delivered as task inputs, not Fabric skill injection.** LAB provides all three skills
(docx/pptx/xlsx) to the agent on every task, and each skill is just a directory of a `SKILL.md` manual
+ `scripts/`. So we seed the skill directories into the workspace and prepend their manuals to the
instruction — exactly what LAB does (mount the skills dir + concatenate the manuals). This mirrors LAB
faithfully and sidesteps Fabric's one-skill-per-runtime / container-no-skills limitations. The scripts
rely on standard tools (pandoc, python-docx, docxtpl, libreoffice, python-redlines) which must exist in
the agent's environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask, AgentEvalTaskset
from nhx_evals_sdk.agent_eval.workspace_seeds import SEED_FILES_INPUT_KEY

from .lab_rubric_metric import LabRubricMetric, OpenAICompatibleJudge

LAB_REVISION = "f46ef86e4788545622db25dcffa3aebb7a139929"
LAB_ARCHIVE_URL = f"https://codeload.github.com/harveyai/harvey-labs/tar.gz/{LAB_REVISION}"
LAB_ARCHIVE_SHA256 = "e45cbdf3236b22866e034bcc62fb23bf00ef2f2e49db7a0cd8a4b07dbae9212c"
LAB_ARCHIVE_ROOT = f"harvey-labs-{LAB_REVISION}"
EXPECTED_TASK_COUNT = 1_749
SKILLS = ("docx", "pptx", "xlsx")

# Workspace layout the agent sees (all relative to the per-task workspace root).
DOCUMENTS_SUBDIR = "documents"
SKILLS_SUBDIR = "skills"
OUTPUT_SUBDIR = "output"


# --- Download + extract the pinned source ------------------------------------------------------
def ensure_lab_source(dest: str | Path, *, allow_download: bool = True) -> Path:
    dest = Path(dest).expanduser().resolve()
    source_root = dest / LAB_ARCHIVE_ROOT
    if (source_root / "tasks").is_dir():
        return source_root
    if not allow_download:
        raise FileNotFoundError(f"LAB source not found under {dest} and downloads are disabled")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".lab-src-") as tmp:
        archive = Path(tmp) / "lab.tar.gz"
        _download(LAB_ARCHIVE_URL, archive, LAB_ARCHIVE_SHA256)
        _safe_extract(archive, dest)
    if not (source_root / "tasks").is_dir():
        raise RuntimeError(f"extracted archive missing expected root {LAB_ARCHIVE_ROOT}/tasks")
    return source_root


def _download(url: str, out: Path, sha256: str) -> None:
    for attempt in range(1, 4):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=120) as response, out.open("wb") as handle:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError(f"LAB archive checksum mismatch: expected {sha256}, got {digest.hexdigest()}")
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient download/verify error
            out.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"  download attempt {attempt}/3 failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(2 ** (attempt - 1))


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0] != LAB_ARCHIVE_ROOT):
                raise ValueError(f"unsafe archive entry: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported archive entry: {member.name}")
        tar.extractall(dest, filter="data")


# --- Read LAB's raw tasks ----------------------------------------------------------------------
def flatten_task_id(source_id: str) -> str:
    parts = PurePosixPath(source_id).parts
    if len(parts) < 2 or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"unexpected LAB task id: {source_id!r}")
    return "__".join(parts)


def iter_source_tasks(source_root: Path) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    tasks_root = source_root / "tasks"
    for task_json in sorted(tasks_root.rglob("task.json")):
        task_dir = task_json.parent
        source_id = task_dir.relative_to(tasks_root).as_posix()
        config = json.loads(task_json.read_text(encoding="utf-8"))
        if not all(config.get(k) for k in ("title", "instructions", "criteria")):
            raise ValueError(f"LAB task {source_id} missing title/instructions/criteria")
        if not (task_dir / "documents").is_dir():
            raise ValueError(f"LAB task {source_id} has no documents/ directory")
        yield source_id, task_dir, config


def _dir_seeds(directory: Path, *, prefix: str) -> dict[str, dict[str, str]]:
    """Path-seeds for every file under ``directory`` (binary-safe), staged under ``prefix/``."""
    root = directory.resolve()
    seeds: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = (PurePosixPath(prefix) / path.relative_to(root).as_posix()).as_posix()
            seeds[rel] = {"kind": "path", "path": str(path)}
    return seeds


def _skill_seeds(source_root: Path) -> dict[str, dict[str, str]]:
    """Seed LAB's docx/pptx/xlsx skill bundles into the workspace under skills/<name>/."""
    skills_root = source_root / "harness" / "skills"
    seeds: dict[str, dict[str, str]] = {}
    for skill in SKILLS:
        src = skills_root / skill
        if not (src / "SKILL.md").is_file():
            raise FileNotFoundError(f"LAB skill {skill} is missing SKILL.md at {src}")
        seeds.update(_dir_seeds(src, prefix=f"{SKILLS_SUBDIR}/{skill}"))
    return seeds


def _skill_manuals(source_root: Path) -> str:
    """Concatenate the SKILL.md manuals (LAB prepends these to the agent's system prompt)."""
    skills_root = source_root / "harness" / "skills"
    sections = []
    for skill in SKILLS:
        manual = (skills_root / skill / "SKILL.md").read_text(encoding="utf-8")
        sections.append(f"\n\n## Skill: {skill}\n\n{manual}")
    return "".join(sections)


def _instruction(config: dict[str, Any], manuals: str) -> str:
    return (
        f"# {config['title']}\n\n{config['instructions']}\n\n"
        "## Working directory\n"
        f"- Input documents are under `{DOCUMENTS_SUBDIR}/`.\n"
        f"- Document skills are under `{SKILLS_SUBDIR}/<name>/` (docx, pptx, xlsx); each has a `scripts/` "
        "directory you can invoke via bash. Their manuals are included below.\n"
        f"- Write your final deliverables as new files under `{OUTPUT_SUBDIR}/`.\n"
        f"{manuals}"
    )


# --- Load LAB's own scorer + judge (caller-side LAB coupling; keeps LabRubricMetric portable) ---------
def load_lab_score_rubric(source_root: str | Path) -> Callable[..., Any]:
    """Import LAB's own ``score_rubric`` from the downloaded source, adding the source root to sys.path.

    LAB's ``evaluation/`` is materialized by :func:`ensure_lab_source`. Returning the callable here (rather
    than importing inside the metric) is what keeps ``LabRubricMetric`` free of filesystem/sys.path coupling.
    """
    source_root = Path(source_root)
    eval_dir = source_root / "evaluation"
    if not (eval_dir / "scoring.py").is_file():
        raise RuntimeError(
            f"LAB's evaluation module was not found at {eval_dir}. It is LAB's own code, downloaded by "
            "ensure_lab_source (not committed) — run the prep or the run script first."
        )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from evaluation.scoring import score_rubric  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - depends on LAB's deps being installed
        raise RuntimeError(
            f"found {eval_dir} but could not import it — LAB's scoring deps must be installed in this "
            "process: pandoc, libreoffice, python-docx, python-redlines, pandas, openpyxl, pdfplumber, "
            "markitdown, anthropic, and the judge provider SDK."
        ) from exc
    return score_rubric


def load_lab_judge_prompts(source_root: str | Path) -> dict[str, str]:
    """Load LAB's rubric prompt templates (``evaluation/prompts/<name>.txt``) as ``{name: template}``."""
    prompts_dir = Path(source_root) / "evaluation" / "prompts"
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(prompts_dir.glob("*.txt"))}


def build_lab_judge(
    source_root: str | Path,
    *,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
    min_interval_s: float = 2.0,
) -> Any:
    """Build the judge LAB's ``score_rubric`` needs (``.evaluate_from_file(name, variables)``).

    With ``base_url`` set, use the OpenAI-compatible adapter over LAB's exact prompts (NVIDIA etc.);
    otherwise fall back to LAB's native prefix-routed ``Judge`` (imported from the LAB source).
    ``min_interval_s`` throttles judge calls — keep it ~2s for build.nvidia.com's rate-limited public
    endpoints; lower it (e.g. 0.3) for higher-limit endpoints like inference-api.
    """
    if base_url:
        api_key = os.environ.get(api_key_env) if api_key_env else None
        return OpenAICompatibleJudge(
            prompts=load_lab_judge_prompts(source_root),
            model=model,
            base_url=base_url,
            api_key=api_key,
            min_interval_s=min_interval_s,
        )
    load_lab_score_rubric(source_root)  # ensure the LAB source root is on sys.path for the import below
    from evaluation.judge import Judge  # ty: ignore[unresolved-import]

    return Judge(model=model)


# --- Build native tasks ------------------------------------------------------------------------
def build_lab_tasks(
    source_root: Path,
    *,
    judge_model: str,
    judge_base_url: str | None = None,
    judge_api_key_env: str | None = None,
    limit: int | None = None,
    task_ids: set[str] | None = None,
    judge_parallel: int = 1,
    judge_min_interval: float = 2.0,
) -> list[AgentEvalTask]:
    """Turn LAB's raw tasks into typed AgentEvalTasks, each scored by LAB's own rubric via LabRubricMetric.

    ``task_ids`` (flattened ids) selects a specific subset — e.g. one task per practice area for a
    heterogeneous run, or the exact set of trials to re-score; ``limit`` still caps the count.
    """
    # The loop below caps *after* appending, so limit<=0 would otherwise yield one task. Settle it here,
    # before the scorer/judge get built (which downloads nothing but does construct a judge client).
    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if limit == 0:
            return []
    # Load LAB's scorer + build the judge ONCE here (caller owns the LAB-source coupling); the metric is
    # handed the ready callable + judge and stays portable. One shared judge => a single global throttle.
    score_rubric = load_lab_score_rubric(source_root)
    judge = build_lab_judge(
        source_root,
        model=judge_model,
        base_url=judge_base_url,
        api_key_env=judge_api_key_env,
        min_interval_s=judge_min_interval,
    )
    skill_seeds = _skill_seeds(source_root)  # same for every task
    manuals = _skill_manuals(source_root)
    tasks: list[AgentEvalTask] = []
    for source_id, task_dir, config in iter_source_tasks(source_root):
        if task_ids is not None and flatten_task_id(source_id) not in task_ids:
            continue
        files = {**_dir_seeds(task_dir / "documents", prefix=DOCUMENTS_SUBDIR), **skill_seeds}
        metric = LabRubricMetric(
            score_rubric=score_rubric,
            judge=judge,
            output_subdir=OUTPUT_SUBDIR,
            parallel=judge_parallel,
        )
        tasks.append(
            AgentEvalTask(
                id=flatten_task_id(source_id),
                intent=str(config["title"]),
                inputs={"instruction": _instruction(config, manuals), SEED_FILES_INPUT_KEY: files},
                reference={"criteria": config["criteria"], "task_title": config["title"], "lab_task_id": source_id},
                metrics=[metric],
                metadata={"lab_task_id": source_id, "lab_source_revision": LAB_REVISION},
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


class LabTasksetLoader:
    """AgentEvalTasksetLoader: 'LAB source dir in -> native taskset out'."""

    def __init__(self, *, judge_model: str, source_root: str | Path) -> None:
        self._judge_model = judge_model
        self._source_root = Path(source_root)

    @property
    def name(self) -> str:
        return "harvey-legal-agent-bench"

    def load(
        self,
        *,
        source: str | Path | None = None,
        limit: int | None = None,
        evidence_dir: Path | None = None,  # noqa: ARG002 - part of the protocol; unused here
    ) -> AgentEvalTaskset:
        root = Path(source) if source is not None else self._source_root
        tasks = build_lab_tasks(root, judge_model=self._judge_model, limit=limit)
        return AgentEvalTaskset(tasks=tasks, metadata={"name": self.name, "lab_source_revision": LAB_REVISION})


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download LAB source and report task/skill inventory.")
    parser.add_argument("--source-dir", default="./data/lab-source", help="Where to download/extract LAB source.")
    parser.add_argument("--no-download", action="store_true", help="Fail if the source is not already present.")
    args = parser.parse_args(argv)

    source_root = ensure_lab_source(args.source_dir, allow_download=not args.no_download)
    n_tasks = sum(1 for _ in iter_source_tasks(source_root))
    n_skill_files = len(_skill_seeds(source_root))
    print(f"LAB source: {source_root}")
    print(f"tasks: {n_tasks} (expected {EXPECTED_TASK_COUNT})")
    print(f"skills: {', '.join(SKILLS)} ({n_skill_files} files seeded per task under {SKILLS_SUBDIR}/)")


if __name__ == "__main__":
    _main()
