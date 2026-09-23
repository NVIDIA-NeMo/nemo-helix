# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimize-skills loop: run evals -> analyze -> fix -> verify -> repeat.

POC shape:
- Operates against an explicit ``--evals`` directory and ``--agent`` directory
  (no more hardcoded ``tests/agentic-use/`` or ``.agents/skills/``).
- ``--skills-path`` is the strategy's writable scope inside the agent dir.
- ``--repeats`` opt-in for noise reduction (median/majority aggregation).
- ``--open-pr`` opt-in for auto-opening a GitLab MR (default: print diff, user
  pushes manually).
"""

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from nemo_agents_plugin.improvement.analysis.llm import generate_gap_analysis
from nemo_agents_plugin.improvement.baselines import load_baselines, save_baselines, update_baselines
from nemo_agents_plugin.improvement.models import (
    AppliedHypothesis,
    BatchResult,
    EvalStatus,
    GapCategory,
    Hypothesis,
    IterationRecord,
    LoopState,
)
from nemo_agents_plugin.improvement.runners._harbor_discovery import discover_evals
from nemo_agents_plugin.improvement.runners._harbor_results import parse_batch_results
from nemo_agents_plugin.improvement.runners.base import Runner
from nemo_agents_plugin.improvement.runners.detect import detect_runner
from nemo_agents_plugin.improvement.traces.claude_code_parser import ClaudeCodeTraceParser
from nemo_agents_plugin.improvement.worktree import create_worktree, remove_worktree
from nemo_helix_plugin.git_url import git_remote_host
from rich.console import Console

# ``discover_evals`` and ``parse_batch_results`` are imported directly because
# they expose richer APIs (filter_glob/filter_names; saved-batch parsing) than
# the ``Runner`` protocol covers today. The loop is harbor-only at the apply
# step (it shells out to the claude CLI to edit skills), so this coupling is
# intentional and asserted up-front in ``run_loop``. When NAT-shaped strategies
# arrive, both the protocol surface and the apply mechanism need to widen.

console = Console()

# Decision thresholds for the improved/regressed/neutral verdict.
# Wallclock duration is the primary signal; tokens are a tiebreaker when
# duration is in the neutral band.
DURATION_THRESHOLD_PCT = 5.0
TOKEN_THRESHOLD_PCT = 10.0


async def _wait_for_proc(
    proc: asyncio.subprocess.Process,
    timeout: float,
    label: str,
) -> tuple[bytes, bytes]:
    """Wait for a subprocess with timeout handling.

    On timeout, kills the process and returns empty byte strings.
    """
    try:
        return await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        # Drain any pending output so the kernel pipes close and the child is
        # reaped — otherwise repeated timeouts leak zombies.
        try:
            await proc.communicate()
        except Exception:
            pass
        console.print(f"[yellow]{label} timed out after {timeout}s[/yellow]")
        return b"", b""


async def _git_uncommitted_files(worktree_path: Path) -> list[str]:
    """Files with uncommitted changes (staged, unstaged, or untracked)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "status",
        "--porcelain",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await _wait_for_proc(proc, timeout=30, label="git status")
    files: list[str] = []
    for line in stdout.decode().splitlines():
        if not line:
            continue
        # Porcelain: "XY <path>" or for renames "R  old -> new". Take post-rename path.
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


async def _git_add_and_commit(worktree_path: Path, message: str) -> str | None:
    """Stage all changes, commit, return new HEAD sha. Returns None on failure."""
    add_proc = await asyncio.create_subprocess_exec(
        "git",
        "add",
        "-A",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, add_err = await _wait_for_proc(add_proc, timeout=30, label="git add")
    if add_proc.returncode != 0:
        console.print(f"  [yellow]git add failed: {add_err.decode()[:200]}[/yellow]")
        return None

    commit_proc = await asyncio.create_subprocess_exec(
        "git",
        "commit",
        "-m",
        message,
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, commit_err = await _wait_for_proc(commit_proc, timeout=30, label="git commit")
    if commit_proc.returncode != 0:
        console.print(f"  [yellow]git commit failed: {commit_err.decode()[:200]}[/yellow]")
        return None

    rev_proc = await asyncio.create_subprocess_exec(
        "git",
        "rev-parse",
        "HEAD",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    rev_out, _ = await _wait_for_proc(rev_proc, timeout=30, label="git rev-parse")
    if rev_proc.returncode != 0:
        return None
    return rev_out.decode().strip()


def load_loop_state(path: Path) -> LoopState:
    """Load loop state from JSON file."""
    if not path.exists():
        return LoopState()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return LoopState()

    iterations = []
    for rec in data.get("iterations", []):
        # Support both old (single hypothesis) and new (list) formats
        h_list = rec.get("hypotheses", [])
        if not h_list and "hypothesis" in rec:
            h_list = [rec["hypothesis"]]

        hypotheses = []
        for h in h_list:
            hypotheses.append(
                Hypothesis(
                    cluster_id=h.get("cluster_id", h.get("eval_name", "")),
                    eval_names=h.get("eval_names", [h["eval_name"]] if "eval_name" in h else []),
                    root_cause=h["root_cause"],
                    category=GapCategory(h["category"]),
                    proposed_fix=h["proposed_fix"],
                    affected_files=h.get("affected_files", []),
                    expected_impact=h.get("expected_impact", ""),
                    confidence=h.get("confidence", 0.5),
                )
            )

        applied = [
            AppliedHypothesis(
                cluster_id=a.get("cluster_id", ""),
                commit_sha=a.get("commit_sha", ""),
                changed_files=a.get("changed_files", []),
                explanation=a.get("explanation", ""),
            )
            for a in rec.get("applied", [])
        ]

        iterations.append(
            IterationRecord(
                iteration=rec["iteration"],
                hypotheses=hypotheses,
                branch_name=rec["branch_name"],
                changes_made=rec.get("changes_made", []),
                eval_results_before=rec.get("eval_results_before", {}),
                eval_results_after=rec.get("eval_results_after", {}),
                improvement_pct=rec.get("improvement_pct", 0.0),
                status=rec.get("status", "error"),
                mr_url=rec.get("mr_url"),
                applied=applied,
            )
        )

    return LoopState(
        iteration=data.get("iteration", 0),
        iterations=iterations,
        current_baseline_batch=data.get("current_baseline_batch", ""),
    )


def save_loop_state(state: LoopState, path: Path) -> None:
    """Save loop state to JSON file."""
    from nemo_agents_plugin.improvement.models import _serialize

    data = {
        "iteration": state.iteration,
        "current_baseline_batch": state.current_baseline_batch,
        "iterations": [_serialize(rec) for rec in state.iterations],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def _select_non_overlapping(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Greedily select hypotheses whose affected_files don't overlap."""
    selected: list[Hypothesis] = []
    used_files: set[str] = set()
    for h in hypotheses:  # already sorted by confidence desc
        h_files = set(h.affected_files)
        # Treat empty affected_files as wildcard — only allow one
        if not h_files:
            if any(not set(s.affected_files) for s in selected):
                continue
            selected.append(h)
        elif not h_files & used_files:
            selected.append(h)
            used_files |= h_files
    return selected


async def apply_hypothesis(
    hypothesis: Hypothesis,
    worktree_path: Path,
    skills_path: str,
    evals_rel_path: str,
) -> tuple[list[str], str]:
    """Launch claude CLI in the worktree to implement the fix.

    Args:
        hypothesis: Hypothesis to implement.
        worktree_path: Path to the isolated git worktree.
        skills_path: Relative path inside the worktree where skills live and the
            strategy is allowed to write (e.g. ".skills" or ".agents/skills").
        evals_rel_path: Relative path inside the worktree to the eval directory
            (immutable from the loop's perspective).

    Returns (list of files changed, Claude's description of changes).
    """
    constraints = f"""- Modify ONLY files under {skills_path}
- NEVER modify the eval directory ({evals_rel_path}) or any of its contents (instructions, task.toml, workflow.yml, tests/test_outputs.py, environment/Dockerfile)
- NEVER modify CLI code, agent configuration, or any other files
- The goal is to improve agent skills only — no other code changes are permitted"""

    prompt = f"""You are implementing a targeted fix to improve agent eval results.

## Target Cluster: {hypothesis.cluster_id}
## Affected Evals: {", ".join(hypothesis.eval_names)}
## Root Cause: {hypothesis.root_cause}
## Category: {hypothesis.category.value}
## Proposed Fix: {hypothesis.proposed_fix}
## Files to modify: {", ".join(hypothesis.affected_files)}

Implement this fix. Be minimal and targeted — only change what's needed.

IMPORTANT CONSTRAINTS:
{constraints}

After making changes, describe what you changed.
"""

    # Strip ANTHROPIC_* so the CLI uses OAuth instead of picking up API keys
    # intended for the NeMo Helix application (which may lack model access). Also
    # strip the CLAUDE_CODE_* / CLAUDECODE markers — when the loop is launched
    # from inside a Claude Code session those signal the child to refuse to
    # nest, leaving an empty diff and a confusing "no changes" verdict.
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("ANTHROPIC_") and k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")
    }

    # Pipe prompt via stdin; use cwd= for working directory
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(worktree_path),
        env=clean_env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=prompt.encode()), timeout=600)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.communicate()
        except Exception:
            pass
        console.print("[yellow]Claude CLI timed out after 600s[/yellow]")
        return [], ""

    claude_explanation = stdout.decode() if stdout else ""

    if proc.returncode != 0:
        console.print(f"[yellow]Claude exited with code {proc.returncode}[/yellow]")
        if stderr:
            console.print(stderr.decode()[:500])
        if stdout:
            console.print(f"[dim]stdout: {stdout.decode()[:500]}[/dim]")

    # Check what changed
    diff_proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "--name-only",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
    )
    diff_out, _ = await _wait_for_proc(diff_proc, timeout=30, label="git diff")
    changed_files = [f for f in diff_out.decode().strip().split("\n") if f]

    # Also check untracked files
    status_proc = await asyncio.create_subprocess_exec(
        "git",
        "status",
        "--porcelain",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
    )
    status_out, _ = await _wait_for_proc(status_proc, timeout=30, label="git status")
    for line in status_out.decode().strip().split("\n"):
        if line.startswith("?? "):
            changed_files.append(line[3:])

    # Reject changes to eval definitions — the loop improves the agent, not evals.
    # In POC the strategy is always skills-optimizer, so anything outside
    # skills_path is forbidden. Plus an explicit guard on the eval directory.
    skills_prefix = skills_path.rstrip("/") + "/"
    forbidden = [
        f for f in changed_files if not f.startswith(skills_prefix) or f.startswith(evals_rel_path.rstrip("/") + "/")
    ]
    if forbidden:
        console.print(f"[red]Rejecting changes to protected files: {', '.join(forbidden)}[/red]")
        for f in forbidden:
            revert_proc = await asyncio.create_subprocess_exec(
                "git",
                "checkout",
                "HEAD",
                "--",
                f,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await _wait_for_proc(revert_proc, timeout=10, label=f"git checkout (revert {f})")
            # If checkout didn't help (untracked file), delete it directly.
            # Untracked entries can be directories (Claude often creates a new
            # `<skill>/` dir), so handle both files and directories.
            filepath = worktree_path / f
            if filepath.is_symlink() or filepath.is_file():
                filepath.unlink()
            elif filepath.is_dir():
                shutil.rmtree(filepath, ignore_errors=True)
        changed_files = [f for f in changed_files if f not in forbidden]

    return changed_files, claude_explanation


async def create_mr(
    worktree_path: Path,
    branch_name: str,
    hypotheses: list[Hypothesis],
    changed_files: list[str],
    claude_explanations: list[str],
    before_results: dict[str, float],
    after_batch: BatchResult,
    avg_improvement: float,
) -> str | None:
    """Push branch and create a GitLab MR with detailed description."""
    # Build per-cluster sections
    cluster_sections = []
    for h, explanation in zip(hypotheses, claude_explanations):
        cluster_sections.append(f"""### {h.cluster_id}: {h.category.value}
**Evals**: {", ".join(h.eval_names)}
**Confidence**: {h.confidence:.0%}

**Root Cause**: {h.root_cause}

**Fix**: {h.proposed_fix}

{explanation[:1000] if explanation else "(no description captured)"}
""")

    # Build eval results table
    eval_rows = []
    for result in after_batch.results:
        name = result.eval_name
        status = result.status.value.upper()
        before_dur = before_results.get(name)
        after_dur = result.agent_timing.duration_sec
        before_str = f"{before_dur:.0f}s" if before_dur else "N/A"
        after_str = f"{after_dur:.0f}s"
        tools = result.tool_calls.total
        eval_rows.append(f"| {name} | {status} | {before_str} | {after_str} | {tools} |")

    total_evals = sum(len(h.eval_names) for h in hypotheses)
    title = f"self-improve: {len(hypotheses)} fixes across {total_evals} evals"

    body = f"""## Self-Improvement Loop — Automated Fix

{"".join(cluster_sections)}

### Files Modified
{", ".join(f"`{f}`" for f in changed_files)}

### Eval Results (before -> after)

| Eval | Status | Before | After | Tool Calls |
|------|--------|--------|-------|------------|
{"".join(eval_rows)}

**Average improvement: {avg_improvement:+.1f}%**

---
*Generated by the self-improvement loop (`nhx-self-improve`)*
"""

    # Push the branch
    push_proc = await asyncio.create_subprocess_exec(
        "git",
        "push",
        "-u",
        "origin",
        branch_name,
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, push_err = await asyncio.wait_for(push_proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        push_proc.kill()
        try:
            await push_proc.communicate()
        except Exception:
            pass
        console.print("[yellow]git push timed out after 60s — branch kept locally[/yellow]")
        return None
    if push_proc.returncode != 0:
        console.print(f"[yellow]git push failed: {push_err.decode()[:300]}[/yellow]")
        return None

    # Detect remote and dispatch to gh or glab
    forge = await _detect_forge(worktree_path)
    if forge == "github":
        cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--head", branch_name]
        forge_label = "GitHub PR"
    elif forge == "gitlab":
        cmd = [
            "glab",
            "mr",
            "create",
            "--title",
            title,
            "--description",
            body,
            "--source-branch",
            branch_name,
            "--remove-source-branch",
            "--yes",
        ]
        forge_label = "GitLab MR"
    else:
        console.print(
            f"[yellow]Could not detect forge from git remote. Branch '{branch_name}' "
            f"pushed; open a PR/MR manually.[/yellow]"
        )
        return None

    if not shutil.which(cmd[0]):
        console.print(
            f"[yellow]Detected {forge} remote but '{cmd[0]}' CLI is not installed. "
            f"Branch '{branch_name}' pushed; open the {forge_label} manually.[/yellow]"
        )
        return None

    mr_proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        mr_out, mr_err = await asyncio.wait_for(mr_proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        mr_proc.kill()
        try:
            await mr_proc.communicate()
        except Exception:
            pass
        console.print(f"[yellow]{cmd[0]} timed out after 60s — branch pushed but {forge_label} not created[/yellow]")
        return None
    if mr_proc.returncode != 0:
        console.print(f"[yellow]{cmd[0]} failed: {mr_err.decode()[:300]}[/yellow]")
        return None

    mr_output = mr_out.decode().strip()
    for line in mr_output.split("\n"):
        if "http" in line:
            return line.strip()
    return mr_output or None


async def _detect_forge(worktree_path: Path) -> str | None:
    """Inspect ``git remote -v`` and return 'github', 'gitlab', or None."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "remote",
        "get-url",
        "origin",
        cwd=str(worktree_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    host = git_remote_host(out.decode().strip())
    if host == "github.com":
        return "github"
    # Match gitlab.com, self-hosted instances (gitlab.example.com), and
    # variants whose first label contains 'gitlab' (gitlab-master.nvidia.com).
    if host == "gitlab.com" or host.startswith("gitlab.") or "gitlab" in host.split(".", 1)[0]:
        return "gitlab"
    return None


async def run_loop(
    agent_root: Path,
    evals_dir: Path,
    skills_path: str = ".agents/skills",
    filter_glob: str | None = None,
    max_iterations: int = 3,
    concurrency: int = 2,
    state_path: Path | None = None,
    initial_batch_dir: Path | None = None,
    full_verification: bool = False,
    open_pr: bool = False,
    repeats: int = 1,
    runner: Runner | None = None,
    trace_parser: str = "claude-code",
) -> LoopState:
    """Run the optimize-skills loop.

    Args:
        agent_root: Root of the agent's repo / directory. The strategy is
            allowed to write under ``agent_root / skills_path``. Worktrees are
            created from this root.
        evals_dir: Directory containing the eval suite. Must be inside
            ``agent_root`` for v0 (so it appears in the worktree). Plugin-wide
            invariant: the loop never writes to this path.
        skills_path: Relative path inside ``agent_root`` where skills live and
            the strategy is allowed to write.
        full_verification: If True, re-run ALL evals (not just targeted ones)
            in the verification step to catch regressions across the full suite.
            Much slower but gives full confidence.
        open_pr: If True, push the branch and open a GitLab MR via ``glab`` on
            improvement. Default False — print the diff and let the user push.
        repeats: Number of trials per eval (median/majority aggregation when >1).
        runner: Eval-suite runner. Defaults to ``HarborRunner``. The loop's
            apply step (edits skills via the claude CLI) is harbor-shaped, so
            non-harbor runners are rejected up-front until that gap is closed.
        trace_parser: Name of the ``TraceParser`` to use. Today only
            ``"claude-code"`` is registered; future parsers extend this set.
    """
    # Discovery runs only when the caller didn't pin a runner. `prefer="harbor"`
    # preserves pre-discovery behavior for ambiguous (both-marker) suites.
    if runner is None:
        runner = detect_runner(evals_dir, prefer="harbor")
        console.print(f"[bold]Discovered AUT: {runner.name}[/bold]")

    # Caller-named trace parser (single explicit branch — no registry).
    # When a second parser ships, add its branch here.
    if trace_parser == "claude-code":
        parser = ClaudeCodeTraceParser()
    else:
        raise ValueError(f"Unknown trace_parser: {trace_parser!r}. Supported: 'claude-code'.")

    state_path = state_path or Path("loop_state.json")
    state = load_loop_state(state_path)
    baseline_path = agent_root / "baselines.json"
    baselines = load_baselines(baseline_path)

    # Per-run id namespaces branch + worktree names so re-running the loop
    # after a kept improvement (or a failure that left state behind) doesn't
    # collide on "self-improve/iter-1" with the prior run's artifacts.
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")

    # Resolve evals_dir relative to agent_root for worktree mirroring
    try:
        evals_rel = evals_dir.resolve().relative_to(agent_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"evals_dir ({evals_dir}) must be inside agent_root ({agent_root}) for v0") from exc
    evals_rel_path = str(evals_rel)
    project_root = agent_root  # backwards-compat for unchanged code below
    agentic_dir = evals_dir

    # Get initial batch results
    if initial_batch_dir and initial_batch_dir.exists():
        batch = parse_batch_results(initial_batch_dir)
        console.print(f"[bold]Loaded existing batch: {batch.batch_id} ({len(batch.results)} results)[/bold]")
    else:
        console.print("[bold]Running initial eval batch...[/bold]")
        evals = discover_evals(agentic_dir, filter_glob=filter_glob)
        batch_name = f"loop-{datetime.now(timezone.utc).strftime('%Y-%m-%d__%H-%M-%S')}"
        batch_dir = project_root / "jobs" / batch_name
        batch = await runner.run_batch(
            evals=evals,
            batch_dir=batch_dir,
            concurrency=concurrency,
            project_root=project_root,
            repeats=repeats,
        )

    state.current_baseline_batch = batch.batch_id
    baselines = update_baselines(baselines, batch)
    save_baselines(baselines, baseline_path)

    # Capture the run's last-iteration index up front; state.iteration
    # advances each loop turn, so recomputing inside the loop makes the
    # displayed denominator climb (Iteration 2 / 4 -> 3 / 5 -> ...).
    total_at_run_end = state.iteration + max_iterations
    for iteration in range(state.iteration, total_at_run_end):
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold]Iteration {iteration + 1} / {total_at_run_end}[/bold]")
        console.print(f"[bold]{'=' * 60}[/bold]")

        # Step 1: Analyze and cluster
        console.print("\n[bold]Step 1: Analyzing gaps and clustering...[/bold]")
        gap_analysis = await generate_gap_analysis(
            batch=batch,
            parser=parser,
            baselines=baselines,
            skills_path=skills_path,
        )

        console.print(f"  Found {len(gap_analysis.clusters)} clusters, {len(gap_analysis.hypotheses)} hypotheses")

        # Step 2: Select non-overlapping untried hypotheses
        untried = [h for h in gap_analysis.hypotheses if not state.was_tried(h)]
        if not untried:
            console.print("[yellow]No new hypotheses to try. Stopping.[/yellow]")
            break

        selected = _select_non_overlapping(untried)
        console.print(f"\n[bold]Step 2: Testing {len(selected)} hypotheses[/bold]")
        for h in selected:
            console.print(f"  [{h.cluster_id}] {h.category.value}: {h.eval_name}")
            console.print(f"    Root cause: {h.root_cause[:100]}")
            console.print(f"    Confidence: {h.confidence:.0%}")

        # Apply step is harbor-only (shells out to `claude` to edit `skills_path`).
        # Analyze-only flows hit `run_analyze_only` and never reach this guard.
        if runner.name != "harbor":
            raise RuntimeError(
                f"run_loop's apply step currently supports only the harbor runner; got {runner.name!r}. "
                "NAT (and other non-skill-based agents) need a different apply mechanism "
                "and hypothesis schema before the loop can target them. "
                "Use --analyze-only to get suggestions without the apply step."
            )

        # Step 3: Create worktree and apply ALL fixes
        branch_name = f"self-improve/{run_id}/iter-{iteration + 1}"
        console.print(f"\n[bold]Step 3: Creating worktree ({branch_name})...[/bold]")

        try:
            worktree_path = await create_worktree(project_root, branch_name)
        except RuntimeError as e:
            console.print(f"[red]Failed to create worktree: {e}[/red]")
            state.iterations.append(
                IterationRecord(
                    iteration=iteration + 1,
                    hypotheses=selected,
                    branch_name=branch_name,
                    changes_made=[],
                    eval_results_before={},
                    eval_results_after={},
                    improvement_pct=0.0,
                    status="error",
                )
            )
            state.iteration = iteration + 1
            save_loop_state(state, state_path)
            continue

        # Apply hypotheses one at a time, committing after each so that each
        # hypothesis's edits land in a standalone commit on the iteration
        # branch. Per-hypothesis commits give us clean attribution: a later
        # prune/revert step can drop one hypothesis's edits without touching
        # the others, and the per-hypothesis file list is just that commit's
        # diff (no manual de-dup against other hypotheses needed).
        applied: list[AppliedHypothesis] = []
        all_changed_files: list[str] = []
        all_explanations: list[str] = []
        seen: set[str] = set()
        for h in selected:
            console.print(f"[bold]Applying fix for {h.cluster_id}...[/bold]")
            _, explanation = await apply_hypothesis(
                h, worktree_path, skills_path=skills_path, evals_rel_path=evals_rel_path
            )
            all_explanations.append(explanation)

            this_files = await _git_uncommitted_files(worktree_path)
            if not this_files:
                console.print(f"  [yellow]No files changed for {h.cluster_id}[/yellow]")
                continue

            commit_msg = f"self-improve [{h.cluster_id}] {h.category.value}: {h.proposed_fix[:100]}"
            commit_sha = await _git_add_and_commit(worktree_path, commit_msg)
            if commit_sha is None:
                console.print(f"  [yellow]Failed to commit edits for {h.cluster_id}; changes left uncommitted[/yellow]")
                continue

            applied.append(
                AppliedHypothesis(
                    cluster_id=h.cluster_id,
                    commit_sha=commit_sha,
                    changed_files=this_files,
                    explanation=explanation,
                )
            )
            new_files = [f for f in this_files if f not in seen]
            seen.update(this_files)
            all_changed_files.extend(new_files)
            console.print(f"  Committed {commit_sha[:8]} — {len(this_files)} files: {', '.join(this_files[:5])}")

        if not all_changed_files:
            console.print("[yellow]No files changed. Discarding worktree.[/yellow]")
            await remove_worktree(project_root, worktree_path, delete_branch=branch_name)
            state.iterations.append(
                IterationRecord(
                    iteration=iteration + 1,
                    hypotheses=selected,
                    branch_name=branch_name,
                    changes_made=[],
                    eval_results_before={},
                    eval_results_after={},
                    improvement_pct=0.0,
                    status="neutral",
                    applied=applied,
                )
            )
            state.iteration = iteration + 1
            save_loop_state(state, state_path)
            continue

        # Step 4: Re-run evals
        all_eval_names: set[str] = set()
        for h in selected:
            all_eval_names.update(h.eval_names)

        # Mirror evals_dir into the worktree
        worktree_evals_dir = worktree_path / evals_rel_path
        if full_verification:
            console.print("\n[bold]Step 4: Re-running ALL evals (full verification)...[/bold]")
            affected_evals = discover_evals(worktree_evals_dir)
        else:
            console.print("\n[bold]Step 4: Re-running affected evals...[/bold]")
            affected_evals = discover_evals(worktree_evals_dir, filter_names=all_eval_names)

        before_results: dict[str, float] = {}
        for result in batch.results:
            if result.eval_name in {e.name for e in affected_evals}:
                before_results[result.eval_name] = result.agent_timing.duration_sec

        after_batch_name = f"verify-iter-{iteration + 1}"
        after_batch_dir = worktree_path / "jobs" / after_batch_name
        try:
            after_batch = await runner.run_batch(
                evals=affected_evals,
                batch_dir=after_batch_dir,
                concurrency=concurrency,
                skip_build=False,
                project_root=worktree_path,
                repeats=repeats,
            )
        except RuntimeError as e:
            console.print(f"[red]Eval batch failed: {e}[/red]")
            await remove_worktree(project_root, worktree_path, delete_branch=branch_name)
            state.iterations.append(
                IterationRecord(
                    iteration=iteration + 1,
                    hypotheses=selected,
                    branch_name=branch_name,
                    changes_made=all_changed_files,
                    eval_results_before={},
                    eval_results_after={},
                    improvement_pct=0.0,
                    status="error",
                    applied=applied,
                )
            )
            state.iteration = iteration + 1
            save_loop_state(state, state_path)
            continue

        after_results: dict[str, float] = {}
        for result in after_batch.results:
            after_results[result.eval_name] = result.agent_timing.duration_sec

        # Step 5: Compare
        # Use default arg to capture batch at definition time (not by late-binding ref)
        def _was_passing(eval_name: str, _batch: BatchResult = batch) -> bool:
            prev = _batch.get_result(eval_name)
            return prev is not None and prev.passed

        def _was_failing(eval_name: str, _batch: BatchResult = batch) -> bool:
            prev = _batch.get_result(eval_name)
            return prev is not None and not prev.passed

        newly_passing = sum(1 for r in after_batch.results if r.status == EvalStatus.PASS and _was_failing(r.eval_name))
        newly_failing = sum(1 for r in after_batch.results if r.status != EvalStatus.PASS and _was_passing(r.eval_name))

        # Duration and token improvements, only for evals that were already passing
        duration_improvements: list[float] = []
        token_improvements: list[float] = []
        for name in before_results:
            if name in after_results and before_results[name] > 0 and _was_passing(name):
                pct = (before_results[name] - after_results[name]) / before_results[name] * 100
                duration_improvements.append(pct)

                prev = batch.get_result(name)
                curr = after_batch.get_result(name)
                if prev and curr and prev.tokens.total > 0:
                    tok_pct = (prev.tokens.total - curr.tokens.total) / prev.tokens.total * 100
                    token_improvements.append(tok_pct)

        avg_improvement = sum(duration_improvements) / len(duration_improvements) if duration_improvements else 0.0
        avg_token_improvement = sum(token_improvements) / len(token_improvements) if token_improvements else 0.0

        # Step 4.5: Per-hypothesis prune.
        #
        # For each applied hypothesis, compute its attribution restricted to
        # the evals it claimed in ``h.eval_names``. Hypotheses whose own
        # targets net-regressed (their proposed_fix made things worse on the
        # very evals they were supposed to help) are revert candidates: they
        # can't justify keeping by their own scope, and they're the most
        # likely culprits for any cross-pollination regressions on shared
        # files.
        #
        # Reverting them via ``git revert`` preserves an audit trail on the
        # branch (the revert commits are visible in ``git log``) and lets us
        # selectively keep the surviving hypotheses without re-running the
        # verify batch — the cross-pollination concern motivates this whole
        # step, but the cost of a second verify batch (~30 min wallclock) is
        # exactly what we're trying to avoid.
        #
        # The post-prune verdict assumes that evals owned exclusively by
        # reverted hypotheses snap back to their baseline status. Evals
        # claimed by any surviving hypothesis trust the verify status. Evals
        # claimed by no hypothesis are treated as collateral and trust verify
        # too (conservative — counts collateral regressions against the
        # iteration).
        hypothesis_by_id: dict[str, Hypothesis] = {h.cluster_id: h for h in selected}
        to_revert: list[AppliedHypothesis] = []
        attribution_lines: list[str] = []
        for app_h in applied:
            h = hypothesis_by_id.get(app_h.cluster_id)
            if h is None or not h.eval_names:
                attribution_lines.append(f"  [{app_h.cluster_id}] no eval_names — keeping")
                continue
            np_h = 0
            nf_h = 0
            for name in h.eval_names:
                r = after_batch.get_result(name)
                if r is None:
                    continue
                if r.status == EvalStatus.PASS and _was_failing(name):
                    np_h += 1
                elif r.status != EvalStatus.PASS and _was_passing(name):
                    nf_h += 1
            attribution_lines.append(
                f"  [{app_h.cluster_id}] attribution: +{np_h} / -{nf_h} (across {len(h.eval_names)} eval(s))"
            )
            if nf_h > np_h:
                to_revert.append(app_h)

        # Only prune when at least one hypothesis would survive. Reverting
        # all of them is semantically the same as discarding the iteration
        # via the existing verdict path, which also produces a cleaner
        # branch (no revert noise on top of dead commits).
        if to_revert and len(to_revert) < len(applied):
            console.print("\n[bold]Step 4.5: Pruning hypotheses with negative attribution...[/bold]")
            for line in attribution_lines:
                console.print(line)

            revert_failed: list[AppliedHypothesis] = []
            # Reverse-chronological order lets each revert apply against the
            # tip without three-way conflicts from later commits touching the
            # same lines.
            for app_h in reversed(to_revert):
                revert_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "revert",
                    "--no-edit",
                    app_h.commit_sha,
                    cwd=str(worktree_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await _wait_for_proc(revert_proc, timeout=30, label=f"git revert {app_h.commit_sha[:8]}")
                if revert_proc.returncode != 0:
                    console.print(
                        f"  [yellow]Revert of [{app_h.cluster_id}] {app_h.commit_sha[:8]} hit a conflict — "
                        "aborting that revert and keeping the hypothesis[/yellow]"
                    )
                    abort_proc = await asyncio.create_subprocess_exec(
                        "git",
                        "revert",
                        "--abort",
                        cwd=str(worktree_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await _wait_for_proc(abort_proc, timeout=30, label="git revert --abort")
                    revert_failed.append(app_h)
                else:
                    console.print(f"  [green]Reverted [{app_h.cluster_id}] {app_h.commit_sha[:8]}[/green]")

            successfully_reverted_shas = {a.commit_sha for a in to_revert if a not in revert_failed}
            applied = [a for a in applied if a.commit_sha not in successfully_reverted_shas]

            # Recompute pass-rate against the post-prune state. Evals
            # claimed by a surviving hypothesis trust verify; evals only
            # claimed by reverted hypotheses are assumed to snap back to
            # baseline; everything else (collateral) trusts verify too.
            kept_eval_names: set[str] = set()
            for app_h in applied:
                h_kept = hypothesis_by_id.get(app_h.cluster_id)
                if h_kept:
                    kept_eval_names.update(h_kept.eval_names)
            reverted_only_eval_names: set[str] = set()
            for app_h in to_revert:
                if app_h.commit_sha not in successfully_reverted_shas:
                    continue
                h_rev = hypothesis_by_id.get(app_h.cluster_id)
                if not h_rev:
                    continue
                for name in h_rev.eval_names:
                    if name not in kept_eval_names:
                        reverted_only_eval_names.add(name)

            new_np = 0
            new_nf = 0
            for r in after_batch.results:
                name = r.eval_name
                if name in reverted_only_eval_names:
                    # Snap to baseline — revert assumed to undo the change here
                    continue
                if r.status == EvalStatus.PASS and _was_failing(name):
                    new_np += 1
                elif r.status != EvalStatus.PASS and _was_passing(name):
                    new_nf += 1
            newly_passing = new_np
            newly_failing = new_nf

            # Surviving hypotheses' files are the new changes_made set.
            all_changed_files = sorted({f for app_h in applied for f in app_h.changed_files})

            console.print(
                f"[bold]Post-prune: kept {len(applied)} of {len(applied) + len(successfully_reverted_shas)} "
                f"hypotheses; newly_passing={newly_passing}, newly_failing={newly_failing}[/bold]"
            )
        elif to_revert:
            console.print(
                f"[yellow]All {len(applied)} applied hypotheses have negative attribution — "
                "prune skipped (would empty the iteration; falling through to discard)[/yellow]"
            )

        # Net pass-rate change dominates: a fix that moves more evals into PASS
        # than it knocks out is always kept, even if it costs duration/tokens.
        # Duration/tokens only break ties when pass-rate is unchanged.
        net_passrate = newly_passing - newly_failing
        if net_passrate > 0:
            status = "improved"
            console.print(
                f"  [green]Net pass-rate +{net_passrate} ({newly_passing} new PASS, {newly_failing} new FAIL)[/green]"
            )
        elif net_passrate < 0:
            status = "regressed"
            console.print(
                f"  [red]Net pass-rate {net_passrate} ({newly_passing} new PASS, {newly_failing} new FAIL)[/red]"
            )
        elif avg_improvement > DURATION_THRESHOLD_PCT:
            status = "improved"
        elif avg_improvement < -DURATION_THRESHOLD_PCT:
            status = "regressed"
        elif avg_token_improvement > TOKEN_THRESHOLD_PCT:
            status = "improved"
            console.print(f"  [green]Tokens dropped {avg_token_improvement:.1f}% (tiebreaker)[/green]")
        elif avg_token_improvement < -TOKEN_THRESHOLD_PCT:
            status = "regressed"
            console.print(f"  [red]Tokens rose {-avg_token_improvement:.1f}% (tiebreaker)[/red]")
        else:
            status = "neutral"

        record = IterationRecord(
            iteration=iteration + 1,
            hypotheses=selected,
            branch_name=branch_name,
            changes_made=all_changed_files,
            eval_results_before=before_results,
            eval_results_after=after_results,
            improvement_pct=avg_improvement,
            status=status,
            applied=applied,
        )

        console.print(f"\n[bold]Result: [{status}] (avg improvement: {avg_improvement:+.1f}%)[/bold]")

        if status == "improved":
            console.print(f"[green]Keeping branch {branch_name}[/green]")
            baselines = update_baselines(baselines, after_batch)
            save_baselines(baselines, baseline_path)

            # Save updated baselines into the worktree and commit
            worktree_baseline_path = worktree_path / "baselines.json"
            save_baselines(baselines, worktree_baseline_path)

            add_baseline_proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "baselines.json",
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await _wait_for_proc(add_baseline_proc, timeout=30, label="git add baselines.json")

            commit_baseline_proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                "chore: update baselines.json with improved eval results",
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await _wait_for_proc(commit_baseline_proc, timeout=30, label="git commit baselines")
            all_changed_files.append("baselines.json")

            if open_pr:
                console.print("[bold]Opening MR (--open-pr)...[/bold]")
                mr_url = await create_mr(
                    worktree_path=worktree_path,
                    branch_name=branch_name,
                    hypotheses=selected,
                    changed_files=all_changed_files,
                    claude_explanations=all_explanations,
                    before_results=before_results,
                    after_batch=after_batch,
                    avg_improvement=avg_improvement,
                )
                if mr_url:
                    record.mr_url = mr_url
                    console.print(f"[green]MR created: {mr_url}[/green]")
                else:
                    console.print("[yellow]MR creation failed — branch kept locally[/yellow]")
            else:
                console.print(
                    f"[green]Branch {branch_name} kept locally. "
                    f"Run with --open-pr to auto-open a PR/MR, or push manually.[/green]"
                )
            batch = after_batch
        else:
            console.print(f"[yellow]Discarding worktree for {branch_name}[/yellow]")
            await remove_worktree(project_root, worktree_path, delete_branch=branch_name)

        state.iterations.append(record)
        state.iteration = iteration + 1
        save_loop_state(state, state_path)

    console.print(f"\n[bold]Loop complete after {state.iteration} iterations.[/bold]")
    improved_count = sum(1 for i in state.iterations if i.status == "improved")
    console.print(f"[bold]Improvements found: {improved_count}[/bold]")

    return state


async def run_analyze_only(
    evals_dir: Path,
    initial_batch_dir: Path,
    skills_path: str = ".agents/skills",
    trace_parser: str = "claude-code",
    runner: Runner | None = None,
) -> LoopState:
    """Load an existing batch, generate suggestions, exit. AUT-agnostic.

    Writes the gap analysis to ``<initial_batch_dir>/optimize-suggestions.json``
    and skips the harbor-only apply/verify/MR steps.
    """
    from nemo_agents_plugin.improvement.models import _serialize

    if runner is None:
        runner = detect_runner(evals_dir, prefer="harbor")
        console.print(f"[bold]Discovered AUT: {runner.name}[/bold]")

    if trace_parser == "claude-code":
        parser = ClaudeCodeTraceParser()
    else:
        raise ValueError(f"Unknown trace_parser: {trace_parser!r}. Supported: 'claude-code'.")

    if not initial_batch_dir.exists():
        raise RuntimeError(f"initial_batch_dir does not exist: {initial_batch_dir}")

    batch = parse_batch_results(initial_batch_dir)
    console.print(f"[bold]Loaded existing batch: {batch.batch_id} ({len(batch.results)} results)[/bold]")

    console.print("\n[bold]Analyzing gaps and clustering...[/bold]")
    gap_analysis = await generate_gap_analysis(
        batch=batch,
        parser=parser,
        baselines=None,
        skills_path=skills_path,
    )
    console.print(f"  Found {len(gap_analysis.clusters)} clusters, {len(gap_analysis.hypotheses)} hypotheses")

    suggestions_path = initial_batch_dir / "optimize-suggestions.json"
    suggestions_path.write_text(json.dumps(_serialize(gap_analysis), indent=2) + "\n")
    console.print(f"[green]Wrote suggestions to {suggestions_path}[/green]")

    state = LoopState(current_baseline_batch=batch.batch_id)
    return state
