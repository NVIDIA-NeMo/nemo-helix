# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-type evaluation adapters that turn a subject into analyst Insights."""

import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
from evaluation.ingest import (
    create_experiment,
    ensure_experiment_group,
    ensure_workspace,
    mint_agent_id,
    poll_visible,
)
from evaluation.intake_client import build_basic_auth_intake_client
from evaluation.otlp_build import session_id_for, sim_to_spans
from evaluation.otlp_ingest import export_spans, post_evaluator_results, trace_id_for
from evaluation.registry import Subject
from evaluation.tau2run import load_tasks, policy_version, read_policy, resolve_paths, run_tau2
from nemo_insights_plugin.analyst.observability import AnalystEvaluationContext
from nemo_insights_plugin.analyst.run import run_analyst
from nemo_insights_plugin.platform_client import make_client
from nemo_platform import AsyncNeMoPlatform

REPO_ROOT = Path(__file__).resolve().parent.parent


class EvaluationAdapter(Protocol):
    def check(self) -> list[str]: ...

    async def produce(self) -> dict[str, object]: ...

    async def analyze(
        self,
        *,
        record: dict[str, object] | None,
        since: datetime | None,
        verbose: bool,
        out_path: Path,
    ) -> str: ...


class IntakeAdapter:
    """Analyze an agent's existing Intake traces (no production step)."""

    def __init__(self, subject: Subject, *, analyst_evaluation: AnalystEvaluationContext | None = None) -> None:
        self.subject = subject
        self.analyst_evaluation = analyst_evaluation

    def check(self) -> list[str]:
        """Unmet prerequisites for this subject (empty list = ready to run)."""
        cfg = self.subject.config
        missing: list[str] = [f"config key '{k}'" for k in ("agent", "workspace", "base_url") if not cfg.get(k)]
        if cfg.get("auth") == "basic":
            missing.extend(self._missing_basic_auth())
        return missing

    def _missing_basic_auth(self) -> list[str]:
        """Return missing basic-auth configuration and environment values."""
        missing: list[str] = []
        for role, key in (("username", "auth_user_env"), ("password", "auth_password_env")):
            env_name = self.subject.config.get(key)
            if not env_name:
                missing.append(f"config key '{key}' (env var name for the basic-auth {role})")
            elif not os.environ.get(str(env_name)):
                missing.append(f"env {env_name} (basic-auth {role}, in evaluation/.env)")
        return missing

    async def produce(self) -> dict[str, object]:
        raise SystemExit(
            f"intake subject '{self.subject.name}' has no produce step — run "
            f"`uv run python -m evaluation analyze {self.subject.name} --live`"
        )

    async def analyze(
        self,
        *,
        record: dict[str, object] | None,
        since: datetime | None,
        verbose: bool,
        out_path: Path,
    ) -> str:
        cfg = self.subject.config
        if missing := self.check():
            raise SystemExit(f"intake evaluation '{self.subject.name}' is missing: {', '.join(missing)}")
        client = self._basic_auth_client() if cfg.get("auth") == "basic" else make_client(str(cfg["base_url"]))
        return await run_analyst(
            agent=cfg["agent"],
            ethos=None,
            workspace=cfg["workspace"],
            base_url=cfg["base_url"],
            client=client,
            insights_output=str(out_path),
            local_only=True,
            verbose=verbose,
            since=since,
            analyst_evaluation=self.analyst_evaluation,
            enable_observability=cfg.get("auth") != "basic",
        )

    def _basic_auth_client(self) -> AsyncNeMoPlatform:
        """Build the basic-auth client configured for this Intake subject."""
        cfg = self.subject.config
        real_prefix = str(cfg.get("intake_path_prefix", "/api/intake")).rstrip("/") + "/"
        return build_basic_auth_intake_client(
            base_url=str(cfg["base_url"]),
            username=os.environ[str(cfg["auth_user_env"])],
            password=os.environ[str(cfg["auth_password_env"])],
            real_prefix=real_prefix,
        )


class BenchmarkAdapter:
    """Run a benchmark to produce traces, ingest them, then analyze."""

    def __init__(self, subject: Subject, *, analyst_evaluation: AnalystEvaluationContext | None = None) -> None:
        self.subject = subject
        self.analyst_evaluation = analyst_evaluation

    def check(self) -> list[str]:
        """Unmet prerequisites for this benchmark (empty list = ready to run)."""
        cfg = self.subject.config
        missing: list[str] = []
        for key in ("domain", "base_url", "workspace", "agent_llm", "user_llm"):
            val = cfg.get(key)
            if not val:
                missing.append(f"config key '{key}'")
            elif key in ("agent_llm", "user_llm") and "<your-model>" in str(val):
                missing.append(f"a real model for '{key}' in evaluations.toml (a model your proxy key serves)")
        # The keys build_argv hard-indexes: absent here = a KeyError mid-run, so
        # doctor must name them up front. Absence only — 0 is a valid seed.
        for key in ("task_split_name", "num_trials", "seed", "max_concurrency"):
            if cfg.get(key) is None:
                missing.append(f"config key '{key}'")
        tau2_bin, data_dir = resolve_paths(cfg, repo_root=REPO_ROOT)
        if tau2_bin is None or data_dir is None:
            missing.append("config key 'tau2_repo' (your tau2-bench checkout; or set tau2_bin/tau2_data_dir)")
        else:
            if not data_dir.is_dir():
                missing.append(f"tau2 data dir ({data_dir}) — clone tau2-bench + `uv sync`")
            if shutil.which(tau2_bin) is None:
                missing.append(f"tau2 binary ({tau2_bin}) — run `uv sync` in the tau2 repo")
        for env_key in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
            if not os.environ.get(env_key):
                missing.append(f"env {env_key} (in evaluation/.env)")
        return missing

    async def produce(self) -> dict[str, object]:
        cfg = self.subject.config
        if missing := self.check():
            raise SystemExit(f"benchmark evaluation '{self.subject.name}' is missing: " + "; ".join(missing))
        tau2_bin, data_dir = resolve_paths(cfg, repo_root=REPO_ROOT)
        assert tau2_bin is not None and data_dir is not None  # guaranteed by check()
        domain = str(cfg["domain"])
        base_url = str(cfg["base_url"])
        base = str(cfg["workspace"])  # stable workspace + agent + experiment-group name
        run_id = mint_agent_id(base)  # the per-run Experiment name + nemo.evaluation.name tag
        agent = base  # stable agent name across runs
        created_at = datetime.now(timezone.utc).isoformat()
        # Stable workspaces: the realistic (oracle-free, blind-eval) target is always
        # produced; the oracle twin (answer key, for the UI) only when include_rewards.
        realistic_workspace = base
        include_rewards = bool(cfg.get("include_rewards", True))
        oracle_workspace = f"{base}-oracle" if include_rewards else None
        dataset_name = f"tau2:{domain}"
        ensure_workspace(base_url, realistic_workspace)  # fail fast before tau2
        if oracle_workspace is not None:
            ensure_workspace(base_url, oracle_workspace)
        sims = run_tau2(cfg, run_id, data_dir=data_dir, tau2_bin=tau2_bin)
        if not sims:
            raise SystemExit(f"benchmark evaluation '{self.subject.name}': tau2 produced no simulations")
        policy = read_policy(data_dir, domain)
        version = policy_version(policy)
        tasks = load_tasks(data_dir, domain)
        agent_llm = str(cfg["agent_llm"])
        experiment_metadata = {
            "agent_llm": agent_llm,
            "user_llm": str(cfg.get("user_llm", "")),
            "num_trials": cfg.get("num_trials"),
            "seed": cfg.get("seed"),
            "task_split_name": cfg.get("task_split_name"),
            "num_tasks": cfg.get("num_tasks"),
            "created_at": created_at,
        }
        for tagged_workspace in [realistic_workspace] + ([oracle_workspace] if oracle_workspace else []):
            group_id = ensure_experiment_group(base_url, tagged_workspace, base)
            create_experiment(
                base_url,
                tagged_workspace,
                name=run_id,
                experiment_group_id=group_id,
                dataset_name=dataset_name,
                dataset_version=version,
                metadata=experiment_metadata,
            )
        session_ids: set[str] = set()
        client = httpx.Client(timeout=30.0)
        try:
            for sim in sims:
                task = tasks.get(str(sim.get("task_id")))
                session_id = session_id_for(sim, evaluation_name=run_id)
                trace_id = trace_id_for(session_id)
                # Stamp spans at ingest time (Intake drops spans dated outside its
                # retention window); one base shared by a sim's realistic + oracle twins.
                base_ns = time.time_ns()
                session_ids.add(session_id)
                realistic_spans = sim_to_spans(
                    sim,
                    agent_name=agent,
                    agent_version=version,
                    session_id=session_id,
                    evaluation_name=run_id,
                    task=task,
                    include_rewards=False,
                    agent_llm=agent_llm,
                    base_ns=base_ns,
                )
                export_spans(base_url, realistic_workspace, session_id, trace_id, realistic_spans, client=client)
                if oracle_workspace is not None:
                    oracle_spans = sim_to_spans(
                        sim,
                        agent_name=agent,
                        agent_version=version,
                        session_id=session_id,
                        evaluation_name=run_id,
                        task=task,
                        include_rewards=True,
                        agent_llm=agent_llm,
                        base_ns=base_ns,
                    )
                    export_spans(base_url, oracle_workspace, session_id, trace_id, oracle_spans, client=client)
                    # OTLP doesn't auto-create the reward row the Analyst reads, so POST it
                    # separately, targeting the EVALUATOR span this oracle build emitted.
                    evaluator = next((s for s in oracle_spans if s["kind"] == "EVALUATOR"), None)
                    if evaluator is not None:
                        post_evaluator_results(
                            base_url,
                            oracle_workspace,
                            span_id=evaluator["span_id"],
                            session_id=session_id,
                            score=float(evaluator["attributes"]["score"]),
                            client=client,
                        )
            if len(session_ids) < 3:
                print(
                    f"warning: only {len(session_ids)} session(s) ingested; the analyst needs 3+ to run.",
                    file=sys.stderr,
                )
            visible = poll_visible(base_url, realistic_workspace, session_ids, client=client)
            if len(visible) < 3:
                print(
                    f"warning: only {len(visible)}/{len(session_ids)} session(s) "
                    "visible in Intake (ingest may still be catching up).",
                    file=sys.stderr,
                )
            if oracle_workspace is not None:
                poll_visible(base_url, oracle_workspace, session_ids, client=client)
        finally:
            client.close()
        return {
            "agent": agent,
            "realistic_workspace": realistic_workspace,
            "oracle_workspace": oracle_workspace,
            "experiment_id": run_id,
            "experiment_group": base,
            "dataset_name": dataset_name,
            "dataset_version": version,
            "base_url": base_url,
            "domain": domain,
            "run_id": run_id,
            "agent_version": version,
            "created_at": created_at,
        }

    async def analyze(
        self,
        *,
        record: dict[str, object] | None,
        since: datetime | None,
        verbose: bool,
        out_path: Path,
    ) -> str:
        if record is None:
            raise SystemExit(
                f"no recorded run for '{self.subject.name}' — run "
                f"`uv run python -m evaluation run {self.subject.name}` first"
            )
        _, data_dir = resolve_paths(self.subject.config, repo_root=REPO_ROOT)
        policy = read_policy(data_dir, str(record["domain"])) if data_dir else None
        workspace = str(record["realistic_workspace"])
        evaluation_id = str(record["experiment_id"])
        print(
            f"analyzing realistic workspace '{workspace}' run '{evaluation_id}' (oracle withheld — unaided eval)",
            file=sys.stderr,
        )
        return await run_analyst(
            agent=str(record["agent"]),
            ethos=policy,
            workspace=workspace,
            base_url=str(record["base_url"]),
            client=make_client(str(record["base_url"])),
            insights_output=str(out_path),
            local_only=True,
            verbose=verbose,
            since=since,
            evaluation_id=evaluation_id,
            analyst_evaluation=self.analyst_evaluation,
        )


_ADAPTERS: dict[str, type[IntakeAdapter] | type[BenchmarkAdapter]] = {
    "intake": IntakeAdapter,
    "benchmark": BenchmarkAdapter,
}


def build_adapter(
    subject: Subject,
    *,
    analyst_evaluation: AnalystEvaluationContext | None = None,
) -> EvaluationAdapter:
    """Construct the adapter for a subject's ``type``."""
    cls = _ADAPTERS.get(subject.type)
    if cls is None:
        raise SystemExit(
            f"evaluation '{subject.name}' has unknown type '{subject.type}'. Known types: {', '.join(sorted(_ADAPTERS))}."
        )
    return cls(subject, analyst_evaluation=analyst_evaluation)
