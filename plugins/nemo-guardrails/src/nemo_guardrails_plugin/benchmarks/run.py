# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level entry point for the nemo-guardrails IGW benchmark harness.

The harness orchestrates the entire benchmark run by running the following steps:

1. Resolve paths and validate the upstream NeMo Guardrails checkout.
2. Write a per-run AIPerf config under ``runs/<id>/generated/``.
3. Start the two mock LLM servers from ``${NEMO_GUARDRAILS_REPO_ROOT}/benchmark``.
4. Start (or reuse) ``nemo services run``.
5. Wait for per-process health probes, seed NMP resources via the SDK, smoke-test the VirtualModel.
6. Invoke ``python -m benchmark.aiperf run --config-file ...`` for the sweep.
7. Collect per-sweep results and exit non-zero on any failure.

Process supervision uses session-scoped subprocesses and an ``ExitStack`` so a
``SIGTERM`` from CI cleans up forked workers (e.g. ``uvicorn --workers 4``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from nemo_guardrails_plugin.benchmarks.aiperf_runner import (
    SweepRunResult,
    collect_sweep_results,
    prepare_runtime_aiperf_config,
    run_aiperf_sweep,
)
from nemo_guardrails_plugin.benchmarks.analyze import analyze_run
from nemo_guardrails_plugin.benchmarks.bootstrap import ensure_aiperf_venv
from nemo_guardrails_plugin.benchmarks.constants import (
    AIPERF_SHIM_BASE_URL,
    ALL_VARIANTS,
    APP_PROVIDER_URL,
    CS_PROVIDER_URL,
    IGW_CHAT_PATH,
    NMP_BASE_URL,
    NMP_HEALTH_PATH,
    VARIANT_WITH_GUARDRAILS,
    VARIANT_WITHOUT_GUARDRAILS,
    WORKSPACE,
)
from nemo_guardrails_plugin.benchmarks.paths import (
    RunPaths,
    build_run_paths,
    default_nemoguardrails_repo_root,
    discover_nmp_repo_root,
)
from nemo_guardrails_plugin.benchmarks.processes import (
    SupervisedProcess,
    supervised_processes,
    wait_http,
)
from nemo_guardrails_plugin.benchmarks.seeding import SeededResources, seed_benchmark
from nemo_platform import APIConnectionError, APIStatusError, NeMoPlatform

log = logging.getLogger("nemo_guardrails_plugin.benchmarks")

_MOCK_HEALTH_TIMEOUT_SECONDS = 60.0
_NMP_HEALTH_TIMEOUT_SECONDS = 180.0


_REQUIRED_NEMOGUARDRAILS_FILES = (
    Path("benchmark/aiperf/__main__.py"),
    Path("benchmark/aiperf/run_aiperf.py"),
    Path("benchmark/mock_llm_server/run_server.py"),
    Path("examples/configs/content_safety_local/config.yml"),
    Path("examples/configs/content_safety_local/prompts.yml"),
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _validate_nemoguardrails_repo(nemoguardrails_repo_root: Path) -> None:
    """Fail fast if the upstream checkout is missing files the harness depends on."""
    missing = [p for p in _REQUIRED_NEMOGUARDRAILS_FILES if not (nemoguardrails_repo_root / p).is_file()]
    if missing:
        bullet = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"NeMo Guardrails checkout at {nemoguardrails_repo_root} is missing required files:\n  - {bullet}"
        )


def _validate_in_repo_mock_configs(paths: RunPaths) -> None:
    """Fail fast if the in-repo mock LLM env files are missing.

    These live in this repo (not upstream) so we control mock behavior
    independently of the NeMo-Guardrails checkout.
    """
    missing = [p for p in (paths.mock_app_env, paths.mock_content_safety_env) if not p.is_file()]
    if missing:
        bullet = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(f"In-repo mock LLM config files missing:\n  - {bullet}")


def _build_mock_nim_processes(paths: RunPaths, workers: int) -> list[SupervisedProcess]:
    """Spawn ``python -m benchmark.mock_llm_server.run_server`` for both mocks.

    Each child is given its own log file and a ``PYTHONPATH`` pointing at the
    upstream checkout so its imports resolve.
    """
    env = {"PYTHONPATH": str(paths.nemoguardrails_repo_root)}
    workdir = paths.nemoguardrails_repo_root / "benchmark"

    # Helper to build a ``SupervisedProcess`` for one of the mock LLM servers.
    def spec(name: str, port: int, env_file: Path, *, health_url: str) -> SupervisedProcess:
        return SupervisedProcess(
            name=name,
            cmd=[
                sys.executable,
                "-m",
                "benchmark.mock_llm_server.run_server",
                "--workers",
                str(workers),
                "--port",
                str(port),
                "--config-file",
                str(env_file),
            ],
            log_path=paths.log_dir / f"{name}.log",
            cwd=workdir,
            env=env,
            health_url=health_url,
            health_timeout_seconds=_MOCK_HEALTH_TIMEOUT_SECONDS,
        )

    # Env files come from this repo, rather than the upstream library.
    return [
        spec(
            "mock-app-llm",
            8000,
            paths.mock_app_env,
            health_url=f"{APP_PROVIDER_URL}/health",
        ),
        spec(
            "mock-content-safety-llm",
            8001,
            paths.mock_content_safety_env,
            health_url=f"{CS_PROVIDER_URL}/health",
        ),
    ]


def _build_nmp_process(paths: RunPaths) -> SupervisedProcess:
    """Start ``nemo services run`` in a supervised process.

    The harness sets ``NMP_BASE_URL`` and ``NMP_DATA_DIR`` env vars so the
    child process can talk to NMP over HTTP and write state to the per-run data dir.
    """
    return SupervisedProcess(
        name="nmp-services",
        cmd=["nemo", "services", "run"],
        log_path=paths.log_dir / "nmp-services.log",
        cwd=paths.nmp_repo_root,
        env={"NMP_BASE_URL": NMP_BASE_URL, "NMP_DATA_DIR": str(paths.nmp_data_dir)},
        health_url=f"{NMP_BASE_URL}{NMP_HEALTH_PATH}",
        health_timeout_seconds=_NMP_HEALTH_TIMEOUT_SECONDS,
    )


def _build_aiperf_shim_process(paths: RunPaths) -> SupervisedProcess:
    """Run the shim that satisfies AIPerf's `/v1/models` pre-check.

    Without this, AIPerf's hard-coded health check against the `/v1/models`
    endpoint would 404, and the sweep would never start. See
    `nemo_guardrails_plugin.benchmarks.shim` for details.
    """
    return SupervisedProcess(
        name="aiperf-shim",
        cmd=[sys.executable, "-m", "nemo_guardrails_plugin.benchmarks.shim"],
        log_path=paths.log_dir / "aiperf-shim.log",
        cwd=paths.nmp_repo_root,
        health_url=f"{AIPERF_SHIM_BASE_URL}/__shim/health",
        health_timeout_seconds=_MOCK_HEALTH_TIMEOUT_SECONDS,
    )


def _smoke_test(sdk: NeMoPlatform, seeded: SeededResources) -> None:
    """Verify the VirtualModel is reachable and returns a chat completion,
    before running the AIPerf sweep.
    """
    payload: dict[str, object] = {
        "model": seeded.vm_ref,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }

    last_error: str = "no attempts made"

    for attempt in range(60):
        try:
            body = sdk.inference.gateway.openai.post(
                "v1/chat/completions",
                workspace=WORKSPACE,
                body=payload,
            )
            if body.get("choices"):
                return
            last_error = f"response missing choices: {body}"
        except APIStatusError as exc:
            last_error = f"HTTP {exc.status_code}: {str(exc)[:500]}"
            log.info("Smoke test attempt %d: %s; retrying", attempt + 1, last_error)
        except APIConnectionError as exc:
            last_error = f"transport error: {str(exc)[:500]}"
            log.info("Smoke test attempt %d: %s; retrying", attempt + 1, last_error)
        time.sleep(1.0)

    raise RuntimeError(f"Smoke test failed after 60 attempts: {last_error}")


@dataclass(frozen=True)
class BenchmarkOutcome:
    """Outcome of a single benchmark variant's AIPerf sweep, used by the summary."""

    variant: str
    aiperf_exit: int
    output_dir: Path
    sweep_results: list[SweepRunResult]

    @property
    def failures(self) -> int:
        return sum(1 for r in self.sweep_results if not r.passed)

    @property
    def passed(self) -> bool:
        return self.aiperf_exit == 0 and bool(self.sweep_results) and self.failures == 0


def _vm_ref_for_variant(variant: str, seeded: SeededResources) -> str:
    """Pick which seeded VirtualModel a benchmark variant should target."""
    if variant == VARIANT_WITH_GUARDRAILS:
        return seeded.vm_ref
    if variant == VARIANT_WITHOUT_GUARDRAILS:
        return seeded.no_guardrails_vm_ref
    raise ValueError(f"Unknown variant: {variant!r}")


def _run_benchmark(
    *,
    variant: str,
    paths: RunPaths,
    seeded: SeededResources,
    aiperf_python: Path,
) -> BenchmarkOutcome:
    """Materialize a per-variant AIPerf config, run the sweep, collect results."""
    vm_ref = _vm_ref_for_variant(variant, seeded)
    runtime_config = paths.runtime_config_for(variant)
    aiperf_output_dir = paths.aiperf_output_dir_for(variant)
    aiperf_log = paths.aiperf_log_for(variant)

    sweep_config = prepare_runtime_aiperf_config(
        template_path=paths.config_template,
        runtime_config_path=runtime_config,
        aiperf_output_dir=aiperf_output_dir,
        model_ref=vm_ref,
    )
    log.info(
        "Benchmark %s: targeting %s; concurrency=%s, duration=%ss",
        variant,
        vm_ref,
        sweep_config.get("sweeps", {}).get("concurrency"),
        sweep_config.get("base_config", {}).get("benchmark_duration"),
    )
    log.info(
        "Starting AIPerf sweep [%s] against %s -> shim -> %s%s",
        variant,
        AIPERF_SHIM_BASE_URL,
        NMP_BASE_URL,
        IGW_CHAT_PATH,
    )

    aiperf_exit = run_aiperf_sweep(
        nemoguardrails_repo_root=paths.nemoguardrails_repo_root,
        runtime_config=runtime_config,
        log_path=aiperf_log,
        python_executable=str(aiperf_python),
        venv_bin_path=paths.aiperf_venv_dir / "bin",
    )

    sweep_results = collect_sweep_results(aiperf_output_dir)
    return BenchmarkOutcome(
        variant=variant,
        aiperf_exit=aiperf_exit,
        output_dir=aiperf_output_dir,
        sweep_results=sweep_results,
    )


def _summarize_benchmark_results(outcomes: list[BenchmarkOutcome]) -> int:
    """Log per-benchmark + overall summary; return process exit code."""
    overall_failed = False
    for outcome in outcomes:
        if not outcome.sweep_results:
            log.error(
                "Benchmark %s: aiperf exited with code %d and produced no per-sweep results in %s",
                outcome.variant,
                outcome.aiperf_exit,
                outcome.output_dir,
            )
        else:
            log.info(
                "Benchmark %s: %d run(s), %d failure(s); per-sweep outputs under %s",
                outcome.variant,
                len(outcome.sweep_results),
                outcome.failures,
                outcome.output_dir,
            )
        if not outcome.passed:
            overall_failed = True

    return 1 if overall_failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nemo-guardrails-benchmark",
        description="Run the nemo-guardrails IGW benchmark sweep.",
    )
    parser.add_argument(
        "--nemo-guardrails-repo-root",
        type=Path,
        default=Path(
            os.environ.get(
                "NEMO_GUARDRAILS_REPO_ROOT",
                str(default_nemoguardrails_repo_root(discover_nmp_repo_root())),
            )
        ),
        help="Path to a local NeMo Guardrails checkout (default: $NEMO_GUARDRAILS_REPO_ROOT or ../NeMo-Guardrails).",
    )
    parser.add_argument(
        "--reuse-services",
        action="store_true",
        default=os.environ.get("NMP_BENCHMARK_REUSE_SERVICES", "0") == "1",
        help="Skip starting `nemo services run` and reuse an existing local NMP at :8080.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        default=os.environ.get("NMP_BENCHMARK_KEEP_RUNNING", "0") == "1",
        help="Leave started processes alive after the sweep (debugging).",
    )
    parser.add_argument(
        "--mock-workers",
        type=int,
        default=int(os.environ.get("NMP_BENCHMARK_MOCK_WORKERS", "4")),
        help="uvicorn worker count for each mock LLM server.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override the per-run directory name (default: current timestamp).",
    )
    parser.add_argument(
        "--variant",
        choices=(*ALL_VARIANTS, "all"),
        default="all",
        help=(
            "Which sweep to run. 'all' (default) runs both variants sequentially "
            "against the same NMP; the with-vs-without delta isolates middleware "
            "overhead. In CI, run the two variants as parallel jobs against "
            "separate NMP instances."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _resolve_variants(variant_arg: str) -> tuple[str, ...]:
    """Translate the ``--variant`` CLI argument into the ordered list to run."""
    if variant_arg == "all":
        return ALL_VARIANTS
    return (variant_arg,)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    # Validate the NeMo Guardrails checkout before attempting to run the benchmark.
    nemoguardrails_repo_root = args.nemo_guardrails_repo_root.resolve()
    _validate_nemoguardrails_repo(nemoguardrails_repo_root)

    log.info("Validated NeMo Guardrails local checkout at: %s", nemoguardrails_repo_root)

    # Build the directory structure that will contain the benchmark results.
    nmp_repo_root = discover_nmp_repo_root()
    paths = build_run_paths(
        nmp_repo_root=nmp_repo_root,
        nemoguardrails_repo_root=nemoguardrails_repo_root,
        run_id=args.run_id,
    )
    paths.ensure_directories()
    _validate_in_repo_mock_configs(paths)

    log.info("Created directory for benchmark results at: %s", paths.run_dir)
    log.info(
        "Mock LLM configs: app=%s, content-safety=%s",
        paths.mock_app_env,
        paths.mock_content_safety_env,
    )

    variants = _resolve_variants(args.variant)
    log.info("Will run %d variant(s): %s", len(variants), ", ".join(variants))

    # Ensure the dedicated aiperf venv exists *before* we start any supervised
    # processes.
    aiperf_python = ensure_aiperf_venv(paths.aiperf_venv_dir)
    log.info("Using aiperf python at %s", aiperf_python)

    processes = _build_mock_nim_processes(paths, args.mock_workers)
    if not args.reuse_services:
        processes.append(_build_nmp_process(paths))

    processes.append(_build_aiperf_shim_process(paths))

    # Start the processes and wait for them to be ready before seeding NMP.
    with ExitStack() as stack:
        stack.enter_context(supervised_processes(processes))
        if args.keep_running:
            # Pop the cleanup so processes outlive this script.
            stack.pop_all()

        if args.reuse_services:
            log.info("Waiting for existing NMP services at %s...", NMP_BASE_URL)
            wait_http(
                f"{NMP_BASE_URL}{NMP_HEALTH_PATH}",
                timeout_seconds=_NMP_HEALTH_TIMEOUT_SECONDS,
                label="nmp-services",
            )

        log.info(f"All services are ready. Seeding benchmark resources in workspace {WORKSPACE}...")

        sdk = NeMoPlatform(base_url=NMP_BASE_URL)
        seeded = seed_benchmark(
            sdk,
            nemoguardrails_repo_root=paths.nemoguardrails_repo_root,
            generated_dir=paths.generated_dir,
        )

        log.info("Waiting for VirtualModel %s to be ready...", seeded.vm_ref)
        _smoke_test(sdk, seeded)

        # Variants run sequentially against the same NMP; only the targeted
        # VirtualModel differs, so the delta isolates middleware overhead.
        outcomes: list[BenchmarkOutcome] = []
        for variant in variants:
            outcomes.append(
                _run_benchmark(
                    variant=variant,
                    paths=paths,
                    seeded=seeded,
                    aiperf_python=aiperf_python,
                )
            )

    exit_code = _summarize_benchmark_results(outcomes)
    _maybe_print_analysis(paths.run_dir, outcomes)
    return exit_code


def _maybe_print_analysis(run_dir: Path, outcomes: list[BenchmarkOutcome]) -> None:
    """Print the analyzer's comparison table when at least one variant has results.

    Wrapped in a broad try/except: the analyzer is post-processing only and
    must not change the harness's exit code or hide a real benchmark failure.
    """
    if not any(o.sweep_results for o in outcomes):
        return
    try:
        report = analyze_run(run_dir)
    except Exception as exc:
        log.warning("Analyzer failed; skipping summary table: %s", exc)
        return
    log.info("Benchmark analysis:\n%s", report)


if __name__ == "__main__":
    sys.exit(main())
