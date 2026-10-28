# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--dry-run`` and ``preflight`` for the Customizer backend commands.

``--dry-run`` sends the job to the backend's dry-run route, which validates and compiles
it as submit would, checks the execution profile and the first rows of the dataset, and
creates nothing. ``preflight`` checks the platform before any data work: it is reachable,
and has the execution profile the backend needs.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import typer
from nemo_helix_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_helix_plugin.cli_state import resolve_output_format
from nemo_helix_plugin.commands import resolve_submit_auth_headers, resolve_submit_base_url
from nemo_helix_plugin.jobs._cli_options import load_options_file, merge_options, parse_dotted_kv_list
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import JobDryRunCheck
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nhx.customization_common.contributor.jobs import BaseSubmitJob
from nhx.customization_common.dry_run import check_local_dataset_rows

_GPU_PROVIDERS = ("gpu", "gpu_distributed")
_KUBERNETES_BACKENDS = ("kubernetes_job", "volcano_job")


def dry_run_remote(
    typer_ctx: typer.Context,
    job_cls: type[BaseSubmitJob],
    spec_json: str,
    *,
    backend: str,
    workspace: str,
    profile: str | None,
    options: list[str],
    options_file: Path | None,
) -> None:
    """Dry-run a job whose references all exist, on the platform, then exit with its result."""
    try:
        merged_options = merge_options(
            load_options_file(options_file) if options_file is not None else {}, parse_dotted_kv_list(options)
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: invalid options — {exc}", err=True)
        raise typer.Exit(code=1) from exc
    headers = resolve_submit_auth_headers(typer_ctx)
    try:
        report = NemoJobScheduler().dry_run_remote(
            job_cls,
            json.loads(spec_json),
            base_url=resolve_submit_base_url(typer_ctx),
            workspace=workspace,
            profile=profile,
            options=merged_options or None,
            headers=headers or None,
        )
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action=f"dry-run {backend} job")
        raise typer.Exit(code=2) from exc
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=f"dry-run {backend} job")
        raise typer.Exit(code=2) from exc
    _report(typer_ctx, report)


def dry_run_local(
    typer_ctx: typer.Context,
    job_cls: type[BaseSubmitJob],
    validated_spec: dict[str, Any],
    *,
    workspace: str,
    dataset_source: str | None,
) -> None:
    """Dry-run a job whose references the --upload flags would create, then exit with its result.

    Those references do not exist yet, so only the job JSON and a local dataset are checked.
    """
    checks = [JobDryRunCheck(name="job JSON", passed=True, message="Matches the schema.")]
    if dataset_source is not None:
        input_spec = job_cls._job_input_schema().model_validate(validated_spec)
        [training, *_] = job_cls._dry_run_datasets(input_spec, workspace)
        path = Path(dataset_source).expanduser()
        if path.exists():
            checks.append(check_local_dataset_rows(training, path))
        else:
            checks.append(JobDryRunCheck(name=training.name, passed=False, message=f"{path} does not exist."))
    report = {
        "spec": validated_spec,
        "steps": [],
        "checks": [check.model_dump() for check in checks],
        "passed": all(check.passed for check in checks),
    }
    _report(
        typer_ctx,
        report,
        note="The model and dataset references and the execution profile are checked at submit, after the upload.",
    )


def _report(typer_ctx: typer.Context, report: dict[str, Any], *, note: str | None = None) -> None:
    if resolve_output_format(typer_ctx) != "table":
        typer.echo(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            typer.echo(f"{'OK  ' if check['passed'] else 'FAIL'} {check['name']}: {check['message']}")
        for step in report["steps"]:
            image = step.get("image") or "set by the execution profile"
            typer.echo(f"     step {step['name']}: {step['provider']}/{step['profile']}, image {image}")
        if note:
            typer.echo(note)
        typer.echo(f"Dry run {'passed' if report['passed'] else 'failed'}. Nothing was created.")
    raise typer.Exit(code=0 if report["passed"] else 1)


def add_preflight_command(group: typer.Typer, *, backend: str, kubernetes_only: bool = False) -> None:
    """Register ``nemo customization <backend> preflight``."""

    @group.command(
        "preflight",
        help=(
            "Check the platform before preparing data: it is reachable and has a GPU execution "
            "profile this backend can use. Creates nothing."
        ),
    )
    def preflight(
        typer_ctx: typer.Context,
        profile: str | None = typer.Option(
            None, "--profile", help="GPU execution profile to check. Any GPU profile passes when omitted."
        ),
    ) -> None:
        base_url = resolve_submit_base_url(typer_ctx)
        headers = resolve_submit_auth_headers(typer_ctx)
        try:
            with JobsClient(base_url=base_url, default_headers=headers or None) as jobs:
                profiles = jobs.list_execution_profiles()
        except Exception as exc:
            checks = [
                JobDryRunCheck(
                    name="platform",
                    passed=False,
                    message=f"Cannot reach {base_url}: {exc}. Start the platform or point NHX_BASE_URL at it.",
                )
            ]
        else:
            checks = [
                JobDryRunCheck(name="platform", passed=True, message=f"Reachable at {base_url}."),
                _check_gpu_profile(profiles, backend=backend, profile=profile, kubernetes_only=kubernetes_only),
            ]
        _report(
            typer_ctx,
            {
                "checks": [check.model_dump() for check in checks],
                "steps": [],
                "passed": all(check.passed for check in checks),
            },
        )


def _check_gpu_profile(
    profiles: list[Any], *, backend: str, profile: str | None, kubernetes_only: bool
) -> JobDryRunCheck:
    gpu = [p for p in profiles if p.provider in _GPU_PROVIDERS and (profile is None or p.profile == profile)]
    configured = ", ".join(sorted(f"{p.provider}/{p.profile} ({p.backend})" for p in profiles)) or "none"
    if not gpu:
        wanted = f"GPU profile '{profile}'" if profile else "a GPU execution profile"
        return JobDryRunCheck(
            name="execution profile",
            passed=False,
            message=f"The platform has no {wanted}. Configured: {configured}.",
        )
    if kubernetes_only:
        runnable = [p for p in gpu if p.backend in _KUBERNETES_BACKENDS]
        if not runnable:
            return JobDryRunCheck(
                name="execution profile",
                passed=False,
                message=(
                    f"{backend} runs on Kubernetes only, and these GPU profiles use another backend: "
                    f"{', '.join(f'{p.provider}/{p.profile} ({p.backend})' for p in gpu)}. "
                    "Use a platform whose jobs run on kubernetes_job or volcano_job."
                ),
            )
        gpu = runnable
    usable = ", ".join(sorted(f"{p.provider}/{p.profile} ({p.backend})" for p in gpu))
    return JobDryRunCheck(name="execution profile", passed=True, message=f"Usable: {usable}.")
