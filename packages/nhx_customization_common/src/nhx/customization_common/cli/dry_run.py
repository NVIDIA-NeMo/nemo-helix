# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--dry-run`` for the Customizer backend commands.

``--dry-run`` submits the job with ``?dry_run=true``: the backend validates and compiles it
as submit would, checks the execution profile and the first rows of the dataset, and
creates nothing. ``--dry-run-full`` adds ``?dry_run_full=true`` to check every row.
"""

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import typer
from nemo_helix_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_helix_plugin.cli_state import resolve_output_format
from nemo_helix_plugin.commands import resolve_submit_auth_headers, resolve_submit_base_url
from nemo_helix_plugin.jobs._cli_options import load_options_file, merge_options, parse_dotted_kv_list
from nemo_helix_plugin.jobs.types import JobDryRunCheck
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nhx.customization_common.contributor.jobs import BaseSubmitJob
from nhx.customization_common.dry_run import check_local_dataset_rows

# A full dry run answers only once it has read the whole dataset.
_FULL_DRY_RUN_TIMEOUT = 1800.0


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
    full: bool = False,
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
        with _waiting(full, "Checking every row of the dataset on the platform"):
            job = NemoJobScheduler().submit_remote(
                job_cls,
                json.loads(spec_json),
                base_url=resolve_submit_base_url(typer_ctx),
                workspace=workspace,
                profile=profile,
                options=merged_options or None,
                headers=headers or None,
                timeout=_FULL_DRY_RUN_TIMEOUT if full else 60.0,
                dry_run=True,
                dry_run_full=full,
            )
    except httpx.HTTPStatusError as exc:
        # A failed check is a 422 that lists every check; any other error is printed as submit would.
        checks = _failed_checks(exc.response)
        if checks is None:
            print_http_status_error(exc, action=f"dry-run {backend} job")
            raise typer.Exit(code=2) from exc
        _report(typer_ctx, {"checks": checks, "passed": False})
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=f"dry-run {backend} job")
        raise typer.Exit(code=2) from exc
    checks = (job.get("status_details") or {}).get("checks", [])
    _report(typer_ctx, {"spec": job.get("spec"), "checks": checks, "passed": True})


def _failed_checks(response: httpx.Response) -> list[dict[str, Any]] | None:
    """The checks of a dry run's 422, or None for any other error."""
    if response.status_code != 422:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return detail.get("checks") if isinstance(detail, dict) else None


def dry_run_local(
    typer_ctx: typer.Context,
    job_cls: type[BaseSubmitJob],
    validated_spec: dict[str, Any],
    *,
    workspace: str,
    dataset_source: str | None,
    full: bool = False,
) -> None:
    """Dry-run a job whose references --upload-model, --upload-dataset or --upload-environment would create.

    Those references do not exist yet, so only the job JSON and a local dataset are checked.
    """
    checks = [JobDryRunCheck(name="job JSON", passed=True, message="Matches the schema.")]
    if dataset_source is not None:
        input_spec = job_cls._job_input_schema().model_validate(validated_spec)
        [training, *_] = job_cls._dry_run_datasets(input_spec, workspace)
        path = Path(dataset_source).expanduser()
        if path.exists():
            with _waiting(full, f"Checking every row of {path}"):
                checks.append(check_local_dataset_rows(training, path, full=full))
        else:
            checks.append(JobDryRunCheck(name=training.name, passed=False, message=f"{path} does not exist."))
    report = {
        "spec": validated_spec,
        "checks": [check.model_dump() for check in checks],
        "passed": all(check.passed for check in checks),
    }
    _report(
        typer_ctx,
        report,
        note="The model and dataset references and the execution profile are checked at submit, after the upload.",
    )


@contextmanager
def _waiting(full: bool, description: str) -> Generator[None, None, None]:
    """While a full dry run reads every row, show a spinner and the time elapsed on stderr.

    A full dry run has no partial result to show, so this tells the user it is still working.
    It is gone when the check returns, leaving stdout to the report.
    """
    if not full:
        yield
        return
    # Imported here: Rich is only needed for a full dry run.
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    columns = (SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn())
    with Progress(*columns, console=Console(stderr=True), transient=True) as progress:
        progress.add_task(description, total=None)
        yield


def _report(typer_ctx: typer.Context, report: dict[str, Any], *, note: str | None = None) -> None:
    if resolve_output_format(typer_ctx) != "table":
        typer.echo(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            typer.echo(f"{'OK  ' if check['passed'] else 'FAIL'} {check['name']}: {check['message']}")
        if note:
            typer.echo(note)
        typer.echo(f"Dry run {'passed' if report['passed'] else 'failed'}. Nothing was created.")
    raise typer.Exit(code=0 if report["passed"] else 1)
