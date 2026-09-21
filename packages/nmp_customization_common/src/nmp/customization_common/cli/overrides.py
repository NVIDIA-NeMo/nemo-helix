# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI override machinery for customization contributor plugins.

After the platform's ``_add_submit_command`` registers the default submit verb,
Customizer backends swap in the same shape:

- ``submit`` → positional ``JOB_JSON`` argument plus standard submit flags;
  loads + validates the JSON (via the backend's ``load_job_json``), then
  delegates to the original ``submit`` callback with ``--spec`` set.
- any pre-existing generated ``run`` command is removed; Customizer jobs are
  submitted to the platform, not executed through local CLI scheduling.
- ``explain`` → unchanged.

After a successful submit, the wrapper reports the created job and the commands
that track it, and follows the job to a terminal state for ``--wait`` and
``--watch``. That reporting lives in
``nmp.customization_common.cli.tracking``.

Only the backend's name, ``load_job_json``, ``JOB_JSON`` help text and ``submit``
help text differ; everything else is shared here.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import httpx
import typer
from nemo_platform_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_platform_plugin.cli_options import WorkspaceOption
from nemo_platform_plugin.cli_state import resolve_cli_workspace
from nemo_platform_plugin.client.errors import NemoClientError
from nemo_platform_plugin.commands import (
    SubmittedJob,
    resolve_submit_auth_headers,
    resolve_submit_base_url,
)
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.models.client import ModelsClient
from nmp.customization_common.cli.tracking import FollowResult, follow_job
from nmp.customization_common.cli.uploads import (
    SpecRefs,
    UploadError,
    UploadReport,
    create_resources,
    find_conflicts,
    write_refs,
)
from pydantic import ValidationError

LoadJobJson = Callable[[Path], str]
ValidateJobSpec = Callable[[dict], str]

_LIFECYCLE_PANEL = "Lifecycle Options"
_UPLOAD_PANEL = "Resource Creation Options"
#: Stands in for a reference that does not exist yet, so the rest of the job
#: JSON can be checked before anything is created.
_PLACEHOLDER_REF = "pending-upload"


def apply_job_cli_overrides(
    group: typer.Typer,
    *,
    backend: str,
    load_job_json: LoadJobJson,
    job_json_help: str,
    submit_help: str | None = None,
    spec_refs: SpecRefs | None = None,
    validate_job_spec: ValidateJobSpec | None = None,
) -> None:
    """Drop generated ``run``/``submit`` verbs, then re-register submit.

    Order matters: drop first, then re-register. Typer iterates
    ``registered_commands`` in insertion order, so stale entries would route
    users back to the auto-generated shapes.
    """
    _drop_command(group, "run")
    _replace_job_submit(group, backend, load_job_json, job_json_help, submit_help, spec_refs, validate_job_spec)


def _pluck_callback(group: typer.Typer, verb: str) -> Callable[..., SubmittedJob | None]:
    command = next((c for c in group.registered_commands if c.name == verb), None)
    if command is None or command.callback is None:
        raise RuntimeError(f"missing {verb!r} callback to override")
    return command.callback


def _drop_command(group: typer.Typer, name: str) -> None:
    group.registered_commands = [c for c in group.registered_commands if c.name != name]


def _replace_job_submit(
    group: typer.Typer,
    backend: str,
    load_job_json: LoadJobJson,
    job_json_help: str,
    submit_help: str | None = None,
    spec_refs: SpecRefs | None = None,
    validate_job_spec: ValidateJobSpec | None = None,
) -> None:
    """Replace ``submit`` with a ``JOB_JSON`` positional + standard submit flags."""
    original = _pluck_callback(group, "submit")
    # Drop the original before re-registering so we don't leave a duplicate
    # ``submit`` entry (Typer would otherwise keep both and dispatch the last).
    _drop_command(group, "submit")

    @group.command("submit", help=submit_help)
    def submit(
        typer_ctx: typer.Context,
        job_json: Path = typer.Argument(..., metavar="JOB_JSON", help=job_json_help),
        workspace: WorkspaceOption = None,
        profile: str | None = typer.Option(
            None,
            "--profile",
            help="Execution profile to run the job on. Uses the backend default when omitted.",
        ),
        cluster: str | None = typer.Option(
            None,
            "--cluster",
            help="Name of a cluster in the CLI config. Submits to that cluster's base URL.",
        ),
        base_url: str | None = typer.Option(
            None,
            "--base-url",
            help=(
                "Override platform API host. If omitted: --cluster, then CLI context, "
                "then $NMP_BASE_URL, then http://localhost:8080."
            ),
        ),
        options: list[str] = typer.Option([], "-o", help="Backend option override, 'backend.key=value'."),
        options_file: Path | None = typer.Option(
            None,
            "--options-file",
            help="JSON or YAML file of backend option overrides. Any -o flag wins over it.",
        ),
        wait: bool = typer.Option(
            False,
            "--wait",
            help="Wait for the job to finish, showing its status.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        watch: bool = typer.Option(
            False,
            "--watch",
            help="Wait for the job to finish, showing its status and logs.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        timeout: int | None = typer.Option(
            None,
            "--timeout",
            min=1,
            help="Give up waiting after this many seconds. Waits indefinitely when omitted.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        poll_interval: int = typer.Option(
            3,
            "--poll-interval",
            min=1,
            help="Seconds between status checks while waiting.",
            rich_help_panel=_LIFECYCLE_PANEL,
        ),
        upload_model: str | None = typer.Option(
            None,
            "--upload-model",
            metavar="SOURCE",
            help=("Local path to model weights, or a HuggingFace repo id. Creates the fileset and model entity."),
            rich_help_panel=_UPLOAD_PANEL,
        ),
        upload_dataset: str | None = typer.Option(
            None,
            "--upload-dataset",
            metavar="SOURCE",
            help=(
                "Local file or directory to upload as the dataset. Pass a directory to send "
                "several files, the same as 'nemo files upload'."
            ),
            rich_help_panel=_UPLOAD_PANEL,
        ),
        upload_environment: str | None = typer.Option(
            None,
            "--upload-environment",
            metavar="SOURCE",
            help="Local directory to upload as a NeMo Gym environment. GRPO only.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
        exist_ok: bool = typer.Option(
            False,
            "--exist-ok",
            help="Reuse whichever of the above already exists. Its files are left as they are.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
        hf_token_secret: str | None = typer.Option(
            None,
            "--hf-token-secret",
            help="Platform secret holding a HuggingFace token, for a gated or private repo.",
            rich_help_panel=_UPLOAD_PANEL,
        ),
    ) -> None:
        workspace = resolve_cli_workspace(typer_ctx, workspace)

        if wait and watch:
            raise typer.BadParameter("Use either --wait or --watch, not both.")

        if upload_model is None and upload_dataset is None and upload_environment is None:
            spec_json = load_job_json(job_json)
        else:
            if spec_refs is None or validate_job_spec is None:
                raise typer.BadParameter(f"The {backend} backend cannot create resources at submit time.")
            if upload_environment is not None and spec_refs.environment is None:
                raise typer.BadParameter(f"The {backend} backend has no environment to upload.")
            spec_json = _upload_then_validate(
                job_json,
                refs=spec_refs,
                validate_job_spec=validate_job_spec,
                model_source=upload_model,
                dataset_source=upload_dataset,
                environment_source=upload_environment,
                typer_ctx=typer_ctx,
                workspace=workspace,
                base_url=base_url,
                cluster=cluster,
                exist_ok=exist_ok,
                hf_token_secret=hf_token_secret,
            )
        submitted = original(
            typer_ctx,
            spec=spec_json,
            spec_file=None,
            options=options,
            options_file=options_file,
            profile=profile,
            cluster=cluster,
            base_url=base_url,
            workspace=workspace,
            config=None,
            config_file=None,
        )
        _report_submitted(submitted, wait=wait, watch=watch, timeout=timeout, poll_interval=poll_interval)


def _upload_then_validate(
    job_json: Path,
    *,
    refs: SpecRefs,
    validate_job_spec: ValidateJobSpec,
    model_source: str | None,
    dataset_source: str | None,
    environment_source: str | None,
    typer_ctx: typer.Context,
    workspace: str,
    base_url: str | None,
    cluster: str | None,
    exist_ok: bool,
    hf_token_secret: str | None,
) -> str:
    """Create the resources, fill their references in, and validate the result.

    The job JSON may leave ``model`` or ``dataset`` out entirely when the
    matching flag is given, because the reference does not exist until the
    resource is created. Both stay required by the API; this fills them in
    before the job is validated and sent.

    Everything else in the file is checked first, against a copy carrying
    placeholders. A typo elsewhere then fails before anything is uploaded,
    rather than after a model has been pushed to the platform.
    """
    spec = _read_job_json(job_json)
    _reject_conflicts(
        spec,
        refs,
        model=model_source is not None,
        dataset=dataset_source is not None,
        environment=environment_source is not None,
    )
    _precheck(
        spec,
        refs,
        validate_job_spec,
        model=model_source is not None,
        dataset=dataset_source is not None,
        environment=environment_source is not None,
    )

    resolved_base_url = resolve_submit_base_url(typer_ctx, base_url=base_url, cluster=cluster)
    headers = resolve_submit_auth_headers(typer_ctx) or None
    report = run_uploads(
        model_source=model_source,
        dataset_source=dataset_source,
        environment_source=environment_source,
        files=FilesClient(base_url=resolved_base_url, workspace=workspace, default_headers=headers),
        models=ModelsClient(base_url=resolved_base_url, workspace=workspace, default_headers=headers),
        workspace=workspace,
        exist_ok=exist_ok,
        hf_token_secret=hf_token_secret,
    )
    write_refs(spec, refs, report)
    return _validate(spec, validate_job_spec)


def _reject_conflicts(
    spec: dict,
    refs: SpecRefs,
    *,
    model: bool,
    dataset: bool,
    environment: bool,
) -> None:
    """Refuse when a reference is given both in the job JSON and on the command line."""
    conflicts = find_conflicts(spec, refs, model=model, dataset=dataset, environment=environment)
    if not conflicts:
        return
    fields = ", ".join(conflicts)
    typer.echo(
        f"Error: the job JSON already sets {fields}, and the matching --upload flag would "
        "replace it. Set each reference in one place: remove the field from the job JSON to "
        "create it here, or drop the flag to use what the file names.",
        err=True,
    )
    raise typer.Exit(code=2)


def _read_job_json(job_json: Path) -> dict:
    """Read the file as a JSON object, reporting a bad file as a CLI error."""
    try:
        spec = json.loads(job_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: could not read {job_json}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(spec, dict):
        typer.echo(f"Error: {job_json} must hold a JSON object.", err=True)
        raise typer.Exit(code=2)
    return spec


def _precheck(
    spec: dict,
    refs: SpecRefs,
    validate_job_spec: ValidateJobSpec,
    *,
    model: bool,
    dataset: bool,
    environment: bool,
) -> None:
    """Validate a copy standing in for the references we are about to create."""
    probe = deepcopy(spec)
    if model:
        _set_placeholder(probe, refs.model)
    if dataset:
        _set_placeholder(probe, refs.dataset)
    if environment and refs.environment is not None:
        _set_placeholder(probe, refs.environment)
    _validate(probe, validate_job_spec)


def _set_placeholder(spec: dict, path: tuple[str, ...]) -> None:
    """Fill *path* with a stand-in value, leaving a value the user supplied alone."""
    node = spec
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node.setdefault(path[-1], _PLACEHOLDER_REF)


def _validate(spec: dict, validate_job_spec: ValidateJobSpec) -> str:
    """Validate *spec*, reporting a schema failure as a CLI error."""
    try:
        return validate_job_spec(spec)
    except ValidationError as exc:
        typer.echo(f"Error: invalid job JSON — {exc}", err=True)
        raise typer.Exit(code=2) from exc


def run_uploads(
    *,
    model_source: str | None,
    dataset_source: str | None,
    environment_source: str | None = None,
    files: FilesClient,
    models: ModelsClient,
    workspace: str,
    exist_ok: bool,
    hf_token_secret: str | None,
) -> UploadReport:
    """Create the resources, reporting failures as CLI errors rather than tracebacks."""
    try:
        report = create_resources(
            model_source=model_source,
            dataset_source=dataset_source,
            environment_source=environment_source,
            files=files,
            models=models,
            workspace=workspace,
            exist_ok=exist_ok,
            hf_token_secret=hf_token_secret,
        )
    except UploadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action="create Customizer resources")
        raise typer.Exit(code=2) from exc
    except httpx.RequestError as exc:
        print_http_request_error(exc, action="create Customizer resources")
        raise typer.Exit(code=2) from exc
    except NemoClientError as exc:
        typer.echo(f"Error: could not create Customizer resources: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    report_uploads(report)
    return report


def report_uploads(report: UploadReport) -> None:
    """Say what was created and what was reused, on stderr."""
    for ref in report.created:
        typer.echo(f"Created {ref}", err=True)
    for ref in report.reused:
        typer.echo(f"Reused {ref}, which already existed.", err=True)
    if report.reused_files_untouched:
        typer.echo(
            "Files in a reused fileset are not re-uploaded, so a local edit since it "
            "was created will not be part of this job.",
            err=True,
        )


def _report_submitted(
    submitted: SubmittedJob | None,
    *,
    wait: bool,
    watch: bool,
    timeout: int | None,
    poll_interval: int,
) -> None:
    """Follow the job to a terminal state when ``--wait`` or ``--watch`` asked for it.

    The commands for tracking the job later are printed by the submit renderer
    (``CustomizationSubmitRenderer``), not here.
    """
    if not (wait or watch):
        return
    # Exit 0 means the job completed, so a job that cannot be followed is an error.
    job_name = submitted.name if submitted is not None else None
    if submitted is None or job_name is None:
        typer.echo("Error: the submit response has no job name, so the job cannot be followed.", err=True)
        raise typer.Exit(code=1)

    result = follow_job(
        base_url=submitted.base_url,
        job_name=job_name,
        workspace=submitted.workspace,
        headers=submitted.headers,
        include_logs=watch,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    if result is FollowResult.INTERRUPTED:
        raise typer.Exit(code=130)
    if result is not FollowResult.SUCCEEDED:
        raise typer.Exit(code=1)
