# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scaled-evals — a thin command-line client for the control-plane API.

Mirrors the creation flow from docs/API.md: register credentials, save config
profiles, publish a task (create + upload tarball), then start an
evaluation. Auth and endpoint resolve as flag > env var > default.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import click

from scaled_evals.api.build.errors import BuildError
from scaled_evals.api.build.uploaded_context import archive_context_directory
from scaled_evals.api.failure_diagnostics import failure_category_for_code, is_retryable_failure
from scaled_evals.benchmark_import import (
    import_id_from_legacy_state,
    load_benchmark_manifest,
    validate_benchmark_import,
    write_legacy_import_state,
)

from .client import (
    DEFAULT_BASE_URL,
    ApiError,
    download_artifact,
    download_presigned,
    emit,
    emit_list,
    fetch_artifact,
    iter_sse,
    load_arg,
    make_client,
    request,
    upload_file,
    upload_multipart_archive,
)

VISIBILITY = ["private", "team", "org", "public"]
CREDENTIAL_PROVIDER = ["openai", "anthropic", "nvidia", "nmp", "openshift", "switchyard"]
CONFIG_PROFILE_TYPE = ["harbor", "gym", "switchyard", "intake"]
FRAMEWORK = ["harbor", "nemo_gym"]
ORDER = ["asc", "desc"]
KNOWN_RUNTIME_HINT = (
    "API default: sandbox_k8s. Runtime strings are resolved by the API runtime backend "
    "registry. Local defaults load sandbox_k8s and Gym plugins; other runtime strings are passed "
    "through to the API."
)
EVAL_STATUS = [
    "blocked",
    "queued",
    "provisioning",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]
FAILURE_CATEGORY = ["infrastructure", "provider", "task", "unknown"]
SUCCESS_STATUSES = frozenset({"succeeded"})
FAILED_STATUSES = frozenset({"failed", "cancelled", "blocked"})
TERMINAL_STATUSES = SUCCESS_STATUSES | FAILED_STATUSES
JSON_OPTION_HELP = "Emit raw JSON instead of a summary; accepted before or after subcommands."


def _mark_json_output(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value:
        return
    root_ctx = ctx.find_root()
    root_obj = root_ctx.ensure_object(dict)
    if root_obj.get("_json_option_seen"):
        raise click.UsageError("--json may be supplied only once.", ctx=ctx)
    root_obj["json"] = True
    root_obj["_json_option_seen"] = True


def _json_output_option() -> click.Option:
    return click.Option(
        ["--json"],
        is_flag=True,
        expose_value=False,
        callback=_mark_json_output,
        help=JSON_OPTION_HELP,
    )


def _install_json_output_option(command: click.Command) -> None:
    if not any("--json" in param.opts for param in command.params if isinstance(param, click.Option)):
        command.params.append(_json_output_option())
    if isinstance(command, click.Group):
        for subcommand in command.commands.values():
            _install_json_output_option(subcommand)


@click.group()
@click.option(
    "--base-url",
    envvar="SCALED_EVALS_BASE_URL",
    default=DEFAULT_BASE_URL,
    show_default=True,
    show_envvar=True,
    help="Control-plane base URL; /v1 is added automatically.",
)
@click.option(
    "--token",
    envvar="SCALED_EVALS_TOKEN",
    default=None,
    show_envvar=True,
    help="Bearer token sent with every API call. Prefer the env var; a --token flag leaks into shell history and ps.",
)
@click.pass_context
def cli(ctx: click.Context, base_url: str, token: str | None) -> None:
    """Client for the scaled-evals control-plane API."""
    ctx.ensure_object(dict)
    ctx.obj.setdefault("json", False)
    if ctx.invoked_subcommand == "benchmark-import":
        ctx.obj.update({"client": None, "base_url": base_url, "token": token})
        return
    # An explicit token is the only credential source. Interactive login used to
    # mint one here against a fixed internal identity provider and cache it in
    # the OS keyring; under the platform, the deployment supplies the token.
    client = make_client(base_url, token)
    ctx.call_on_close(client.close)
    ctx.obj.update({"client": client, "base_url": base_url})


@cli.command("whoami")
@click.pass_context
def whoami(ctx: click.Context) -> None:
    """Show the principal the control plane resolves this caller as."""
    data = request(ctx.obj["client"], "GET", "/users/me")
    emit(
        data,
        ctx.obj["json"],
        [
            f"User: {data.get('name') or data.get('id')}",
            f"ID: {data.get('id')}",
            f"Email: {data.get('email') or '(not provided)'}",
            f"Auth source: {(data.get('principal') or {}).get('source')}",
        ],
    )


# ---------- benchmark import ---------------------------------------------


@cli.group("benchmark-import")
@click.pass_context
def benchmark_import_group(ctx: click.Context) -> None:
    """Validate and import materialized Harbor benchmark catalogs."""
    if ctx.invoked_subcommand == "validate":
        return
    base_url = ctx.obj["base_url"]
    token = ctx.obj.get("token")
    client = make_client(base_url, token)
    ctx.call_on_close(client.close)
    ctx.obj["client"] = client


def _import_summary(data: dict[str, Any]) -> list[str]:
    raw_tasks = data.get("tasks")
    tasks = [task for task in raw_tasks if isinstance(task, dict)] if isinstance(raw_tasks, list) else []
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    task_status = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"
    return [
        f"benchmark import {data.get('id')}",
        f"  status:     {data.get('status')}",
        f"  visibility: {data.get('visibility')}",
        f"  manifest:   {data.get('manifest_sha256')}",
        f"  tasks:      {task_status}",
        f"  benchmarks: {len(data.get('benchmarks') or [])}",
    ]


def _load_import_images(raw: str | None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    text = load_arg(raw)
    stripped = text.lstrip()
    payload: Any
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(text)
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict) and payload.get("slug"):
        records = [payload]
    elif (
        isinstance(payload, dict)
        and "images" not in payload
        and all(isinstance(value, dict) for value in payload.values())
    ):
        records = [{"slug": slug, **value} for slug, value in payload.items()]
    elif isinstance(payload, dict):
        records = payload.get("images") or payload.get("results") or []
    else:
        records = payload
    images: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("status") == "failed":
            continue
        slug = str(record.get("slug") or "")
        image_ref = str(record.get("image_ref") or "")
        image_digest = str(record.get("image_digest") or "")
        if slug and image_ref and image_digest:
            image = dict(record)
            image.pop("slug", None)
            image.pop("error", None)
            image["image_ref"] = image_ref
            image["image_digest"] = image_digest
            images[slug] = image
    return images


def _merge_import_images(values: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    for value in values:
        images.update(_load_import_images(value))
    return images


def _upload_import_packs(ctx: click.Context, data: dict[str, Any], output_root: Path) -> int:
    uploaded = 0
    for task in data.get("tasks") or []:
        upload = task.get("upload")
        if not upload:
            continue
        pack = (output_root / str(task["pack_path"])).resolve()
        try:
            pack.relative_to(output_root.resolve())
        except ValueError as exc:
            raise click.ClickException(f"unsafe pack path for {task['slug']}: {pack}") from exc
        upload_file(ctx.obj["client"], upload, pack)
        uploaded += 1
    return uploaded


def _wait_for_import(
    ctx: click.Context,
    import_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    started = time.monotonic()
    data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    while data.get("status") in {"uploading", "preparing"}:
        if time.monotonic() - started >= timeout:
            raise click.ClickException(f"timed out waiting for benchmark import {import_id}; it remains resumable")
        time.sleep(poll_interval)
        data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    return data


def _run_benchmark_import(
    ctx: click.Context,
    *,
    manifest: Path,
    output_root: Path,
    visibility: str,
    description: str,
    images: dict[str, dict[str, Any]],
    reserve_only: bool,
    retry_failed: bool,
    wait: bool,
    timeout: float,
    poll_interval: float,
    publish: bool,
    smoke_size: int | None = None,
) -> dict[str, Any]:
    validation = validate_benchmark_import(manifest, output_root=output_root)
    if not validation.valid:
        raise click.ClickException("manifest is not conformant; run benchmark-import validate")
    data = request(
        ctx.obj["client"],
        "POST",
        "/benchmark-imports",
        json={
            "manifest_sha256": validation.manifest_sha256,
            "manifest": load_benchmark_manifest(manifest),
            "visibility": visibility,
            "description": description,
            "images": images,
        },
    )
    if reserve_only:
        return data
    if retry_failed and data.get("status") == "failed":
        data = request(ctx.obj["client"], "POST", f"/benchmark-imports/{data['id']}/retry")
    _upload_import_packs(ctx, data, output_root)
    request(ctx.obj["client"], "POST", f"/benchmark-imports/{data['id']}/prepare")
    data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{data['id']}")
    if wait or publish:
        data = _wait_for_import(ctx, data["id"], timeout=timeout, poll_interval=poll_interval)
    if publish:
        if data.get("status") == "failed":
            raise click.ClickException(f"benchmark import {data['id']} failed; inspect status")
        data = request(
            ctx.obj["client"],
            "POST",
            f"/benchmark-imports/{data['id']}/publish",
            json={"smoke_size": smoke_size},
        )
    return data


@benchmark_import_group.command("validate")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--max-pack-bytes", type=click.IntRange(min=1), default=None, hidden=True)
@click.pass_context
def benchmark_import_validate(
    ctx: click.Context, manifest: Path, output_root: Path | None, max_pack_bytes: int | None
) -> None:
    """Validate one materialized catalog without server access or state mutation."""
    kwargs: dict[str, Any] = {"output_root": output_root}
    if max_pack_bytes is not None:
        kwargs["max_pack_bytes"] = max_pack_bytes
    result = validate_benchmark_import(manifest, **kwargs)
    payload = result.model_dump(mode="json", exclude_none=True)
    if ctx.obj["json"]:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(f"benchmark import manifest: {'valid' if result.valid else 'invalid'}")
        for check in result.checks:
            subject = f" [{check.subject}]" if check.subject else ""
            click.echo(f"  {check.status:26} {check.code}{subject}: {check.message}")
    if not result.valid:
        ctx.exit(1)


@benchmark_import_group.command("start")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--visibility",
    type=click.Choice(["private", "team", "org", "public"]),
    default="public",
    show_default=True,
)
@click.option("--description", required=True)
@click.option("--images", default=None, help="Optional prebuilt image result JSON/JSONL or @file.")
@click.option("--reserve-only", is_flag=True, help="Reserve exact revisions without uploading.")
@click.option("--wait", is_flag=True, help="Wait for all task revisions to finish preparing.")
@click.option("--publish", is_flag=True, help="Wait and publish immutable benchmark revisions.")
@click.option("--timeout", type=float, default=3600.0, show_default=True)
@click.option("--poll-interval", type=float, default=5.0, show_default=True)
@click.pass_context
def benchmark_import_start(
    ctx: click.Context,
    manifest: Path,
    output_root: Path,
    visibility: str,
    description: str,
    images: str | None,
    reserve_only: bool,
    wait: bool,
    publish: bool,
    timeout: float,
    poll_interval: float,
) -> None:
    """Create or resume an import, upload missing packs, and begin preparation."""
    if reserve_only and (wait or publish):
        raise click.ClickException("--reserve-only cannot be combined with --wait or --publish")
    data = _run_benchmark_import(
        ctx,
        manifest=manifest,
        output_root=output_root,
        visibility=visibility,
        description=description,
        images=_load_import_images(images),
        reserve_only=reserve_only,
        retry_failed=False,
        wait=wait,
        timeout=timeout,
        poll_interval=poll_interval,
        publish=publish,
    )
    emit(data, ctx.obj["json"], _import_summary(data))


@benchmark_import_group.command("status")
@click.argument("import_id")
@click.pass_context
def benchmark_import_status(ctx: click.Context, import_id: str) -> None:
    """Inspect durable upload, build, image, and benchmark progress."""
    data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    emit(data, ctx.obj["json"], _import_summary(data))


@benchmark_import_group.command("retry")
@click.argument("import_id")
@click.option("--output-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.pass_context
def benchmark_import_retry(ctx: click.Context, import_id: str, output_root: Path) -> None:
    """Create replacement revisions only for failed imported task packs."""
    data = request(ctx.obj["client"], "POST", f"/benchmark-imports/{import_id}/retry")
    _upload_import_packs(ctx, data, output_root)
    request(ctx.obj["client"], "POST", f"/benchmark-imports/{import_id}/prepare")
    data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    emit(data, ctx.obj["json"], _import_summary(data))


@benchmark_import_group.command("publish")
@click.argument("import_id")
@click.option("--smoke-size", type=click.IntRange(min=1), default=None)
@click.pass_context
def benchmark_import_publish(ctx: click.Context, import_id: str, smoke_size: int | None) -> None:
    """Publish prepared exact task pins as immutable benchmark revisions."""
    data = request(
        ctx.obj["client"],
        "POST",
        f"/benchmark-imports/{import_id}/publish",
        json={"smoke_size": smoke_size},
    )
    emit(data, ctx.obj["json"], _import_summary(data))


@benchmark_import_group.command("compat-sync", hidden=True)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--state", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--description", default=None)
@click.option("--visibility", type=click.Choice(["private", "team", "org", "public"]), default=None)
@click.option("--attempts", type=int, default=6, hidden=True)
@click.pass_context
def benchmark_import_compat_sync(
    ctx: click.Context,
    manifest: Path,
    state: Path,
    description: str | None,
    visibility: str | None,
    attempts: int,
) -> None:
    """Compatibility projection for sync-standard-benchmark-tasks."""
    del attempts
    payload = load_benchmark_manifest(manifest)
    output_root = manifest.resolve().parent.parent
    data = _run_benchmark_import(
        ctx,
        manifest=manifest,
        output_root=output_root,
        visibility=visibility or "public",
        description=description or str(payload.get("catalog_id") or manifest.stem),
        images={},
        reserve_only=True,
        retry_failed=False,
        wait=False,
        timeout=0,
        poll_interval=0,
        publish=False,
    )
    write_legacy_import_state(state, data, phase="upload")
    emit(data, ctx.obj["json"], _import_summary(data))


@benchmark_import_group.command("compat-finalize", hidden=True)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--upload-state", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--output-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--state", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "--prebuilt-image-results",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--prebuilt-results-only", is_flag=True)
@click.option("--only-failed", is_flag=True)
@click.option("--resolve-only", is_flag=True)
@click.option("--finalize-resolved-only", is_flag=True)
@click.option("--skip-error-substring", multiple=True)
@click.option("--skip-slug-substring", multiple=True)
@click.option("--limit", type=int, default=None)
@click.option("--finalize-timeout", type=float, default=2100.0)
@click.option("--finalize-poll-seconds", type=float, default=5.0)
@click.option("--image-builder-url", default=None, hidden=True)
@click.option("--builder-source-commit", default=None, hidden=True)
@click.option("--context-path", default=None, hidden=True)
@click.option("--token", default=None, hidden=True)
@click.option("--token-command", default=None, hidden=True)
@click.option("--auth-login-command", default=None, hidden=True)
@click.option("--auth-refresh-seconds", type=int, default=900, hidden=True)
@click.option("--workers", type=int, default=4, hidden=True)
@click.option("--attempts", type=int, default=6, hidden=True)
@click.option("--retry-sleep", type=float, default=5.0, hidden=True)
@click.pass_context
def benchmark_import_compat_finalize(
    ctx: click.Context,
    manifest: Path,
    upload_state: Path,
    output_root: Path,
    state: Path,
    prebuilt_image_results: tuple[Path, ...],
    prebuilt_results_only: bool,
    only_failed: bool,
    resolve_only: bool,
    finalize_resolved_only: bool,
    skip_error_substring: tuple[str, ...],
    skip_slug_substring: tuple[str, ...],
    limit: int | None,
    finalize_timeout: float,
    finalize_poll_seconds: float,
    image_builder_url: str | None,
    builder_source_commit: str | None,
    context_path: str | None,
    token: str | None,
    token_command: str | None,
    auth_login_command: str | None,
    auth_refresh_seconds: int,
    workers: int,
    attempts: int,
    retry_sleep: float,
) -> None:
    """Compatibility projection for finalize-standard-benchmark-catalog."""
    del (
        image_builder_url,
        builder_source_commit,
        context_path,
        token,
        token_command,
        auth_login_command,
        auth_refresh_seconds,
        workers,
        attempts,
        retry_sleep,
    )
    unsupported = bool(
        resolve_only or finalize_resolved_only or skip_error_substring or skip_slug_substring or limit is not None
    )
    if unsupported:
        raise click.ClickException(
            "selection and direct image-resolution modes are no longer implemented by this "
            "helper; supply prebuilt image results or use the server-managed finalize path"
        )
    images = _merge_import_images(tuple(f"@{path}" for path in prebuilt_image_results))
    payload = load_benchmark_manifest(manifest)
    if prebuilt_results_only:
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise click.ClickException("benchmark import manifest must contain a tasks list")
        missing = sorted({str(task["slug"]) for task in tasks} - set(images))
        if missing:
            raise click.ClickException(f"--prebuilt-results-only is missing image results for {len(missing)} tasks")
    try:
        import_id = import_id_from_legacy_state(upload_state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if images:
        request(
            ctx.obj["client"],
            "POST",
            f"/benchmark-imports/{import_id}/images",
            json={"images": images},
        )
    data = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    if only_failed and data.get("status") == "failed":
        data = request(ctx.obj["client"], "POST", f"/benchmark-imports/{import_id}/retry")
    _upload_import_packs(ctx, data, output_root)
    request(ctx.obj["client"], "POST", f"/benchmark-imports/{import_id}/prepare")
    data = _wait_for_import(
        ctx,
        import_id,
        timeout=finalize_timeout,
        poll_interval=finalize_poll_seconds,
    )
    write_legacy_import_state(state, data, phase="finalize")
    emit(data, ctx.obj["json"], _import_summary(data))
    if data.get("status") == "failed":
        ctx.exit(1)


@benchmark_import_group.command("compat-publish", hidden=True)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--finalize-state", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--description", required=True)
@click.option("--visibility", type=click.Choice(["private", "team", "org", "public"]), default=None)
@click.option("--smoke-size", type=click.IntRange(min=1), default=None)
@click.option("--attempts", type=int, default=6, hidden=True)
@click.pass_context
def benchmark_import_compat_publish(
    ctx: click.Context,
    manifest: Path,
    finalize_state: Path,
    description: str,
    visibility: str | None,
    smoke_size: int | None,
    attempts: int,
) -> None:
    """Compatibility projection for publish-standard-benchmark-catalog."""
    del manifest, description, attempts
    try:
        import_id = import_id_from_legacy_state(finalize_state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    current = request(ctx.obj["client"], "GET", f"/benchmark-imports/{import_id}")
    if visibility is not None and current.get("visibility") != visibility:
        raise click.ClickException(f"import {import_id} visibility is {current.get('visibility')}, not {visibility}")
    data = request(
        ctx.obj["client"],
        "POST",
        f"/benchmark-imports/{import_id}/publish",
        json={"smoke_size": smoke_size},
    )
    emit(data, ctx.obj["json"], _import_summary(data))


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(load_arg(raw))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise click.ClickException(f"{label} must be a JSON object")
    return parsed


def _parse_yaml_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is locked via uvicorn[standard].
        raise click.ClickException("YAML input requires PyYAML to be installed") from exc
    try:
        parsed = yaml.safe_load(load_arg(raw))
    except yaml.YAMLError as exc:
        raise click.ClickException(f"{label} is not valid YAML: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise click.ClickException(f"{label} must be a YAML mapping")
    return parsed


def _parse_config(
    config: str | None,
    config_yaml: str | None,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if config and config_yaml:
        raise click.ClickException("provide only one of --config or --config-yaml")
    if config:
        return _parse_json_object(config, label="--config")
    if config_yaml:
        return _parse_yaml_object(config_yaml, label="--config-yaml")
    if required:
        raise click.ClickException("provide at least one field to update")
    return None


def _require_patch(body: dict[str, object]) -> None:
    if not body:
        raise click.ClickException("provide at least one field to update")


def _event_line(event: dict[str, Any]) -> str:
    return f"{event.get('at', '')}  {event.get('type')}  {event.get('status', '')} {event.get('detail') or ''}".rstrip()


def _parse_sse_data(event_name: str, raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"stream event {event_name!r} carried invalid JSON") from exc
    return data if isinstance(data, dict) else {"value": data}


def _wait_for_evaluation(
    ctx: click.Context,
    evaluation_id: str,
    *,
    interval: float,
    timeout: float | None,
) -> None:
    deadline = time.monotonic() + timeout if timeout is not None else None
    last_status = None
    try:
        while True:
            data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}")
            status = str(data.get("status") or "")
            if not ctx.obj["json"] and status != last_status:
                detail = f" ({data['status_detail']})" if data.get("status_detail") else ""
                click.echo(f"{evaluation_id}: {status}{detail}")
            last_status = status
            if status in TERMINAL_STATUSES:
                _emit_wait_result(ctx, data)
                if status in SUCCESS_STATUSES:
                    return
                raise click.exceptions.Exit(1)
            if deadline is not None and time.monotonic() >= deadline:
                raise click.ClickException(f"timed out waiting for {evaluation_id}")
            time.sleep(interval)
    except KeyboardInterrupt as exc:
        click.echo("interrupted", err=True)
        raise click.exceptions.Exit(130) from exc


def _emit_wait_result(ctx: click.Context, data: dict[str, Any]) -> None:
    if ctx.obj["json"]:
        emit(data, True, [])
        return
    for line in _evaluation_summary(data):
        click.echo(line)
    links = data.get("links") if isinstance(data.get("links"), dict) else {}
    if links:
        for label in ("artifacts", "provenance", "sbom", "archive"):
            if links.get(label):
                click.echo(f"  {label}: {links[label]}")


# ---------- Switchyard ----------------------------------------------------


@cli.group()
def switchyard() -> None:
    """Publish and manage Switchyard runtime images."""


def _git_head(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException("--source-ref is required when --context-dir is not inside a git checkout") from exc
    return proc.stdout.strip()


def _git_dirty(path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException("--context-dir must be inside a git checkout") from exc
    return bool(proc.stdout.strip())


def _clean_relative_path(value: str, *, label: str) -> str:
    normalized = value.strip() or "."
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise click.ClickException(f"{label} must stay within --context-dir")
    return path.as_posix()


def _default_switchyard_dockerfile_path(
    context_dir: Path,
    *,
    context_path: str,
    requested: str | None,
) -> str:
    if requested:
        return _clean_relative_path(requested, label="--dockerfile-path")
    context_dockerfile = "Dockerfile" if context_path == "." else f"{context_path.rstrip('/')}/Dockerfile"
    official_dockerfile = "benchmark/switchyard-rust-server.Dockerfile"
    if (
        context_path == "."
        and not (context_dir / context_dockerfile).is_file()
        and (context_dir / official_dockerfile).is_file()
    ):
        return official_dockerfile
    return context_dockerfile


@switchyard.command("publish")
@click.option(
    "--source-project",
    default="NVIDIA-NeMo/Switchyard",
    show_default=True,
    help="GitHub Switchyard project path to publish from.",
)
@click.option(
    "--source-ref",
    default=None,
    help="Full 40-character GitHub Switchyard commit SHA; defaults to --context-dir HEAD.",
)
@click.option(
    "--context-path",
    default=".",
    show_default=True,
    help="Build context path inside --context-dir.",
)
@click.option(
    "--dockerfile-path",
    default=None,
    help=(
        "Dockerfile path inside --context-dir. Defaults to Dockerfile, or "
        "benchmark/switchyard-rust-server.Dockerfile for the GitHub Switchyard repo."
    ),
)
@click.option(
    "--context-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Local Switchyard build-context directory checked out at --source-ref.",
)
@click.option(
    "--profile-name",
    default=None,
    help="Optional name for the reusable Switchyard config profile.",
)
@click.option("--wait/--no-wait", default=True, help="Wait for the publication to finish.")
@click.option(
    "--poll-interval",
    type=click.FloatRange(min=1.0),
    default=5.0,
    show_default=True,
    help="Seconds between asynchronous publication status checks.",
)
@click.option(
    "--timeout",
    type=click.FloatRange(min=1.0),
    default=2400.0,
    show_default=True,
    help="Maximum seconds to wait for an asynchronous publication.",
)
@click.pass_context
def switchyard_publish(
    ctx: click.Context,
    source_project: str,
    source_ref: str | None,
    context_path: str,
    dockerfile_path: str | None,
    context_dir: Path,
    profile_name: str | None,
    wait: bool,
    poll_interval: float,
    timeout: float,
) -> None:
    """Build/sign a Switchyard commit and create or reuse a run profile."""
    context_path = _clean_relative_path(context_path, label="--context-path")
    context_head = _git_head(context_dir).lower()
    source_ref = source_ref.lower() if source_ref else context_head
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_ref):
        raise click.ClickException("--source-ref must be a full 40-character commit SHA")
    if source_ref != context_head:
        raise click.ClickException("--source-ref must match --context-dir HEAD")
    if _git_dirty(context_dir):
        raise click.ClickException("--context-dir must be clean to publish a committed source ref")
    dockerfile_path = _default_switchyard_dockerfile_path(
        context_dir,
        context_path=context_path,
        requested=dockerfile_path,
    )
    try:
        archive = archive_context_directory(context_dir, dockerfile_path=dockerfile_path)
    except BuildError as exc:
        raise click.ClickException(str(exc)) from exc
    form: dict[str, str] = {
        "source_project": source_project,
        "source_ref": source_ref.lower(),
        "context_path": context_path,
        "dockerfile_path": dockerfile_path,
    }
    if profile_name:
        form["profile_name"] = profile_name
    started = time.monotonic()
    data = request(
        ctx.obj["client"],
        "POST",
        "/switchyard/publish",
        data=form,
        files={"context": ("context.tar.gz", archive, "application/gzip")},
    )
    build_id = str(data.get("build_id") or "")
    if build_id and not wait:
        emit(
            data,
            ctx.obj["json"],
            [f"switchyard publication {build_id}", f"  status: {data.get('status')}"],
        )
        return
    while build_id:
        status = str(data.get("status") or "")
        if status == "succeeded":
            result = data.get("result")
            if not isinstance(result, dict):
                raise click.ClickException("completed Switchyard publication has no profile")
            data = result
            break
        if status == "failed":
            raise click.ClickException(str(data.get("build_error") or "Switchyard publication failed"))
        if time.monotonic() - started >= timeout:
            raise click.ClickException(
                f"timed out waiting for Switchyard publication {build_id}; the Cloud Build "
                f"continues and remains available at /v1/switchyard/publishes/{build_id}"
            )
        time.sleep(poll_interval)
        data = request(ctx.obj["client"], "GET", f"/switchyard/publishes/{build_id}")
    emit(
        data,
        ctx.obj["json"],
        [
            f"switchyard profile {data.get('profile_id')}",
            f"  name:       {data.get('profile_name')}",
            f"  source:     {data.get('source_project')}@{data.get('source_ref')}",
            f"  context:    {data.get('context_path')} ({data.get('context_hash')})",
            f"  dockerfile: {data.get('dockerfile_path')}",
            f"  image:      {data.get('image_ref')}",
            f"  digest:     {data.get('image_digest')}",
            f"  reused:     {data.get('reused_profile')}",
        ],
    )


# ---------- tasks ----------------------------------------------------


@cli.group()
def task() -> None:
    """Create and upload tasks."""


@task.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option("--slug", default=None, help="Globally unique handle.")
@click.option("--description", default=None)
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.option(
    "--tarball",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional: upload this tarball immediately after create.",
)
@click.pass_context
def task_create(
    ctx: click.Context,
    name: str,
    slug: str | None,
    description: str | None,
    visibility: str | None,
    tarball: Path | None,
) -> None:
    """Create a task; prints the presigned upload URL for its tarball."""
    body: dict[str, object] = {"name": name}
    if slug:
        body["slug"] = slug
    if description:
        body["description"] = description
    if visibility:
        body["visibility"] = visibility

    data = request(ctx.obj["client"], "POST", "/tasks", json=body)
    upload = data.get("upload", {})
    emit(
        data,
        ctx.obj["json"],
        [
            f"task {data['id']}",
            f"  revision:   {data.get('revision')}",
            f"  status:     {data.get('status')}",
            f"  upload url: {upload.get('url')}",
        ],
    )
    if tarball:
        upload_file(ctx.obj["client"], upload, tarball)
        if not ctx.obj["json"]:
            click.echo(f"  uploaded:   {tarball}")


@task.command("list")
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--limit", type=int, default=None, help="Page size (1-100, default 20).")
@click.option("--order", type=click.Choice(ORDER), default=None, help="Sort order by creation time.")
@click.pass_context
def task_list(
    ctx: click.Context,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List live tasks."""
    params: dict[str, object] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if order:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/tasks", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda b: (f"{b['id']}  {str(b.get('visibility')):8}  {b.get('slug')}  {b.get('name')}").rstrip(),
    )


@task.command("update")
@click.argument("task_id")
@click.option("--name", default=None, help="New display name.")
@click.option("--slug", default=None, help="New globally unique slug.")
@click.option("--description", default=None, help="New description.")
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.pass_context
def task_update(
    ctx: click.Context,
    task_id: str,
    name: str | None,
    slug: str | None,
    description: str | None,
    visibility: str | None,
) -> None:
    """Update mutable task metadata."""
    body: dict[str, object] = {}
    for field, value in (
        ("name", name),
        ("slug", slug),
        ("description", description),
        ("visibility", visibility),
    ):
        if value is not None:
            body[field] = value
    _require_patch(body)
    data = request(ctx.obj["client"], "PATCH", f"/tasks/{task_id}", json=body)
    emit(data, ctx.obj["json"], _task_summary(data))


@task.command("delete")
@click.argument("task_id")
@click.pass_context
def task_delete(ctx: click.Context, task_id: str) -> None:
    """Soft-delete a task."""
    data = request(ctx.obj["client"], "DELETE", f"/tasks/{task_id}")
    emit(data, ctx.obj["json"], [f"deleted task {data.get('id', task_id)}"])


@task.command("upload")
@click.argument("task_id")
@click.argument("tarball_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def task_upload(ctx: click.Context, task_id: str, tarball_path: Path) -> None:
    """Upload a tarball as a new revision of an existing task.

    The presigned URL only comes from a POST: a new tarball means a new
    revision, so this mints one via POST /tasks/{id}/revisions and PUTs
    to the upload block it returns.
    """
    data = request(ctx.obj["client"], "POST", f"/tasks/{task_id}/revisions")
    upload = data.get("upload")
    if not upload:
        raise click.ClickException(f"task {task_id} revision response carried no upload block")
    upload_file(ctx.obj["client"], upload, tarball_path)
    if ctx.obj["json"]:
        emit(
            {
                "task_id": task_id,
                "revision": data.get("revision"),
                "path": str(tarball_path),
                "uploaded": True,
            },
            True,
            [],
        )
    else:
        click.echo(f"uploaded {tarball_path} to {task_id} (revision {data.get('revision')})")


@task.command("finalize")
@click.argument("task_id")
@click.option(
    "--image-ref",
    default=None,
    help="Reuse this already signed image for the revision. Omit to build and sign "
    "the uploaded task pack through the configured finalize backend.",
)
@click.option(
    "--image-digest",
    default=None,
    help="Immutable digest for --image-ref. Required by hosted policy.",
)
@click.pass_context
def task_finalize(
    ctx: click.Context,
    task_id: str,
    image_ref: str | None,
    image_digest: str | None,
) -> None:
    """Finalize the latest revision so it can be evaluated.

    By default this finalizes the task pack already uploaded to scaled-evals.
    Hosted deployments use their managed builder (the image-builder service,
    Cloud Build + GAR on GKE); local deployments without one keep the
    BuildKit fallback. Pass `--image-ref` with `--image-digest` to reuse an
    already signed or otherwise approved image.
    """
    if image_digest and not image_ref:
        raise click.ClickException("--image-digest requires --image-ref")
    body: dict[str, object] = {
        k: v
        for k, v in (
            ("image_ref", image_ref),
            ("image_digest", image_digest),
        )
        if v
    }
    kwargs = {"json": body} if body else {}
    data = request(ctx.obj["client"], "POST", f"/tasks/{task_id}/finalize", **kwargs)
    emit(
        data,
        ctx.obj["json"],
        [
            f"task {data.get('id', task_id)}",
            f"  revision: {data.get('revision')}",
            f"  status:   {data.get('status')}",
        ],
    )


@task.command("get")
@click.argument("task_id")
@click.pass_context
def task_get(ctx: click.Context, task_id: str) -> None:
    """Show a task and its latest revision's build status."""
    data = request(ctx.obj["client"], "GET", f"/tasks/{task_id}")
    emit(data, ctx.obj["json"], _task_summary(data))


@task.command("get-by-slug")
@click.argument("slug")
@click.pass_context
def task_get_by_slug(ctx: click.Context, slug: str) -> None:
    """Resolve a globally unique live task by slug."""
    data = request(ctx.obj["client"], "GET", f"/tasks/by-slug/{slug}")
    emit(data, ctx.obj["json"], _task_summary(data))


def _task_summary(data: dict[str, object]) -> list[str]:
    summary = [
        f"task {data['id']}",
        f"  name:       {data.get('name')}",
        f"  slug:       {data.get('slug')}",
        f"  visibility: {data.get('visibility')}",
        f"  revision:   {data.get('revision', data.get('current_revision'))}",
        f"  status:     {data.get('status')}",
    ]
    extra = (("image", "image_ref"), ("digest", "image_digest"), ("error", "build_error"))
    for label, key in extra:
        if data.get(key):
            summary.append(f"  {label + ':':11} {data[key]}")
    return summary


# ---------- benchmarks ----------------------------------------------------


def _parse_task_refs(specs: tuple[str, ...]) -> list[dict[str, object]]:
    """Parse repeatable ``--task task_id[:revision]`` into membership refs.

    ``task_abc`` floats to the task's latest revision (null pin); ``task_abc:2``
    pins task revision 2.
    """
    refs: list[dict[str, object]] = []
    for spec in specs:
        task_id, _, rev = spec.partition(":")
        if not task_id:
            raise click.ClickException(f"invalid --task value: {spec!r}")
        ref: dict[str, object] = {"task_id": task_id}
        if rev:
            try:
                ref["task_revision"] = int(rev)
            except ValueError as exc:
                raise click.ClickException(f"invalid task revision in {spec!r}: must be an integer") from exc
        refs.append(ref)
    return refs


def _benchmark_summary(data: dict[str, Any]) -> list[str]:
    summary = [
        f"benchmark {data['id']}",
        f"  name:       {data.get('name')}",
        f"  slug:       {data.get('slug')}",
        f"  visibility: {data.get('visibility')}",
        f"  qualification: {data.get('qualification_status')}",
        f"  revision:   {data.get('revision', data.get('current_revision'))}",
    ]
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        summary.append(f"  tasks:      {len(tasks)}")
        for t in tasks:
            rev = t.get("task_revision")
            summary.append(f"    - {t.get('task_id')}{f' @rev{rev}' if rev is not None else ''}")
    return summary


@cli.group()
def benchmark() -> None:
    """Create and manage benchmarks (revisioned collections of tasks)."""


@benchmark.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option("--slug", default=None, help="Globally unique handle.")
@click.option("--description", default=None)
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Member task as task_id[:revision] (omit :revision to float to latest). Repeatable.",
)
@click.pass_context
def benchmark_create(
    ctx: click.Context,
    name: str,
    slug: str | None,
    description: str | None,
    visibility: str | None,
    tasks: tuple[str, ...],
) -> None:
    """Create a benchmark and its first revision from the given member tasks."""
    body: dict[str, object] = {"name": name, "tasks": _parse_task_refs(tasks)}
    if slug:
        body["slug"] = slug
    if description:
        body["description"] = description
    if visibility:
        body["visibility"] = visibility
    data = request(ctx.obj["client"], "POST", "/benchmarks", json=body)
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.group("variant")
def benchmark_variant() -> None:
    """Derive metadata-only benchmark variants (ops overrides, same task pins)."""


@benchmark_variant.command("create")
@click.argument("benchmark_id")
@click.option("--name", required=True, help="Display name for the variant.")
@click.option("--slug", default=None, help="Globally unique handle.")
@click.option("--description", default=None)
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.option(
    "--from-revision",
    type=int,
    default=None,
    help="Base revision to copy (default: current revision).",
)
@click.option(
    "--agent-timeout-floor-sec",
    type=int,
    required=True,
    help="Raise staged [agent].timeout_sec to max(original, this floor).",
)
@click.pass_context
def benchmark_variant_create(
    ctx: click.Context,
    benchmark_id: str,
    name: str,
    slug: str | None,
    description: str | None,
    visibility: str | None,
    from_revision: int | None,
    agent_timeout_floor_sec: int,
) -> None:
    """Create a variant that copies member pins and stores an operational policy."""
    body: dict[str, object] = {
        "name": name,
        "operational_policy": {"agent_timeout_floor_sec": agent_timeout_floor_sec},
    }
    if slug:
        body["slug"] = slug
    if description:
        body["description"] = description
    if visibility:
        body["visibility"] = visibility
    if from_revision is not None:
        body["from_revision"] = from_revision
    data = request(ctx.obj["client"], "POST", f"/benchmarks/{benchmark_id}/variants", json=body)
    summary = _benchmark_summary(data)
    derived = data.get("derived_from")
    if isinstance(derived, dict):
        summary.append(f"  derived_from: {derived.get('benchmark_id')} @rev{derived.get('revision')}")
    policy = data.get("operational_policy")
    if isinstance(policy, dict) and policy.get("agent_timeout_floor_sec") is not None:
        summary.append(f"  agent_timeout_floor_sec: {policy['agent_timeout_floor_sec']}")
    emit(data, ctx.obj["json"], summary)


@benchmark.command("revise")
@click.argument("benchmark_id")
@click.option("--description", default=None)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Member task as task_id[:revision]. Repeatable; replaces the member set.",
)
@click.pass_context
def benchmark_revise(ctx: click.Context, benchmark_id: str, description: str | None, tasks: tuple[str, ...]) -> None:
    """Snapshot a new immutable revision with a new member set."""
    body: dict[str, object] = {"tasks": _parse_task_refs(tasks)}
    if description:
        body["description"] = description
    data = request(ctx.obj["client"], "POST", f"/benchmarks/{benchmark_id}/revisions", json=body)
    emit(
        data,
        ctx.obj["json"],
        [f"benchmark {data.get('id', benchmark_id)}", f"  revision: {data.get('revision')}"],
    )


@benchmark.command("list")
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--limit", type=int, default=None, help="Page size (1-100, default 20).")
@click.option("--order", type=click.Choice(ORDER), default=None, help="Sort order by creation time.")
@click.pass_context
def benchmark_list(ctx: click.Context, cursor: str | None, limit: int | None, order: str | None) -> None:
    """List live benchmarks."""
    params: dict[str, object] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if order:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/benchmarks", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda b: (f"{b['id']}  {str(b.get('visibility')):8}  {b.get('slug')}  {b.get('name')}").rstrip(),
    )


@benchmark.command("get")
@click.argument("benchmark_id")
@click.pass_context
def benchmark_get(ctx: click.Context, benchmark_id: str) -> None:
    """Show a benchmark and its latest revision's member tasks."""
    data = request(ctx.obj["client"], "GET", f"/benchmarks/{benchmark_id}")
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.command("get-by-slug")
@click.argument("slug")
@click.pass_context
def benchmark_get_by_slug(ctx: click.Context, slug: str) -> None:
    """Resolve a globally unique benchmark by slug."""
    data = request(ctx.obj["client"], "GET", f"/benchmarks/by-slug/{slug}")
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.command("tasks")
@click.argument("benchmark_id")
@click.option("--revision", type=int, default=None, help="Revision to list (default: latest).")
@click.option("--limit", type=int, default=20, show_default=True, help="Rows to fetch.")
@click.option("--cursor", default=None, help="Cursor from the previous page.")
@click.pass_context
def benchmark_tasks(
    ctx: click.Context,
    benchmark_id: str,
    revision: int | None,
    limit: int,
    cursor: str | None,
) -> None:
    """List the member tasks of a benchmark revision."""
    params: dict[str, object] = {"limit": limit}
    if revision is not None:
        params["revision"] = revision
    if cursor:
        params["cursor"] = cursor
    data = request(ctx.obj["client"], "GET", f"/benchmarks/{benchmark_id}/tasks", params=params)
    emit_list(data, ctx.obj["json"], _member_line)


def _member_line(member: dict[str, object]) -> str:
    rev = member.get("task_revision")
    pin = f"@rev{rev}" if rev is not None else "@latest"
    return f"{member.get('position')}  {member.get('task_id')}  {pin}"


@benchmark.command("update")
@click.argument("benchmark_id")
@click.option("--name", default=None, help="New display name.")
@click.option("--slug", default=None, help="New globally unique slug.")
@click.option("--description", default=None, help="New description.")
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.pass_context
def benchmark_update(
    ctx: click.Context,
    benchmark_id: str,
    name: str | None,
    slug: str | None,
    description: str | None,
    visibility: str | None,
) -> None:
    """Update mutable benchmark metadata."""
    body: dict[str, object] = {}
    for field, value in (
        ("name", name),
        ("slug", slug),
        ("description", description),
        ("visibility", visibility),
    ):
        if value is not None:
            body[field] = value
    _require_patch(body)
    data = request(ctx.obj["client"], "PATCH", f"/benchmarks/{benchmark_id}", json=body)
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.command("qualify")
@click.argument("benchmark_id")
@click.option("--status", type=click.Choice(["qualified", "rejected"]), required=True)
@click.option("--evidence", default=None, help="Non-secret qualification JSON or @file.json.")
@click.pass_context
def benchmark_qualify(ctx: click.Context, benchmark_id: str, status: str, evidence: str | None) -> None:
    """Admin-only: record technical qualification evidence."""
    body = {
        "status": status,
        "evidence": (_parse_json_object(evidence, label="--evidence") if evidence is not None else {}),
    }
    data = request(ctx.obj["client"], "POST", f"/benchmarks/{benchmark_id}/qualification", json=body)
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.command("promote")
@click.argument("benchmark_id")
@click.pass_context
def benchmark_promote(ctx: click.Context, benchmark_id: str) -> None:
    """Admin-only: promote a qualified benchmark into the shared public catalog."""
    data = request(ctx.obj["client"], "POST", f"/benchmarks/{benchmark_id}/promote", json={})
    emit(data, ctx.obj["json"], _benchmark_summary(data))


@benchmark.command("delete")
@click.argument("benchmark_id")
@click.pass_context
def benchmark_delete(ctx: click.Context, benchmark_id: str) -> None:
    """Soft-delete a benchmark."""
    data = request(ctx.obj["client"], "DELETE", f"/benchmarks/{benchmark_id}")
    emit(data, ctx.obj["json"], [f"deleted benchmark {data.get('id', benchmark_id)}"])


# ---------- credentials ---------------------------------------------------


@cli.group()
def credential() -> None:
    """Register BYOK credentials (write-once secrets)."""


@credential.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option(
    "--provider",
    type=click.Choice(CREDENTIAL_PROVIDER),
    required=True,
    help="Secret category; model providers carry a key, nmp/openshift carry yaml or key payloads.",
)
@click.option("--key", default=None, help="Single-string secret (model API key).")
@click.option("--yaml", "yaml_", default=None, help="Structured secret blob, inline or @file.")
@click.pass_context
def credential_create(ctx: click.Context, name: str, provider: str, key: str | None, yaml_: str | None) -> None:
    """Register a credential. The secret is never echoed back."""
    if bool(key) == bool(yaml_):
        raise click.ClickException("provide exactly one of --key or --yaml")

    body: dict[str, object] = {"name": name, "provider": provider}
    if key:
        body["key"] = key
    else:
        assert yaml_ is not None  # The exclusive key/yaml check above guarantees this.
        body["yaml"] = load_arg(yaml_)

    data = request(ctx.obj["client"], "POST", "/credentials", json=body)
    emit(
        data,
        ctx.obj["json"],
        [
            f"credential {data['id']}",
            f"  name:        {data.get('name')}",
            f"  provider:    {data.get('provider')}",
            f"  kind:        {data.get('payload_kind')}",
            f"  fingerprint: {data.get('fingerprint')}",
        ],
    )


@credential.command("list")
@click.option("--provider", type=click.Choice(CREDENTIAL_PROVIDER), default=None)
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--limit", type=int, default=None, help="Page size (1-100, default 20).")
@click.option("--order", type=click.Choice(ORDER), default=None, help="Sort order by creation time.")
@click.pass_context
def credential_list(
    ctx: click.Context,
    provider: str | None,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List caller-owned credential metadata; never includes plaintext secrets."""
    params: dict[str, object] = {}
    if provider:
        params["provider"] = provider
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if order:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/credentials", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda c: (f"{c['id']}  {str(c.get('provider')):10}  {c.get('name')}  {c.get('fingerprint') or ''}").rstrip(),
    )


@credential.command("get")
@click.argument("credential_id")
@click.pass_context
def credential_get(ctx: click.Context, credential_id: str) -> None:
    """Show credential metadata; plaintext secrets are never available."""
    data = request(ctx.obj["client"], "GET", f"/credentials/{credential_id}")
    emit(data, ctx.obj["json"], _credential_summary(data))


@credential.command("verify")
@click.argument("credential_id")
@click.pass_context
def credential_verify(ctx: click.Context, credential_id: str) -> None:
    """Ask the control plane to validate a stored credential upstream."""
    data = request(ctx.obj["client"], "POST", f"/credentials/{credential_id}/verify")
    verified = data.get("verified")
    emit(
        data,
        ctx.obj["json"],
        [
            f"credential {data.get('id', credential_id)}",
            f"  verified: {verified if verified is not None else 'inconclusive'}",
            f"  reason:   {data.get('reason') or ''}",
        ],
    )


@credential.command("rename")
@click.argument("credential_id")
@click.option("--name", required=True, help="New display name.")
@click.pass_context
def credential_rename(ctx: click.Context, credential_id: str, name: str) -> None:
    """Rename a credential without changing its secret."""
    data = request(ctx.obj["client"], "PATCH", f"/credentials/{credential_id}", json={"name": name})
    emit(data, ctx.obj["json"], _credential_summary(data))


@credential.command("rotate")
@click.argument("credential_id")
@click.option("--key", default=None, help="Replacement single-string secret.")
@click.option(
    "--yaml",
    "yaml_",
    default=None,
    help="Replacement structured secret, inline or @file.",
)
@click.pass_context
def credential_rotate(ctx: click.Context, credential_id: str, key: str | None, yaml_: str | None) -> None:
    """Replace a credential secret in place. The new secret is never echoed."""
    if bool(key) == bool(yaml_):
        raise click.ClickException("provide exactly one of --key or --yaml")
    body: dict[str, object] = {}
    if key:
        body["key"] = key
    else:
        assert yaml_ is not None  # The exclusive key/yaml check above guarantees this.
        body["yaml"] = load_arg(yaml_)
    data = request(ctx.obj["client"], "POST", f"/credentials/{credential_id}/rotate", json=body)
    emit(data, ctx.obj["json"], _credential_summary(data))


@credential.command("delete")
@click.argument("credential_id")
@click.pass_context
def credential_delete(ctx: click.Context, credential_id: str) -> None:
    """Revoke a credential."""
    data = request(ctx.obj["client"], "DELETE", f"/credentials/{credential_id}")
    emit(data, ctx.obj["json"], [f"deleted credential {data.get('id', credential_id)}"])


def _credential_summary(data: dict[str, object]) -> list[str]:
    summary = [
        f"credential {data['id']}",
        f"  name:        {data.get('name')}",
        f"  provider:    {data.get('provider')}",
        f"  kind:        {data.get('payload_kind')}",
        f"  fingerprint: {data.get('fingerprint')}",
    ]
    if data.get("created_at"):
        summary.append(f"  created:     {data['created_at']}")
    return summary


# ---------- config profiles -----------------------------------------------


@cli.group(name="config-profile")
def config_profile() -> None:
    """Create reusable config profiles."""


@config_profile.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option("--type", "type_", type=click.Choice(CONFIG_PROFILE_TYPE), required=True)
@click.option("--config", default=None, help="Config as a JSON string or @file.json.")
@click.option("--config-yaml", default=None, help="Config as a YAML mapping or @file.yaml.")
@click.pass_context
def config_profile_create(
    ctx: click.Context,
    name: str,
    type_: str,
    config: str | None,
    config_yaml: str | None,
) -> None:
    """Create a config profile of the given type."""
    body: dict[str, object] = {"name": name, "type": type_}
    parsed_config = _parse_config(config, config_yaml, required=False)
    if parsed_config is not None:
        body["config"] = parsed_config

    data = request(ctx.obj["client"], "POST", "/config-profiles", json=body)
    emit(
        data,
        ctx.obj["json"],
        [
            f"config-profile {data['id']}",
            f"  name: {data.get('name')}",
            f"  type: {data.get('type')}",
        ],
    )


@config_profile.command("list")
@click.option("--type", "type_", type=click.Choice(CONFIG_PROFILE_TYPE), default=None)
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--limit", type=int, default=None, help="Page size (1-100, default 20).")
@click.option("--order", type=click.Choice(ORDER), default=None, help="Sort order by creation time.")
@click.pass_context
def config_profile_list(
    ctx: click.Context,
    type_: str | None,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List reusable non-secret config profiles."""
    params: dict[str, object] = {}
    if type_:
        params["type"] = type_
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if order:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/config-profiles", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda p: f"{p['id']}  {str(p.get('type')):10}  {p.get('name')}",
    )


@config_profile.command("get")
@click.argument("profile_id")
@click.pass_context
def config_profile_get(ctx: click.Context, profile_id: str) -> None:
    """Show one non-secret config profile."""
    data = request(ctx.obj["client"], "GET", f"/config-profiles/{profile_id}")
    emit(
        data,
        ctx.obj["json"],
        [
            f"config-profile {data['id']}",
            f"  name: {data.get('name')}",
            f"  type: {data.get('type')}",
        ],
    )


@config_profile.command("update")
@click.argument("profile_id")
@click.option("--name", default=None, help="New display name.")
@click.option("--config", default=None, help="Replacement config as a JSON string or @file.json.")
@click.option(
    "--config-yaml",
    default=None,
    help="Replacement config as a YAML mapping or @file.yaml.",
)
@click.pass_context
def config_profile_update(
    ctx: click.Context,
    profile_id: str,
    name: str | None,
    config: str | None,
    config_yaml: str | None,
) -> None:
    """Update a config profile's name and/or config. Type is immutable."""
    body: dict[str, object] = {}
    if name is not None:
        body["name"] = name
    parsed_config = _parse_config(config, config_yaml, required=False)
    if parsed_config is not None:
        body["config"] = parsed_config
    _require_patch(body)
    data = request(ctx.obj["client"], "PATCH", f"/config-profiles/{profile_id}", json=body)
    emit(
        data,
        ctx.obj["json"],
        [
            f"config-profile {data['id']}",
            f"  name: {data.get('name')}",
            f"  type: {data.get('type')}",
        ],
    )


@config_profile.command("delete")
@click.argument("profile_id")
@click.pass_context
def config_profile_delete(ctx: click.Context, profile_id: str) -> None:
    """Soft-delete a config profile."""
    data = request(ctx.obj["client"], "DELETE", f"/config-profiles/{profile_id}")
    emit(data, ctx.obj["json"], [f"deleted config-profile {data.get('id', profile_id)}"])


# ---------- evaluations ---------------------------------------------------


@cli.group(name="agent-bundle")
def agent_bundle() -> None:
    """Register and discover immutable agent runtime bundles."""


def _agent_bundle_summary(data: dict[str, object]) -> list[str]:
    return [
        f"agent-bundle {data['id']}",
        f"  name:          {data.get('bundle_name')}",
        f"  agent:         {data.get('agent_name')}@{data.get('agent_version')}",
        f"  visibility:    {data.get('visibility')}",
        f"  qualification: {data.get('qualification_status')}",
        f"  runtime image: {data.get('image_ref')}",
        f"  provenance:    {data.get('image_digest')}",
    ]


@agent_bundle.command("create")
@click.option("--bundle-name", required=True, help="Owner-scoped catalog label.")
@click.option("--agent-name", required=True, help="Identity from the signed descriptor.")
@click.option("--agent-version", required=True, help="Exact descriptor version.")
@click.option("--image-ref", required=True, help="Signed runtime tag submitted to Kubernetes.")
@click.option("--image-digest", required=True, help="Immutable REPOSITORY@sha256:... image.")
@click.option("--entrypoint", required=True, help="Bundle-relative executable, e.g. bin/codex.")
@click.option("--source-lock-digest", required=True, help="Bundle source-lock sha256 digest.")
@click.option("--fingerprint", required=True, help="Bundle content fingerprint sha256 digest.")
@click.option(
    "--builder-profile",
    default=None,
    help="Exact bundle builder profile; API default is node22-npm-v1.",
)
@click.option("--metadata", default=None, help="Optional non-secret JSON object or @file.json.")
@click.pass_context
def agent_bundle_create(
    ctx: click.Context,
    bundle_name: str,
    agent_name: str,
    agent_version: str,
    image_ref: str,
    image_digest: str,
    entrypoint: str,
    source_lock_digest: str,
    fingerprint: str,
    builder_profile: str | None,
    metadata: str | None,
) -> None:
    """Register an immutable private bundle owned by the current user."""
    body: dict[str, object] = {
        "bundle_name": bundle_name,
        "agent_name": agent_name,
        "agent_version": agent_version,
        "image_ref": image_ref,
        "image_digest": image_digest,
        "entrypoint": entrypoint,
        "source_lock_digest": source_lock_digest,
        "fingerprint": fingerprint,
    }
    if builder_profile is not None:
        body["builder_profile"] = builder_profile
    if metadata is not None:
        body["metadata"] = _parse_json_object(metadata, label="--metadata")
    data = request(ctx.obj["client"], "POST", "/agent-bundles", json=body)
    emit(data, ctx.obj["json"], _agent_bundle_summary(data))


@agent_bundle.command("list")
@click.option("--mine", is_flag=True, help="Show only bundles owned by the current user.")
@click.option("--visibility", type=click.Choice(["private", "public"]), default=None)
@click.option("--cursor", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--order", type=click.Choice(ORDER), default=None)
@click.pass_context
def agent_bundle_list(
    ctx: click.Context,
    mine: bool,
    visibility: str | None,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List your private bundles and shared public bundles."""
    params: dict[str, object] = {"mine": mine}
    if visibility is not None:
        params["visibility"] = visibility
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    if order is not None:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/agent-bundles", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda b: (
            f"{b['id']}  {b['bundle_name']}  {b['agent_name']}@{b['agent_version']}  "
            f"{b['visibility']}  {b['qualification_status']}"
        ),
    )


@agent_bundle.command("get")
@click.argument("bundle_id")
@click.pass_context
def agent_bundle_get(ctx: click.Context, bundle_id: str) -> None:
    """Show one accessible bundle and its immutable image identity."""
    data = request(ctx.obj["client"], "GET", f"/agent-bundles/{bundle_id}")
    emit(data, ctx.obj["json"], _agent_bundle_summary(data))


@agent_bundle.command("qualify")
@click.argument("bundle_id")
@click.option("--status", type=click.Choice(["qualified", "rejected"]), required=True)
@click.option("--evidence", default=None, help="Non-secret qualification JSON or @file.json.")
@click.pass_context
def agent_bundle_qualify(ctx: click.Context, bundle_id: str, status: str, evidence: str | None) -> None:
    """Admin-only: record technical qualification evidence."""
    body = {
        "status": status,
        "evidence": (_parse_json_object(evidence, label="--evidence") if evidence is not None else {}),
    }
    data = request(ctx.obj["client"], "POST", f"/agent-bundles/{bundle_id}/qualification", json=body)
    emit(data, ctx.obj["json"], _agent_bundle_summary(data))


@agent_bundle.command("promote")
@click.argument("bundle_id")
@click.pass_context
def agent_bundle_promote(ctx: click.Context, bundle_id: str) -> None:
    """Admin-only: promote a qualified bundle into the shared public catalog."""
    data = request(ctx.obj["client"], "POST", f"/agent-bundles/{bundle_id}/promote", json={})
    emit(data, ctx.obj["json"], _agent_bundle_summary(data))


@agent_bundle.command("delete")
@click.argument("bundle_id")
@click.pass_context
def agent_bundle_delete(ctx: click.Context, bundle_id: str) -> None:
    """Delete your private bundle; admins may remove public bundles."""
    data = request(ctx.obj["client"], "DELETE", f"/agent-bundles/{bundle_id}")
    emit(data, ctx.obj["json"], [f"deleted agent-bundle {data.get('id', bundle_id)}"])


# ---------- evaluations ---------------------------------------------------


@cli.group()
def evaluation() -> None:
    """Run evaluations and pull down their results."""


def _runnability_summary(data: dict[str, Any]) -> list[str]:
    verdict = "runnable" if data.get("runnable") else "not runnable"
    lines = [f"Preflight: {verdict}"]
    for check in data.get("checks", []):
        marker = "BLOCKED" if check.get("blocking") else str(check.get("state", "unknown"))
        lines.append(f"  {check.get('prerequisite')}: {marker} — {check.get('message')}")
    summary = data.get("member_summary")
    if isinstance(summary, dict):
        lines.append(
            "  members: "
            f"{summary.get('ready', 0)} ready, {summary.get('blocked', 0)} blocked, "
            f"{summary.get('total', 0)} total"
        )
    return lines


def _preflight_or_abort(
    ctx: click.Context,
    *,
    path: str,
    body: dict[str, object],
) -> None:
    """Check the exact create body and stop before mutation when blocked."""
    data = request(ctx.obj["client"], "POST", path, json=body)
    if not ctx.obj["json"] or not data.get("runnable"):
        emit(data, ctx.obj["json"], _runnability_summary(data))
    if not data.get("runnable"):
        raise click.exceptions.Exit(1)


@evaluation.command("preflight")
@click.option(
    "--request",
    "request_body",
    required=True,
    metavar="JSON|@FILE",
    help="Complete evaluation-create request as JSON or @file.json.",
)
@click.pass_context
def evaluation_preflight(ctx: click.Context, request_body: str) -> None:
    """Check a complete evaluation request without creating it."""
    body = _parse_json_object(request_body, label="--request")
    data = request(ctx.obj["client"], "POST", "/evaluations/preflight", json=body)
    emit(data, ctx.obj["json"], _runnability_summary(data))


@evaluation.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option("--task-id", required=True)
@click.option("--task-revision", type=int, required=True)
@click.option(
    "--framework-version",
    default=None,
    help=("Exact supported framework version or alias (for Harbor, use a catalog version or stable)."),
)
@click.option(
    "--framework-profile-id",
    default=None,
    help=("Generic framework config profile id: harbor profile for harbor, gym profile for nemo_gym."),
)
@click.option(
    "--harbor-profile-id",
    default=None,
    help="Compatibility alias for --framework-profile-id on harbor requests.",
)
@click.option("--switchyard-profile-id", default=None, help="Optional Switchyard profile id.")
@click.option("--intake-profile-id", default=None, help="Optional Intake profile id.")
@click.option(
    "--credential",
    "credentials",
    multiple=True,
    metavar="KEY=cred_id",
    help="Credential mapping, repeatable (e.g. --credential anthropic=cred_123).",
)
@click.option(
    "--agent-bundle",
    default=None,
    metavar="BUNDLE_ID",
    help="Accessible private or public agent-bundle id from `agent-bundle list`.",
)
@click.option("--extra-skill-object-key", multiple=True, help="Skill object key, repeatable.")
@click.option("--instruction-prefix", default=None)
@click.option("--instruction-postfix", default=None)
@click.option("--initial-user-turn", multiple=True, help="Initial Harbor user turn, repeatable.")
@click.option("--runtime", default=None, help=f"Dispatch runtime backend. {KNOWN_RUNTIME_HINT}")
@click.option(
    "--network-policy",
    type=click.Choice(("unrestricted", "default_deny", "scoped_egress")),
    default=None,
    help="Direct sandbox egress policy.",
)
@click.option(
    "--network-policy-config",
    default=None,
    metavar="YAML|@FILE",
    help="Scoped-egress policy config as YAML/JSON or @file.",
)
@click.option("--parallelism", type=int, default=None)
@click.option("--n-attempts", type=int, default=None)
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.option(
    "--framework",
    type=click.Choice(FRAMEWORK),
    default=None,
    help="Runner framework. API default: harbor.",
)
@click.option(
    "--preflight",
    is_flag=True,
    help="Check the exact request and stop before creation when it is not runnable.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for terminal status after creating the evaluation.",
)
@click.option(
    "--wait-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds between polls.",
)
@click.option("--wait-timeout", type=float, default=None, help="Maximum seconds to wait.")
@click.pass_context
def evaluation_create(
    ctx: click.Context,
    name: str,
    task_id: str,
    task_revision: int,
    framework_version: str | None,
    framework_profile_id: str | None,
    harbor_profile_id: str | None,
    switchyard_profile_id: str | None,
    intake_profile_id: str | None,
    credentials: tuple[str, ...],
    agent_bundle: str | None,
    extra_skill_object_key: tuple[str, ...],
    instruction_prefix: str | None,
    instruction_postfix: str | None,
    initial_user_turn: tuple[str, ...],
    runtime: str | None,
    network_policy: str | None,
    network_policy_config: str | None,
    parallelism: int | None,
    n_attempts: int | None,
    visibility: str | None,
    framework: str | None,
    preflight: bool,
    wait: bool,
    wait_interval: float,
    wait_timeout: float | None,
) -> None:
    """Start an evaluation run over a single task (run a benchmark via `benchmark-run create`)."""
    body: dict[str, object] = {"name": name, "task_id": task_id, "task_revision": task_revision}
    cred_map: dict[str, str] = {}
    for item in credentials:
        if "=" not in item:
            raise click.ClickException(f"--credential must be KEY=cred_id, got {item!r}")
        k, v = item.split("=", 1)
        cred_map[k] = v
    if cred_map:
        body["credentials"] = cred_map
    if agent_bundle is not None:
        body["agent_bundle_id"] = agent_bundle
    if extra_skill_object_key:
        body["extra_skill_object_keys"] = list(extra_skill_object_key)
    if initial_user_turn:
        body["initial_user_turns"] = list(initial_user_turn)

    for field, value in (
        ("framework_profile_id", framework_profile_id),
        ("framework_version", framework_version),
        ("harbor_profile_id", harbor_profile_id),
        ("switchyard_profile_id", switchyard_profile_id),
        ("intake_profile_id", intake_profile_id),
        ("runtime", runtime),
        ("network_policy", network_policy),
        ("instruction_prefix", instruction_prefix),
        ("instruction_postfix", instruction_postfix),
        ("n_attempts", n_attempts),
        ("parallelism", parallelism),
        ("visibility", visibility),
        ("framework", framework),
    ):
        if value is not None:
            body[field] = value
    if network_policy_config is not None:
        body["network_policy_config"] = _parse_yaml_object(network_policy_config, label="network policy config")

    if preflight:
        _preflight_or_abort(ctx, path="/evaluations/preflight", body=body)

    data = request(ctx.obj["client"], "POST", "/evaluations", json=body)
    if wait:
        if not ctx.obj["json"]:
            click.echo(f"created evaluation {data['id']}")
        _wait_for_evaluation(
            ctx,
            str(data["id"]),
            interval=wait_interval,
            timeout=wait_timeout,
        )
        return
    emit(data, ctx.obj["json"], _evaluation_summary(data))


@evaluation.command("list")
@click.option("--status", "status_", type=click.Choice(EVAL_STATUS), default=None)
@click.option("--task-id", default=None)
@click.option("--team-id", default=None, help="Only evaluations visible to this team.")
@click.option("--mine", is_flag=True, help="Only my evaluations.")
@click.option("--shared", is_flag=True, help="Only non-private evaluations.")
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--limit", type=int, default=None, help="Page size (1-100, default 20).")
@click.option("--order", type=click.Choice(ORDER), default=None, help="Sort order by creation time.")
@click.pass_context
def evaluation_list(
    ctx: click.Context,
    status_: str | None,
    task_id: str | None,
    team_id: str | None,
    mine: bool,
    shared: bool,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List evaluations, newest first (paginated)."""
    params: dict[str, object] = {}
    if status_:
        params["status"] = status_
    if task_id:
        params["task_id"] = task_id
    if team_id:
        params["team_id"] = team_id
    if mine:
        params["mine"] = "true"
    if shared:
        params["shared"] = "true"
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if order:
        params["order"] = order
    data = request(ctx.obj["client"], "GET", "/evaluations", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda e: f"{e['id']}  {str(e.get('status')):12}  {e.get('name')}",
    )


@evaluation.command("get")
@click.argument("evaluation_id")
@click.pass_context
def evaluation_get(ctx: click.Context, evaluation_id: str) -> None:
    """Show one evaluation, including its result summary once terminal."""
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}")
    emit(data, ctx.obj["json"], _evaluation_summary(data))


@evaluation.command("retry")
@click.argument("evaluation_id")
@click.pass_context
def evaluation_retry(ctx: click.Context, evaluation_id: str) -> None:
    """Retry a failed task evaluation.

    Benchmark members retain their existing benchmark aggregate membership.
    """
    data = request(ctx.obj["client"], "POST", f"/evaluations/{evaluation_id}/retry")
    emit(data, ctx.obj["json"], _evaluation_summary(data))


@evaluation.command("cancel")
@click.argument("evaluation_id")
@click.pass_context
def evaluation_cancel(ctx: click.Context, evaluation_id: str) -> None:
    """Cancel an in-flight run (idempotent on terminal runs)."""
    data = request(ctx.obj["client"], "POST", f"/evaluations/{evaluation_id}/cancel")
    emit(data, ctx.obj["json"], [f"evaluation {data['id']}", f"  status: {data.get('status')}"])


@evaluation.command("delete")
@click.argument("evaluation_id")
@click.pass_context
def evaluation_delete(ctx: click.Context, evaluation_id: str) -> None:
    """Soft-delete an evaluation's metadata (artifacts are retained)."""
    data = request(ctx.obj["client"], "DELETE", f"/evaluations/{evaluation_id}")
    emit(data, ctx.obj["json"], [f"deleted evaluation {data.get('id', evaluation_id)}"])


@evaluation.command("logs")
@click.argument("evaluation_id")
@click.option("--tail", type=int, default=None, help="Trailing lines to fetch (default 100).")
@click.option("--follow", is_flag=True, help="Follow live log SSE until terminal status.")
@click.pass_context
def evaluation_logs(ctx: click.Context, evaluation_id: str, tail: int | None, follow: bool) -> None:
    """Fetch a snapshot of runner logs."""
    if follow:
        if tail is not None:
            raise click.UsageError("--tail cannot be combined with --follow")
        _stream_logs(ctx, evaluation_id)
        return
    params: dict[str, object] = {}
    if tail is not None:
        params["tail_lines"] = tail
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}/logs", params=params)
    if ctx.obj["json"]:
        emit(data, True, [])
        return
    for line in data.get("lines", []):
        click.echo(line)
    done = " (complete)" if data.get("complete") else ""
    click.echo(f"-- status: {data.get('status')}{done}", err=True)


@evaluation.command("events")
@click.argument("evaluation_id")
@click.option("--limit", type=int, default=None, help="Page size (1-200, default 100).")
@click.option("--cursor", default=None, help="Pagination cursor from a prior page.")
@click.option("--offset", type=int, default=None, help="Offset pagination compatibility option.")
@click.option("--follow", is_flag=True, help="Follow live event SSE until terminal status.")
@click.pass_context
def evaluation_events(
    ctx: click.Context,
    evaluation_id: str,
    limit: int | None,
    cursor: str | None,
    offset: int | None,
    follow: bool,
) -> None:
    """List state-transition events for an evaluation."""
    if follow:
        if limit is not None or cursor is not None or offset is not None:
            raise click.UsageError("--limit, --cursor, and --offset cannot be combined with --follow")
        _stream_events(ctx, evaluation_id)
        return
    params: dict[str, object] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    if offset is not None:
        params["offset"] = offset
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}/events", params=params)
    emit_list(data, ctx.obj["json"], _event_line)


@evaluation.command("follow")
@click.argument("evaluation_id")
@click.pass_context
def evaluation_follow(ctx: click.Context, evaluation_id: str) -> None:
    """Follow live log lines and status updates until terminal status."""
    _stream_logs(ctx, evaluation_id)


@evaluation.command("wait")
@click.argument("evaluation_id")
@click.option(
    "--interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds between polls.",
)
@click.option("--timeout", type=float, default=None, help="Maximum seconds to wait.")
@click.pass_context
def evaluation_wait(
    ctx: click.Context,
    evaluation_id: str,
    interval: float,
    timeout: float | None,
) -> None:
    """Poll an existing evaluation until succeeded, failed, cancelled, or blocked."""
    _wait_for_evaluation(ctx, evaluation_id, interval=interval, timeout=timeout)


@evaluation.command("provenance")
@click.argument("evaluation_id")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the provenance JSON to this file.",
)
@click.pass_context
def evaluation_provenance(ctx: click.Context, evaluation_id: str, output: Path | None) -> None:
    """Print or download the run provenance manifest."""
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}")
    links = data.get("links")
    if not isinstance(links, dict):
        links = {}
    api_path = links.get("provenance") or (f"/evaluations/{evaluation_id}/artifacts/scaled-evals-provenance.json")
    try:
        content = fetch_artifact(ctx.obj["client"], api_path)
    except ApiError as exc:
        if exc.code == "not_found":
            raise click.ClickException(f"provenance for {evaluation_id} is not available yet") from exc
        raise
    if output:
        output.write_bytes(content)
        if ctx.obj["json"]:
            emit({"evaluation_id": evaluation_id, "path": str(output)}, True, [])
        else:
            click.echo(f"downloaded provenance -> {output}")
        return
    if ctx.obj["json"]:
        try:
            click.echo(json.dumps(json.loads(content), indent=2))
        except json.JSONDecodeError as exc:
            raise click.ClickException("provenance artifact is not valid JSON") from exc
    else:
        click.echo(content.decode("utf-8"))


@evaluation.command("sbom")
@click.argument("evaluation_id")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the CycloneDX JSON to this file.",
)
@click.pass_context
def evaluation_sbom(ctx: click.Context, evaluation_id: str, output: Path | None) -> None:
    """Print or download the run-composition CycloneDX BOM."""
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}")
    links = data.get("links")
    if not isinstance(links, dict):
        links = {}
    api_path = links.get("sbom") or (f"/evaluations/{evaluation_id}/artifacts/scaled-evals-sbom.cdx.json")
    try:
        content = fetch_artifact(ctx.obj["client"], api_path)
    except ApiError as exc:
        if exc.code == "not_found":
            raise click.ClickException(f"SBOM for {evaluation_id} is not available yet") from exc
        raise
    if output:
        output.write_bytes(content)
        if ctx.obj["json"]:
            emit({"evaluation_id": evaluation_id, "path": str(output)}, True, [])
        else:
            click.echo(f"downloaded SBOM -> {output}")
        return
    if ctx.obj["json"]:
        try:
            click.echo(json.dumps(json.loads(content), indent=2))
        except json.JSONDecodeError as exc:
            raise click.ClickException("SBOM artifact is not valid JSON") from exc
    else:
        click.echo(content.decode("utf-8"))


@evaluation.command("reproduce")
@click.argument("evaluation_id")
@click.option(
    "--rerun",
    is_flag=True,
    help="Submit the returned request immediately instead of only printing it.",
)
@click.option("--name", default=None, help="Override the rerun evaluation name.")
@click.option("--wait", is_flag=True, help="Wait for terminal status after --rerun.")
@click.option(
    "--wait-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds between polls when --wait is set.",
)
@click.option("--wait-timeout", type=float, default=None, help="Maximum seconds to wait.")
@click.pass_context
def evaluation_reproduce(
    ctx: click.Context,
    evaluation_id: str,
    rerun: bool,
    name: str | None,
    wait: bool,
    wait_interval: float,
    wait_timeout: float | None,
) -> None:
    """Show or submit the safe rerun request for a prior evaluation."""
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}/reproduce")
    body = data.get("request")
    if not isinstance(body, dict):
        raise click.ClickException("reproduce response did not include a request body")
    if name:
        body["name"] = name
        command = _reproduce_command(body)
    else:
        command = data.get("cli_command") if isinstance(data.get("cli_command"), list) else None
        if not command:
            command = _reproduce_command(body)
    if rerun:
        created = request(ctx.obj["client"], "POST", "/evaluations", json=body)
        if wait:
            if not ctx.obj["json"]:
                click.echo(f"created evaluation {created['id']}")
            _wait_for_evaluation(ctx, str(created["id"]), interval=wait_interval, timeout=wait_timeout)
            return
        emit(created, ctx.obj["json"], _evaluation_summary(created))
        return
    if ctx.obj["json"]:
        data["request"] = body
        data["cli_command"] = command
        emit(data, True, [])
        return
    click.echo("rerun command:")
    click.echo("  " + shlex.join(command))
    click.echo("request:")
    click.echo(json.dumps(body, indent=2))
    notes = data.get("notes")
    if isinstance(notes, list) and notes:
        click.echo("notes:")
        for note in notes:
            click.echo(f"  - {note}")


def _reproduce_command(body: dict[str, object]) -> list[str]:
    required = ("name", "task_id", "task_revision", "runtime", "parallelism")
    missing = [key for key in required if body.get(key) is None]
    if missing:
        raise click.ClickException("reproduce response is missing required field(s): " + ", ".join(missing))
    command = [
        "scaled-evals",
        "evaluation",
        "create",
        "--name",
        str(body["name"]),
        "--task-id",
        str(body["task_id"]),
        "--task-revision",
        str(body["task_revision"]),
        "--framework",
        str(body.get("framework") or "harbor"),
        "--runtime",
        str(body["runtime"]),
        "--network-policy",
        str(body.get("network_policy") or "unrestricted"),
        "--parallelism",
        str(body["parallelism"]),
        "--visibility",
        str(body.get("visibility") or "private"),
    ]
    network_policy_config = body.get("network_policy_config")
    if isinstance(network_policy_config, dict) and network_policy_config:
        command.extend(["--network-policy-config", json.dumps(network_policy_config, separators=(",", ":"))])
    for option, key in (
        ("--framework-profile-id", "framework_profile_id"),
        ("--switchyard-profile-id", "switchyard_profile_id"),
        ("--intake-profile-id", "intake_profile_id"),
    ):
        if value := body.get(key):
            command.extend([option, str(value)])
    credentials = body.get("credentials")
    if isinstance(credentials, dict):
        for role, credential_id in sorted(credentials.items()):
            command.extend(["--credential", f"{role}={credential_id}"])
    return command


def _stream_logs(ctx: click.Context, evaluation_id: str) -> None:
    for event_name, raw in iter_sse(ctx.obj["client"], f"/evaluations/{evaluation_id}/logs/stream"):
        data = _parse_sse_data(event_name, raw)
        if event_name == "ping":
            continue
        if ctx.obj["json"]:
            click.echo(json.dumps({"event": event_name, "data": data}))
        elif event_name == "log":
            click.echo(data.get("line", ""))
        elif event_name == "status":
            status = data.get("status")
            detail = f" ({data['detail']})" if data.get("detail") else ""
            click.echo(f"-- status: {status}{detail}", err=True)
        if event_name == "status" and data.get("status") in TERMINAL_STATUSES:
            return
    raise click.ClickException("log stream ended before a terminal status event")


def _stream_events(ctx: click.Context, evaluation_id: str) -> None:
    for event_name, raw in iter_sse(ctx.obj["client"], f"/evaluations/{evaluation_id}/events/stream"):
        data = _parse_sse_data(event_name, raw)
        if event_name == "ping":
            continue
        if ctx.obj["json"]:
            click.echo(json.dumps({"event": event_name, "data": data}))
        else:
            click.echo(_event_line(data))
        if data.get("type", event_name) == "status" and data.get("status") in TERMINAL_STATUSES:
            return
    raise click.ClickException("event stream ended before a terminal status event")


@evaluation.command("artifacts")
@click.argument("evaluation_id")
@click.option("--prefix", default=None, help="Only artifacts under this path prefix.")
@click.pass_context
def evaluation_artifacts(ctx: click.Context, evaluation_id: str, prefix: str | None) -> None:
    """List an evaluation's result artifacts."""
    params: dict[str, object] = {}
    if prefix:
        params["prefix"] = prefix
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}/artifacts", params=params)
    emit_list(data, ctx.obj["json"], lambda a: f"{str(a.get('size_bytes')):>10}  {a['path']}")


@evaluation.command("download")
@click.argument("evaluation_id")
@click.argument("artifact_path")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Local destination (default: the artifact's basename).",
)
@click.pass_context
def evaluation_download(ctx: click.Context, evaluation_id: str, artifact_path: str, output: Path | None) -> None:
    """Download one artifact file to disk."""
    dest = output or Path(Path(artifact_path).name)
    download_artifact(ctx.obj["client"], f"/evaluations/{evaluation_id}/artifacts/{artifact_path}", dest)
    if ctx.obj["json"]:
        emit(
            {
                "evaluation_id": evaluation_id,
                "artifact_path": artifact_path,
                "path": str(dest),
                "downloaded": True,
            },
            True,
            [],
        )
    else:
        click.echo(f"downloaded {artifact_path} -> {dest}")


@evaluation.command("archive")
@click.argument("evaluation_id")
@click.option("--build", is_flag=True, help="Queue or requeue archive generation.")
@click.option("--force", is_flag=True, help="Rebuild an existing ready archive.")
@click.option("--download", "download_", is_flag=True, help="Download the ready archive.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Local archive destination when --download is set.",
)
@click.pass_context
def evaluation_archive(
    ctx: click.Context,
    evaluation_id: str,
    build: bool,
    force: bool,
    download_: bool,
    output: Path | None,
) -> None:
    """Show, build, or download the full-run results archive."""
    method = "POST" if build or force else "GET"
    kwargs = {"json": {"force": force}} if method == "POST" else {}
    data = request(ctx.obj["client"], method, f"/evaluations/{evaluation_id}/archive", **kwargs)
    dl = data.get("download") or {}
    if download_:
        if data.get("status") != "ready" or not dl:
            raise click.ClickException(f"archive for {evaluation_id} is not ready (status: {data.get('status')})")
        dest = output or Path(f"{evaluation_id}-results.tar.gz")
        download_presigned(ctx.obj["client"], dl, dest)
        if ctx.obj["json"]:
            emit(data, True, [])
        else:
            click.echo(f"downloaded archive -> {dest}")
        return
    summary = [
        f"archive for {data.get('evaluation_id', evaluation_id)}",
        f"  status: {data.get('status')}",
        f"  format: {data.get('format')}",
    ]
    if dl.get("url"):
        summary.append(f"  url:    {dl['url']}")
    emit(data, ctx.obj["json"], summary)


@evaluation.command("harbor-viewer")
@click.argument("evaluation_id")
@click.option("--download", is_flag=True, help="Keep the compatible archive locally.")
@click.option(
    "--upload",
    "upload_",
    is_flag=True,
    help="Upload the compatible archive to the configured Harbor Viewer.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Local archive destination (implies --download).",
)
@click.pass_context
def evaluation_harbor_viewer(
    ctx: click.Context,
    evaluation_id: str,
    download: bool,
    upload_: bool,
    output: Path | None,
) -> None:
    """Show, download, or manually upload a Harbor Viewer-compatible archive."""
    data = request(ctx.obj["client"], "GET", f"/evaluations/{evaluation_id}")
    links = data.get("links")
    if not isinstance(links, dict):
        links = {}
    archive_url = links.get("harbor_viewer_archive")
    upload_url = links.get("harbor_viewer_upload")
    viewer_url = links.get("harbor_viewer")
    if not archive_url:
        raise click.ClickException(f"Harbor Viewer archive for {evaluation_id} is not available")
    if upload_ and not upload_url:
        raise click.ClickException("evaluation has no Harbor Viewer upload endpoint")

    result: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "archive_url": archive_url,
        "upload_url": upload_url,
        "viewer_url": viewer_url,
    }
    keep_archive = download or output is not None
    if not keep_archive and not upload_:
        emit(
            result,
            ctx.obj["json"],
            [
                f"Harbor Viewer export for {evaluation_id}",
                f"  archive: {archive_url}",
                f"  upload:  {upload_url}",
                f"  command: scaled-evals evaluation harbor-viewer {evaluation_id} --upload",
            ],
        )
        return

    def transfer(archive_path: Path) -> None:
        download_artifact(ctx.obj["client"], str(archive_url), archive_path)
        if keep_archive:
            result["path"] = str(archive_path)
        if upload_:
            result["upload"] = upload_multipart_archive(
                ctx.obj["client"],
                str(upload_url),
                archive_path,
            )

    if keep_archive:
        archive_path = output or Path(f"{evaluation_id}-harbor-viewer.tar.gz")
        transfer(archive_path)
    else:
        with tempfile.TemporaryDirectory(prefix="scaled-evals-harbor-viewer-") as tmp:
            transfer(Path(tmp) / f"{evaluation_id}-harbor-viewer.tar.gz")

    emit(
        result,
        ctx.obj["json"],
        [
            f"Harbor Viewer export for {evaluation_id}",
            *([f"  downloaded: {result['path']}"] if result.get("path") else []),
            *(
                [f"  uploaded:   {(result.get('upload') or {}).get('job_name', evaluation_id)}"]
                if result.get("upload")
                else []
            ),
            *([f"  viewer:     {viewer_url}"] if viewer_url else []),
        ],
    )


# ---------- benchmark runs --------------------------------------------------


def _benchmark_member_failure_codes(member: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    direct = member.get("failure_code") or member.get("last_failure_code")
    if direct:
        codes.append(str(direct))
    exception_counts = member.get("exception_counts")
    if isinstance(exception_counts, dict):
        for code, count in exception_counts.items():
            if int(count or 0) > 0 and str(code) not in codes:
                codes.append(str(code))
    return codes


def _benchmark_member_failure_category(member: dict[str, object]) -> str | None:
    raw = member.get("failure_category") or member.get("last_failure_category")
    if raw in FAILURE_CATEGORY:
        return str(raw)
    if raw == "retryable_task":
        return "infrastructure"
    codes = _benchmark_member_failure_codes(member)
    if raw == "non_retryable" or (member.get("status") == "failed" and codes):
        return failure_category_for_code(
            codes[0] if codes else None,
            str(member.get("status_detail") or ""),
            default="unknown",
        )
    return None


def _filter_benchmark_members(
    members: list[dict[str, object]],
    *,
    status: str | None,
    failure_code: str | None,
    failure_category: str | None,
) -> list[dict[str, object]]:
    filtered = []
    for member in members:
        if status and member.get("status") != status:
            continue
        if failure_code and (
            member.get("status") != "failed" or failure_code not in _benchmark_member_failure_codes(member)
        ):
            continue
        if failure_category and (
            member.get("status") != "failed" or _benchmark_member_failure_category(member) != failure_category
        ):
            continue
        filtered.append(member)
    return filtered


def _benchmark_member_evaluation_summary(member: dict[str, Any]) -> str:
    codes = _benchmark_member_failure_codes(member)
    category = _benchmark_member_failure_category(member)
    failure = f"  failure={category or 'unknown'}/{','.join(codes)}" if codes else ""
    current = int(member.get("current_execution") or 1)
    maximum = int(member.get("max_executions") or current)
    retryable = (
        member.get("status") == "failed"
        and current < maximum
        and any(is_retryable_failure(code, str(member.get("status_detail") or "")) for code in codes)
    )
    recovered = member.get("status") == "succeeded" and current > 1 and bool(codes)
    marker = "  auto-retryable" if retryable else "  recovered" if recovered else ""
    return (
        f"{member['id']}  {str(member.get('status')):12}  reward={member.get('reward')}  "
        f"{member.get('task_id')}  attempt={current}/{maximum}{failure}{marker}"
    )


def _benchmark_run_summary(data: dict[str, Any], *, member_total: int | None = None) -> list[str]:
    detail = f" ({data['status_detail']})" if data.get("status_detail") else ""
    lines = [
        f"benchmark run {data['id']}",
        f"  name:      {data.get('name')}",
        f"  status:    {data.get('status')}{detail}",
        f"  benchmark: {data.get('benchmark_id')} rev {data.get('benchmark_revision')}",
    ]
    if data.get("framework_version"):
        lines.append(
            f"  runner:    {data.get('framework')} {data['framework_version']} "
            f"(sandbox-k8s {data.get('sandbox_k8s_version') or 'n/a'})"
        )
    if data.get("reward") is not None:
        lines.append(f"  reward:    {data['reward']} (mean across scored member tasks)")
    if data.get("n_trials") is not None:
        lines.append(f"  trials:    {data['n_trials']} ({data.get('n_errored')} errored)")
    if data.get("failure_counts"):
        lines.append(f"  failures:  {data['failure_counts']}")
    if data.get("n_retryable_failures"):
        lines.append(f"  auto-retryable: {data['n_retryable_failures']} member failure(s)")
    if data.get("n_recovered"):
        lines.append(f"  recovered: {data['n_recovered']} member(s) after retry")
    result = data.get("result")
    if isinstance(result, dict):
        raw_per_task = result.get("per_task")
        per_task = raw_per_task if isinstance(raw_per_task, list) else []
        if member_total is not None:
            lines.append(f"  members:   {len(per_task)} of {member_total} matched filters")
        for t in per_task:
            slug = t.get("task_slug") or t.get("task_id")
            ev = t.get("evaluation_id")
            attempts = f" attempt={t.get('attempt')}/{t.get('max_attempts')}"
            failure = ""
            if t.get("failure_category") or t.get("failure_code"):
                failure = f" failure={t.get('failure_category') or 'unknown'}/{t.get('failure_code') or 'unknown'}"
            retry = " retryable" if t.get("retryable") else ""
            recovered = " recovered" if t.get("recovered") else ""
            lines.append(
                f"    - {slug}: {t.get('status')} reward={t.get('reward')}{attempts}{failure}{retry}{recovered} [{ev}]"
            )
    return lines


def _wait_for_benchmark_run(ctx: click.Context, run_id: str, *, interval: float, timeout: float | None) -> None:
    deadline = time.monotonic() + timeout if timeout is not None else None
    last_status = None
    try:
        while True:
            data = request(ctx.obj["client"], "GET", f"/benchmark-runs/{run_id}")
            status = str(data.get("status") or "")
            if not ctx.obj["json"] and status != last_status:
                click.echo(f"{run_id}: {status}")
            last_status = status
            if status in TERMINAL_STATUSES:
                emit(data, ctx.obj["json"], _benchmark_run_summary(data))
                if status in SUCCESS_STATUSES:
                    return
                raise click.exceptions.Exit(1)
            if deadline is not None and time.monotonic() >= deadline:
                raise click.ClickException(f"timed out waiting for {run_id}")
            time.sleep(interval)
    except KeyboardInterrupt as exc:
        click.echo("interrupted", err=True)
        raise click.exceptions.Exit(130) from exc


@cli.group(name="benchmark-run")
def benchmark_run() -> None:
    """Run a benchmark by aggregating member task evaluations."""


@benchmark_run.command("archive")
@click.argument("run_id")
@click.option("--build", is_flag=True, help="Queue archive generation.")
@click.option("--force", is_flag=True, help="Build a fresh snapshot of member executions.")
@click.pass_context
def benchmark_run_archive(ctx: click.Context, run_id: str, build: bool, force: bool) -> None:
    """Show or build a combined Harbor experiment archive."""
    method = "POST" if build or force else "GET"
    kwargs = {"json": {"force": force}} if method == "POST" else {}
    data = request(ctx.obj["client"], method, f"/benchmark-runs/{run_id}/archive", **kwargs)
    emit(data, ctx.obj["json"], [f"{run_id}: {data['status']}", data.get("error") or ""])


@benchmark_run.command("download")
@click.argument("run_id")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination directory (or tar.gz file with --archive-only). Must not exist.",
)
@click.option("--archive-only", is_flag=True, help="Save the tarball without extracting it.")
@click.option("--wait", is_flag=True, help="Wait for archive generation, then download.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0.1),
    default=600.0,
    show_default=True,
    help="Maximum seconds to wait for archive generation.",
)
@click.pass_context
def benchmark_run_download(
    ctx: click.Context,
    run_id: str,
    output: Path | None,
    archive_only: bool,
    wait: bool,
    timeout: float,
) -> None:
    """Download all member trials into one Harbor experiment directory.

    Queues a missing archive. Use --wait to download when the build completes,
    or rerun this command once the archive is ready.
    """
    from scaled_evals.cli.benchmark_archive import save_benchmark_archive

    dest = output or Path(f"{run_id}-results.tar.gz" if archive_only else run_id)
    if dest.exists():
        raise click.ClickException(f"destination already exists: {dest}")
    path = f"/benchmark-runs/{run_id}/archive"
    data = request(ctx.obj["client"], "GET", path)
    if data["status"] == "missing":
        data = request(ctx.obj["client"], "POST", path, json={"force": False})
    deadline = time.monotonic() + timeout
    while data["status"] in {"queued", "building"}:
        if not wait:
            emit(
                data,
                ctx.obj["json"],
                [f"{run_id}: archive {data['status']}; rerun download when ready, or use --wait."],
            )
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise click.ClickException("timed out waiting for archive; the server build continues")
        time.sleep(min(5.0, remaining))
        data = request(ctx.obj["client"], "GET", path)
    if data["status"] != "ready":
        raise click.ClickException(
            f"archive {data['status']}: {data.get('error') or 'not ready'}; "
            f"retry with benchmark-run archive {run_id} --build"
        )
    save_benchmark_archive(ctx.obj["client"], data["download"], dest, archive_only=archive_only)
    summary = [f"downloaded Harbor experiment -> {dest}"]
    if data.get("partial"):
        summary.append("Some member trial data is missing; see scaled-evals-benchmark-archive.json.")
    emit({**data, "path": str(dest)}, ctx.obj["json"], summary)


@benchmark_run.command("preflight")
@click.option(
    "--request",
    "request_body",
    required=True,
    metavar="JSON|@FILE",
    help="Complete benchmark-run create request as JSON or @file.json.",
)
@click.pass_context
def benchmark_run_preflight(ctx: click.Context, request_body: str) -> None:
    """Check a complete benchmark-run request without creating it."""
    body = _parse_json_object(request_body, label="--request")
    data = request(ctx.obj["client"], "POST", "/benchmark-runs/preflight", json=body)
    emit(data, ctx.obj["json"], _runnability_summary(data))


@benchmark_run.command("create")
@click.option("--name", required=True, help="Display name.")
@click.option("--benchmark-id", required=True, help="Benchmark to run.")
@click.option(
    "--benchmark-revision",
    type=int,
    default=None,
    help="Benchmark revision (default: the benchmark's current revision).",
)
@click.option(
    "--framework-version",
    default=None,
    help=("Exact supported framework version or alias (for Harbor, use a catalog version or stable)."),
)
@click.option("--framework-profile-id", default=None)
@click.option(
    "--member-framework-profile",
    "member_framework_profiles",
    multiple=True,
    metavar="TASK_ID=PROFILE_ID",
    help="Per-member framework profile override, repeatable.",
)
@click.option("--harbor-profile-id", default=None)
@click.option("--switchyard-profile-id", default=None)
@click.option("--intake-profile-id", default=None)
@click.option(
    "--credential",
    "credentials",
    multiple=True,
    metavar="KEY=cred_id",
    help="Credential mapping, repeatable.",
)
@click.option(
    "--agent-bundle",
    default=None,
    metavar="BUNDLE_ID",
    help="Accessible private or public agent-bundle id for every benchmark member.",
)
@click.option("--extra-skill-object-key", multiple=True, help="Skill object key, repeatable.")
@click.option("--instruction-prefix", default=None)
@click.option("--instruction-postfix", default=None)
@click.option("--initial-user-turn", multiple=True, help="Initial Harbor user turn, repeatable.")
@click.option("--runtime", default=None, help=f"Dispatch runtime backend. {KNOWN_RUNTIME_HINT}")
@click.option(
    "--network-policy",
    type=click.Choice(("unrestricted", "default_deny", "scoped_egress")),
    default=None,
    help="Direct sandbox egress policy.",
)
@click.option(
    "--network-policy-config",
    default=None,
    metavar="YAML|@FILE",
    help="Scoped-egress policy config as YAML/JSON or @file.",
)
@click.option(
    "--parallelism",
    type=int,
    default=None,
    help="Trials within each member task (cross-task concurrency comes from the worker pool).",
)
@click.option("--n-attempts", type=int, default=None, help="Attempts for each member task.")
@click.option(
    "--max-concurrent-members",
    type=int,
    default=None,
    help=(
        "Maximum active benchmark member evaluations. Also caps members sharing "
        "one managed Switchyard gateway when a Switchyard profile is set."
    ),
)
@click.option("--visibility", type=click.Choice(VISIBILITY), default=None)
@click.option("--framework", type=click.Choice(FRAMEWORK), default=None)
@click.option(
    "--preflight",
    is_flag=True,
    help="Check the exact request and stop before creation when it is not runnable.",
)
@click.option("--wait", is_flag=True, help="Wait for the run to reach a terminal status.")
@click.option("--wait-interval", type=float, default=5.0, show_default=True)
@click.option("--wait-timeout", type=float, default=None, help="Maximum seconds to wait.")
@click.pass_context
def benchmark_run_create(
    ctx: click.Context,
    name: str,
    benchmark_id: str,
    benchmark_revision: int | None,
    framework_version: str | None,
    framework_profile_id: str | None,
    member_framework_profiles: tuple[str, ...],
    harbor_profile_id: str | None,
    switchyard_profile_id: str | None,
    intake_profile_id: str | None,
    credentials: tuple[str, ...],
    agent_bundle: str | None,
    extra_skill_object_key: tuple[str, ...],
    instruction_prefix: str | None,
    instruction_postfix: str | None,
    initial_user_turn: tuple[str, ...],
    runtime: str | None,
    network_policy: str | None,
    network_policy_config: str | None,
    parallelism: int | None,
    n_attempts: int | None,
    max_concurrent_members: int | None,
    visibility: str | None,
    framework: str | None,
    preflight: bool,
    wait: bool,
    wait_interval: float,
    wait_timeout: float | None,
) -> None:
    """Start a benchmark run: spawns one member evaluation per member task."""
    body: dict[str, object] = {"name": name, "benchmark_id": benchmark_id}
    if benchmark_revision is not None:
        body["benchmark_revision"] = benchmark_revision
    cred_map: dict[str, str] = {}
    for item in credentials:
        if "=" not in item:
            raise click.ClickException(f"--credential must be KEY=cred_id, got {item!r}")
        k, v = item.split("=", 1)
        cred_map[k] = v
    if cred_map:
        body["credentials"] = cred_map
    member_profile_map: dict[str, str] = {}
    for item in member_framework_profiles:
        if "=" not in item:
            raise click.ClickException(f"--member-framework-profile must be TASK_ID=PROFILE_ID, got {item!r}")
        task_id, profile_id = item.split("=", 1)
        member_profile_map[task_id] = profile_id
    if member_profile_map:
        body["member_framework_profile_ids"] = member_profile_map
    if agent_bundle is not None:
        body["agent_bundle_id"] = agent_bundle
    if extra_skill_object_key:
        body["extra_skill_object_keys"] = list(extra_skill_object_key)
    if initial_user_turn:
        body["initial_user_turns"] = list(initial_user_turn)
    for field, value in (
        ("framework_profile_id", framework_profile_id),
        ("framework_version", framework_version),
        ("harbor_profile_id", harbor_profile_id),
        ("switchyard_profile_id", switchyard_profile_id),
        ("intake_profile_id", intake_profile_id),
        ("runtime", runtime),
        ("network_policy", network_policy),
        ("instruction_prefix", instruction_prefix),
        ("instruction_postfix", instruction_postfix),
        ("n_attempts", n_attempts),
        ("parallelism", parallelism),
        ("max_concurrent_members", max_concurrent_members),
        ("visibility", visibility),
        ("framework", framework),
    ):
        if value is not None:
            body[field] = value
    if network_policy_config is not None:
        body["network_policy_config"] = _parse_yaml_object(network_policy_config, label="network policy config")

    if preflight:
        _preflight_or_abort(ctx, path="/benchmark-runs/preflight", body=body)

    data = request(ctx.obj["client"], "POST", "/benchmark-runs", json=body)
    if wait:
        if not ctx.obj["json"]:
            click.echo(f"created benchmark run {data['id']}")
        _wait_for_benchmark_run(ctx, str(data["id"]), interval=wait_interval, timeout=wait_timeout)
        return
    emit(data, ctx.obj["json"], _benchmark_run_summary(data))


@benchmark_run.command("list")
@click.option("--benchmark-id", default=None, help="Only runs of this benchmark.")
@click.option("--cursor", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--order", type=click.Choice(ORDER), default=None)
@click.pass_context
def benchmark_run_list(
    ctx: click.Context,
    benchmark_id: str | None,
    cursor: str | None,
    limit: int | None,
    order: str | None,
) -> None:
    """List benchmark runs, newest first."""
    params: dict[str, object] = {}
    for key, value in (
        ("benchmark_id", benchmark_id),
        ("cursor", cursor),
        ("limit", limit),
        ("order", order),
    ):
        if value is not None:
            params[key] = value
    data = request(ctx.obj["client"], "GET", "/benchmark-runs", params=params)
    emit_list(
        data,
        ctx.obj["json"],
        lambda r: f"{r['id']}  {str(r.get('status')):12}  reward={r.get('reward')}  {r.get('name')}",  # noqa: E501
    )


@benchmark_run.command("get")
@click.argument("run_id")
@click.option("--status", type=click.Choice(EVAL_STATUS), default=None)
@click.option("--failure-code", default=None, help="Only failed members with this exact code.")
@click.option("--failure-category", type=click.Choice(FAILURE_CATEGORY), default=None)
@click.pass_context
def benchmark_run_get(
    ctx: click.Context,
    run_id: str,
    status: str | None,
    failure_code: str | None,
    failure_category: str | None,
) -> None:
    """Show a run and optionally filter its per-task member breakdown."""
    data = request(ctx.obj["client"], "GET", f"/benchmark-runs/{run_id}")
    result = data.get("result")
    member_total = None
    if isinstance(result, dict) and isinstance(result.get("per_task"), list):
        members = result["per_task"]
        member_total = len(members)
        filtered = _filter_benchmark_members(
            members,
            status=status,
            failure_code=failure_code,
            failure_category=failure_category,
        )
        data = {**data, "result": {**result, "per_task": filtered}}
    filters_applied = any((status, failure_code, failure_category))
    emit(
        data,
        ctx.obj["json"],
        _benchmark_run_summary(data, member_total=member_total if filters_applied else None),
    )


@benchmark_run.command("evaluations")
@click.argument("run_id")
@click.option("--status", type=click.Choice(EVAL_STATUS), default=None)
@click.option("--failure-code", default=None, help="Only failed members with this exact code.")
@click.option("--failure-category", type=click.Choice(FAILURE_CATEGORY), default=None)
@click.pass_context
def benchmark_run_evaluations(
    ctx: click.Context,
    run_id: str,
    status: str | None,
    failure_code: str | None,
    failure_category: str | None,
) -> None:
    """List and optionally filter all member task evaluations."""
    members: list[dict[str, object]] = []
    cursor = None
    seen_cursors: set[str] = set()
    while True:
        params: dict[str, object] = {"limit": 200, "order": "asc"}
        if cursor:
            params["cursor"] = cursor
        page = request(
            ctx.obj["client"],
            "GET",
            f"/benchmark-runs/{run_id}/evaluations",
            params=params,
        )
        members.extend(page.get("data", []))
        cursor = page.get("next_cursor")
        if not cursor:
            break
        if str(cursor) in seen_cursors:
            raise click.ClickException("benchmark member pagination returned a repeated cursor")
        seen_cursors.add(str(cursor))
    filtered = _filter_benchmark_members(
        members,
        status=status,
        failure_code=failure_code,
        failure_category=failure_category,
    )
    data = {"data": filtered, "next_cursor": None}
    emit_list(
        data,
        ctx.obj["json"],
        _benchmark_member_evaluation_summary,
    )


@benchmark_run.command("reproduce")
@click.argument("run_id")
@click.option(
    "--rerun",
    is_flag=True,
    help="Submit the returned request immediately instead of only printing it.",
)
@click.option("--name", default=None, help="Override the rerun benchmark-run name.")
@click.option("--wait", is_flag=True, help="Wait for terminal status after --rerun.")
@click.option("--wait-interval", type=float, default=5.0, show_default=True)
@click.option("--wait-timeout", type=float, default=None, help="Maximum seconds to wait.")
@click.pass_context
def benchmark_run_reproduce(
    ctx: click.Context,
    run_id: str,
    rerun: bool,
    name: str | None,
    wait: bool,
    wait_interval: float,
    wait_timeout: float | None,
) -> None:
    """Show or submit the safe rerun request for a prior benchmark run."""
    data = request(ctx.obj["client"], "GET", f"/benchmark-runs/{run_id}/reproduce")
    body = data.get("request")
    if not isinstance(body, dict):
        raise click.ClickException("reproduce response did not include a request body")
    if name:
        body["name"] = name
    command = _benchmark_reproduce_command(body)
    if rerun:
        created = request(ctx.obj["client"], "POST", "/benchmark-runs", json=body)
        if wait:
            if not ctx.obj["json"]:
                click.echo(f"created benchmark run {created['id']}")
            _wait_for_benchmark_run(
                ctx,
                str(created["id"]),
                interval=wait_interval,
                timeout=wait_timeout,
            )
            return
        emit(created, ctx.obj["json"], _benchmark_run_summary(created))
        return
    if ctx.obj["json"]:
        data["request"] = body
        data["cli_command"] = command
        emit(data, True, [])
        return
    click.echo("rerun command:")
    click.echo("  " + shlex.join(command))
    click.echo("request:")
    click.echo(json.dumps(body, indent=2))
    notes = data.get("notes")
    if isinstance(notes, list) and notes:
        click.echo("notes:")
        for note in notes:
            click.echo(f"  - {note}")


def _benchmark_reproduce_command(body: dict[str, object]) -> list[str]:
    required = ("name", "benchmark_id", "benchmark_revision", "runtime", "parallelism")
    missing = [key for key in required if body.get(key) is None]
    if missing:
        raise click.ClickException("reproduce response is missing required field(s): " + ", ".join(missing))
    command = [
        "scaled-evals",
        "benchmark-run",
        "create",
        "--name",
        str(body["name"]),
        "--benchmark-id",
        str(body["benchmark_id"]),
        "--benchmark-revision",
        str(body["benchmark_revision"]),
        "--framework",
        str(body.get("framework") or "harbor"),
        "--runtime",
        str(body["runtime"]),
        "--network-policy",
        str(body.get("network_policy") or "unrestricted"),
        "--n-attempts",
        str(body.get("n_attempts") or 1),
        "--parallelism",
        str(body["parallelism"]),
        "--visibility",
        str(body.get("visibility") or "private"),
    ]
    for option, key in (
        ("--framework-version", "framework_version"),
        ("--framework-profile-id", "framework_profile_id"),
        ("--switchyard-profile-id", "switchyard_profile_id"),
        ("--intake-profile-id", "intake_profile_id"),
        ("--agent-bundle", "agent_bundle_id"),
        ("--instruction-prefix", "instruction_prefix"),
        ("--instruction-postfix", "instruction_postfix"),
        ("--max-concurrent-members", "max_concurrent_members"),
    ):
        value = body.get(key)
        if value is not None:
            command.extend([option, str(value)])
    network_config = body.get("network_policy_config")
    if isinstance(network_config, dict) and network_config:
        command.extend(["--network-policy-config", json.dumps(network_config, separators=(",", ":"))])
    credentials = body.get("credentials")
    if isinstance(credentials, dict):
        for role, credential_id in sorted(credentials.items()):
            command.extend(["--credential", f"{role}={credential_id}"])
    for key, option in (
        ("extra_skill_object_keys", "--extra-skill-object-key"),
        ("initial_user_turns", "--initial-user-turn"),
    ):
        values = body.get(key)
        if isinstance(values, list):
            for value in values:
                command.extend([option, str(value)])
    return command


@benchmark_run.command("cancel")
@click.argument("run_id")
@click.pass_context
def benchmark_run_cancel(ctx: click.Context, run_id: str) -> None:
    """Cancel a benchmark run and its still-active member evaluations."""
    data = request(ctx.obj["client"], "POST", f"/benchmark-runs/{run_id}/cancel")
    emit(data, ctx.obj["json"], _benchmark_run_summary(data))


def _evaluation_summary(data: dict[str, Any]) -> list[str]:
    """Human-readable lines shared by evaluation create/get."""
    detail = f" ({data['status_detail']})" if data.get("status_detail") else ""
    lines = [
        f"evaluation {data['id']}",
        f"  name:      {data.get('name')}",
        f"  status:    {data.get('status')}{detail}",
        f"  task: {data.get('task_id')} rev {data.get('task_revision')}",
    ]
    if data.get("framework_version"):
        lines.append(
            f"  runner:    {data.get('framework')} {data['framework_version']} "
            f"(sandbox-k8s {data.get('sandbox_k8s_version') or 'n/a'})"
        )
    if data.get("benchmark_run_id"):
        lines.append(f"  benchmark run: {data['benchmark_run_id']}")
    outcome = data.get("outcome")
    if isinstance(outcome, dict):
        lines.append(f"  outcome:   {outcome.get('category')}")
    if data.get("reward") is not None:
        lines.append(f"  reward:    {data['reward']}")
    if data.get("n_trials") is not None:
        lines.append(f"  trials:    {data['n_trials']} ({data.get('n_errored')} errored)")
        details = []
        if data.get("n_completed") is not None:
            details.append(f"{data['n_completed']} completed")
        if data.get("n_failed_solve") is not None:
            details.append(f"{data['n_failed_solve']} failed solve")
        if details:
            lines.append(f"  result:    {', '.join(details)}")
    if isinstance(outcome, dict) and outcome.get("exception_counts"):
        exceptions = ", ".join(f"{name} x{count}" for name, count in outcome["exception_counts"].items())
        lines.append(f"  exceptions: {exceptions}")
    current_execution = data.get("current_execution")
    if isinstance(current_execution, int) and current_execution > 1:
        lines.append(f"  retries:    {current_execution - 1}")
    if data.get("finished_at"):
        lines.append(f"  finished:  {data['finished_at']}")
    return lines


_install_json_output_option(cli)


if __name__ == "__main__":
    cli()
