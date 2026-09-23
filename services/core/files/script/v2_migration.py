#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub",
#   "nemo-helix>=2.0.0.dev0,<2.1.0",
# ]
# ///
"""
Migrate artifacts from V1 datastore (hf://datasets) into V2 filesets.

This script follows a three-phase workflow:
1) setup: validate credentials/endpoints and optional connectivity checks
2) plan: discover datastore artifacts and generate a migration plan JSON
3) apply: execute a generated plan by downloading datastore files and uploading
   them into Files service filesets

Usage examples:
  nemo auth login --base-url <your-nhx-base-url>  # if auth is enabled
  uv run v2_migration.py setup --check --repo-prefix <namespace/>
  uv run v2_migration.py plan --repo-id <namespace/repo-a> --repo-id <namespace/repo-b> --output plan.json
  uv run v2_migration.py plan --repo-prefix <namespace/> --output plan.json
  uv run v2_migration.py apply --plan plan.json
  # Optional override: force all repos into a single workspace
  uv run v2_migration.py apply --files-workspace <target-workspace> --plan plan.json

Dependencies are embedded above for uv script execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi
from nemo_helix import NeMoHelix, NotFoundError
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError as ClientConflictError
from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, ListFilesetsQueryParams
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASET_PREFIX = "hf://datasets/"


@dataclass
class RuntimeConfig:
    datastore_url: str
    datastore_token: str | None
    dataset_prefix: str
    files_base_url: str | None
    files_workspace: str | None


@dataclass
class ArtifactPlan:
    source_path: str
    target_path: str
    size_bytes: int | None


@dataclass
class RepoPlan:
    repo_id: str
    target_workspace: str
    target_fileset: str
    artifacts: list[ArtifactPlan]
    total_files: int
    total_size_bytes: int


def _is_default_dataset_prefix(prefix: str) -> bool:
    return prefix.rstrip("/") == DEFAULT_HF_DATASET_PREFIX.rstrip("/")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sanitize_fileset_name(repo_id: str) -> str:
    """
    Convert namespace/repo into Files-service-safe name.

    Files API name regex (current deployment):
      ^[a-z](?!.*--)[a-z0-9\\-@.+_]{1,62}(?<!-)$
    """
    # Lowercase and replace path separators.
    name = repo_id.lower().replace("/", "-")
    # Keep only allowed characters; replace others with hyphen.
    name = re.sub(r"[^a-z0-9\-@.+_]+", "-", name)
    # Disallow repeated hyphens.
    name = re.sub(r"-{2,}", "-", name)
    # Name cannot end with '-'.
    name = name.strip("-")
    if not name:
        name = "migrated-fileset"
    # Must start with a letter.
    if not name[0].isalpha():
        name = f"f-{name}"
    # Max length is 63 characters.
    name = name[:63].rstrip("-")
    # Re-guard after truncation.
    if "--" in name:
        name = re.sub(r"-{2,}", "-", name).rstrip("-")
    if not name:
        name = "migrated-fileset"
    return name


def _resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    datastore_url = args.datastore_url or os.environ.get("DATASTORE_URL")
    datastore_token = args.datastore_token or os.environ.get("DATASTORE_TOKEN")
    dataset_prefix = (
        os.environ.get("HF_DATASET_PREFIX") or os.environ.get("DATASET_PREFIX") or DEFAULT_HF_DATASET_PREFIX
    )
    files_base_url = (
        args.files_base_url or os.environ.get("NEMO_MICROSERVICES_FILES_URL") or os.environ.get("NHX_BASE_URL")
    )
    files_workspace = args.files_workspace

    if not datastore_url:
        raise ValueError("Missing required configuration: DATASTORE_URL (or --datastore-url)")

    if not dataset_prefix.endswith("/"):
        dataset_prefix = f"{dataset_prefix}/"

    return RuntimeConfig(
        datastore_url=datastore_url,
        datastore_token=datastore_token,
        dataset_prefix=dataset_prefix,
        files_base_url=files_base_url,
        files_workspace=files_workspace,
    )


def _get_datastore_api(cfg: RuntimeConfig) -> HfApi:
    return HfApi(endpoint=cfg.datastore_url, token=cfg.datastore_token)


def _get_files_sdk(cfg: RuntimeConfig) -> NeMoHelix:
    kwargs: dict[str, Any] = {}
    if cfg.files_workspace:
        kwargs["workspace"] = cfg.files_workspace
    if cfg.files_base_url:
        kwargs["base_url"] = cfg.files_base_url
    return NeMoHelix(**kwargs)


def _resolve_target_workspace(repo_id: str, explicit_files_workspace: str | None) -> str:
    """Resolve target Files workspace.

    If --files-workspace is provided, use it for all repos.
    Otherwise infer from repo namespace: <namespace>/<repo> -> <namespace>.
    """
    if explicit_files_workspace:
        return explicit_files_workspace
    if "/" not in repo_id:
        raise ValueError(
            f"Cannot infer files workspace from repo_id '{repo_id}'. "
            "Expected 'namespace/repo' or provide --files-workspace."
        )
    namespace, _ = repo_id.split("/", 1)
    if not namespace:
        raise ValueError(
            f"Cannot infer files workspace from repo_id '{repo_id}'. Namespace is empty; provide --files-workspace."
        )
    return namespace


def _list_repo_ids(api: HfApi, explicit_repo_ids: list[str], repo_prefix: str | None, limit: int | None) -> list[str]:
    if explicit_repo_ids:
        return sorted(set(explicit_repo_ids))

    # By this point, run_plan has enforced a non-empty prefix.
    if not repo_prefix:
        raise ValueError("Internal error: repo_prefix must be provided when repo_ids are not explicit.")

    normalized_prefix = repo_prefix.strip()
    if not normalized_prefix:
        raise ValueError("Internal error: repo_prefix cannot be empty/whitespace.")

    # Support both:
    # - namespace-wide scan: "namespace/" (or "namespace")
    # - namespace fragment scan: "namespace/repo-fragment"
    if "/" in normalized_prefix:
        namespace, repo_fragment = normalized_prefix.split("/", 1)
    else:
        namespace, repo_fragment = normalized_prefix, ""

    if not namespace:
        raise ValueError("Internal error: repo_prefix must include a namespace.")

    list_kwargs: dict[str, Any] = {
        "full": False,
        "author": namespace,
    }
    if repo_fragment:
        list_kwargs["search"] = repo_fragment
    if limit is not None:
        list_kwargs["limit"] = limit

    repo_ids: list[str] = []
    prefix_for_match = normalized_prefix if normalized_prefix.endswith("/") else f"{normalized_prefix}/"
    if "/" in normalized_prefix and repo_fragment:
        prefix_for_match = normalized_prefix

    for dataset in api.list_datasets(**list_kwargs):
        repo_id = getattr(dataset, "id", None)
        if not repo_id:
            continue
        if not str(repo_id).startswith(prefix_for_match):
            continue
        repo_ids.append(str(repo_id))

    return sorted(set(repo_ids))


def _build_size_lookup(repo_info: Any) -> dict[str, int]:
    lookup: dict[str, int] = {}
    siblings = getattr(repo_info, "siblings", None) or []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if path is None or size is None:
            continue
        if isinstance(size, int):
            lookup[str(path)] = size
    return lookup


def create_plan(
    cfg: RuntimeConfig,
    *,
    repo_ids: list[str],
    repo_prefix: str | None,
    repo_limit: int | None,
) -> dict[str, Any]:
    api = _get_datastore_api(cfg)
    files_sdk = _get_files_sdk(cfg)
    resolved_files_endpoint = str(files_sdk.base_url)
    selected_repo_ids = _list_repo_ids(api, repo_ids, repo_prefix, repo_limit)

    repo_plans: list[RepoPlan] = []
    skipped_repos: list[dict[str, str]] = []
    total_files = 0
    total_size = 0

    for repo_id in selected_repo_ids:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as exc:
            logger.warning("Skipping repo %s due to list failure: %s", repo_id, exc)
            skipped_repos.append({"repo_id": repo_id, "reason": f"list failed: {exc}"})
            continue

        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
            size_lookup = _build_size_lookup(repo_info)
        except Exception as exc:
            logger.warning("Continuing without file sizes for %s: %s", repo_id, exc)
            size_lookup = {}

        artifacts: list[ArtifactPlan] = []
        repo_size = 0

        for path in sorted(repo_files):
            size = size_lookup.get(path)
            if size is not None:
                repo_size += size
            artifacts.append(
                ArtifactPlan(
                    source_path=path,
                    target_path=path,
                    size_bytes=size,
                )
            )

        target_workspace = _resolve_target_workspace(repo_id, cfg.files_workspace)
        target_fileset = _sanitize_fileset_name(repo_id)
        repo_plan = RepoPlan(
            repo_id=repo_id,
            target_workspace=target_workspace,
            target_fileset=target_fileset,
            artifacts=artifacts,
            total_files=len(artifacts),
            total_size_bytes=repo_size,
        )
        repo_plans.append(repo_plan)
        total_files += len(artifacts)
        total_size += repo_size

    return {
        "generated_at": _now_iso(),
        "source": {
            "type": "datastore_hf_datasets",
            "endpoint": cfg.datastore_url,
            "prefix": cfg.dataset_prefix,
        },
        "target": {
            "type": "files_v2",
            "endpoint": resolved_files_endpoint,
            "workspace": cfg.files_workspace or "inferred_from_repo_namespace",
        },
        "summary": {
            "repo_count": len(repo_plans),
            "artifact_count": total_files,
            "total_size_bytes": total_size,
            "skipped_repo_count": len(skipped_repos),
        },
        "repos": [
            {
                **asdict(repo),
                "source_url": f"{cfg.dataset_prefix}{repo.repo_id}",
            }
            for repo in repo_plans
        ],
        "skipped_repos": skipped_repos,
    }


def _ensure_fileset(
    sdk: NeMoHelix, workspace: str, fileset: str, dry_run: bool
) -> Literal["dry_run", "exists", "created"]:
    """
    Ensure that the target fileset exists, and create it if it doesn't.
    """
    if dry_run:
        return "dry_run"
    files = client_from_platform(sdk, FilesClient)
    try:
        files.get_fileset(name=fileset, workspace=workspace)
        return "exists"
    except ClientNotFoundError:
        files.create_fileset(body=CreateFilesetRequest(name=fileset), workspace=workspace)
        return "created"


def _ensure_workspace(sdk: NeMoHelix, workspace: str, dry_run: bool) -> Literal["dry_run", "exists", "created"]:
    """
    Ensure that the target workspace exists, and create it if it doesn't.
    """
    workspaces = client_from_platform(sdk, WorkspacesClient)
    if dry_run:
        return "dry_run"
    try:
        workspaces.get_workspace(name=workspace).data()
        return "exists"
    except ClientNotFoundError:
        try:
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        except ClientConflictError:
            # Another actor may have created the workspace concurrently.
            return "exists"
        return "created"


def _get_existing_target_paths(sdk: NeMoHelix, workspace: str, fileset: str) -> set[str]:
    """
    Return existing file paths in the target fileset.

    If the fileset does not exist yet, return an empty set.
    """
    try:
        files = client_from_platform(sdk, FilesClient).list_files(name=fileset, workspace=workspace).data().data
        return {f.path for f in files}
    except NotFoundError:
        return set()


def apply_plan(
    cfg: RuntimeConfig,
    plan: dict[str, Any],
    *,
    dry_run: bool,
    max_repos: int | None,
) -> dict[str, Any]:
    api = _get_datastore_api(cfg)
    sdk = _get_files_sdk(cfg)

    repo_entries: list[dict[str, Any]] = list(plan.get("repos", []))
    if max_repos is not None:
        repo_entries = repo_entries[:max_repos]

    results: list[dict[str, Any]] = []
    uploaded = 0
    skipped = 0
    failed = 0

    for repo in repo_entries:
        repo_id = str(repo["repo_id"])
        workspace = str(repo["target_workspace"])
        fileset = str(repo["target_fileset"])
        artifacts: list[dict[str, Any]] = list(repo.get("artifacts", []))

        repo_result = {
            "repo_id": repo_id,
            "target": f"{workspace}/{fileset}",
            "workspace_status": "unknown",
            "fileset_status": "unknown",
            "artifacts": [],
        }

        try:
            repo_result["workspace_status"] = _ensure_workspace(sdk, workspace, dry_run=dry_run)
        except Exception as exc:
            repo_result["workspace_status"] = f"failed: {exc}"
            failed += len(artifacts)
            for artifact in artifacts:
                repo_result["artifacts"].append(
                    {
                        "source_path": artifact["source_path"],
                        "target_path": artifact["target_path"],
                        "status": "failed",
                        "error": f"workspace create/retrieve failed: {exc}",
                    }
                )
            results.append(repo_result)
            continue

        try:
            repo_result["fileset_status"] = _ensure_fileset(sdk, workspace, fileset, dry_run=dry_run)
        except Exception as exc:
            repo_result["fileset_status"] = f"failed: {exc}"
            failed += len(artifacts)
            for artifact in artifacts:
                repo_result["artifacts"].append(
                    {
                        "source_path": artifact["source_path"],
                        "target_path": artifact["target_path"],
                        "status": "failed",
                        "error": f"fileset create/retrieve failed: {exc}",
                    }
                )
            results.append(repo_result)
            continue

        existing_paths = _get_existing_target_paths(sdk, workspace, fileset)
        repo_result["existing_target_file_count"] = len(existing_paths)

        with tempfile.TemporaryDirectory(prefix="v2-migration-") as tmpdir:
            local_root = Path(tmpdir)
            for artifact in artifacts:
                source_path = str(artifact["source_path"])
                target_path = str(artifact["target_path"])
                artifact_result = {
                    "source_path": source_path,
                    "target_path": target_path,
                    "status": "unknown",
                }
                try:
                    if target_path in existing_paths:
                        artifact_result["status"] = "skipped_exists"
                        skipped += 1
                    elif dry_run:
                        artifact_result["status"] = "would_upload"
                        skipped += 1
                    else:
                        local_file = api.hf_hub_download(
                            repo_id=repo_id,
                            filename=source_path,
                            local_dir=str(local_root / repo_id),
                            repo_type="dataset",
                        )
                        with open(local_file, "rb") as fh:
                            client_from_platform(sdk, FilesClient).upload_file(
                                content=fh,
                                name=fileset,
                                workspace=workspace,
                                path=target_path,
                            )
                        artifact_result["status"] = "uploaded"
                        uploaded += 1
                        # Keep this set up to date for duplicate target paths in the same plan.
                        existing_paths.add(target_path)
                except Exception as exc:
                    artifact_result["status"] = "failed"
                    artifact_result["error"] = str(exc)
                    failed += 1
                repo_result["artifacts"].append(artifact_result)
        results.append(repo_result)

    return {
        "applied_at": _now_iso(),
        "dry_run": dry_run,
        "summary": {
            "repo_count": len(results),
            "uploaded_artifacts": uploaded,
            "skipped_artifacts": skipped,
            "failed_artifacts": failed,
        },
        "results": results,
    }


def run_setup(args: argparse.Namespace) -> int:
    cfg = _resolve_runtime_config(args)
    files_mode = "explicit base URL/env" if cfg.files_base_url else "nemo CLI context/config"
    print("Resolved configuration:")
    print(f"  datastore_url: {cfg.datastore_url}")
    print(f"  datastore_token: {'set' if cfg.datastore_token else 'not set'}")
    print(f"  dataset_prefix: {cfg.dataset_prefix}")
    print(f"  files_base_url: {cfg.files_base_url or '<from nemo context>'}")
    print(f"  files_auth_mode: {files_mode}")
    print(f"  files_workspace: {cfg.files_workspace or '<inferred from datastore repo namespace>'}")
    print("  auth_note: If auth is enabled, run 'nemo auth login' before using this script.")

    if not args.check:
        return 0

    requested_repo_ids: list[str] = list(getattr(args, "repo_id", []) or [])
    requested_repo_prefix: str | None = getattr(args, "repo_prefix", None)
    if _is_default_dataset_prefix(cfg.dataset_prefix) and not requested_repo_ids and not requested_repo_prefix:
        raise ValueError(
            "Setup check requires scope when using default dataset prefix "
            f"'{DEFAULT_HF_DATASET_PREFIX}'. Provide --repo-id or --repo-prefix "
            "to avoid broad scans."
        )

    print("\nRunning connectivity checks...")
    try:
        datastore = _get_datastore_api(cfg)
        requested_repo_limit: int | None = getattr(args, "repo_limit", None)

        if requested_repo_ids:
            for repo_id in requested_repo_ids:
                files = datastore.list_repo_files(repo_id=repo_id, repo_type="dataset")
                print(f"  datastore: OK (repo={repo_id}, files={len(files)})")
        elif requested_repo_prefix:
            matched_repo_ids = _list_repo_ids(
                datastore,
                explicit_repo_ids=[],
                repo_prefix=requested_repo_prefix,
                limit=requested_repo_limit,
            )
            print(f"  datastore: OK (prefix={requested_repo_prefix}, matched_repos={len(matched_repo_ids)})")
        else:
            first = next(iter(datastore.list_datasets(full=False)), None)
            sample = getattr(first, "id", "<none visible>")
            print(f"  datastore: OK (sample repo: {sample})")
    except Exception as exc:
        print(f"  datastore: FAIL ({exc})")
        return 1

    try:
        sdk = _get_files_sdk(cfg)
        files = client_from_platform(sdk, FilesClient)
        # Lightweight Files API connectivity check against default workspace.
        files.list_filesets(workspace="default", query_params=ListFilesetsQueryParams(page_size=1))
        print(f"  files service: OK (resolved base_url: {sdk.base_url}, check_workspace=default)")
    except Exception as exc:
        print(f"  files service: FAIL ({exc})")
        return 1

    return 0


def run_plan(args: argparse.Namespace) -> int:
    cfg = _resolve_runtime_config(args)
    repo_ids: list[str] = args.repo_id or []
    repo_prefix: str | None = args.repo_prefix

    # Enforce explicit scoping to avoid expensive global scans.
    if not repo_ids and not repo_prefix:
        raise ValueError(
            "Plan requires scope. Provide --repo-id (repeatable) or "
            "--repo-prefix in the form 'namespace/' or 'namespace/repo-fragment'."
        )
    if repo_prefix is not None and not repo_prefix.strip():
        raise ValueError("Invalid --repo-prefix. Prefix cannot be empty.")
    if repo_prefix is not None:
        # Accept either namespace-only ("namespace" or "namespace/")
        # or namespace+fragment ("namespace/repo-fragment").
        normalized_prefix = repo_prefix.strip()
        if normalized_prefix.startswith("/"):
            raise ValueError("Invalid --repo-prefix. Use 'namespace/' or 'namespace/repo-fragment'.")

    plan = create_plan(
        cfg,
        repo_ids=repo_ids,
        repo_prefix=repo_prefix,
        repo_limit=args.repo_limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    summary = plan["summary"]
    print(f"Wrote plan: {output}")
    print(
        "Summary: "
        f"{summary['repo_count']} repos, "
        f"{summary['artifact_count']} artifacts, "
        f"{summary['total_size_bytes']} bytes"
    )
    if summary["skipped_repo_count"] > 0:
        print(f"Skipped repos: {summary['skipped_repo_count']}")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    cfg = _resolve_runtime_config(args)
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    result = apply_plan(
        cfg,
        plan,
        dry_run=args.dry_run,
        max_repos=args.max_repos,
    )

    result_path = Path(args.result_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = result["summary"]
    print(f"Wrote apply result: {result_path}")
    print(
        "Apply summary: "
        f"{summary['uploaded_artifacts']} uploaded, "
        f"{summary['skipped_artifacts']} skipped, "
        f"{summary['failed_artifacts']} failed"
    )
    return 0 if summary["failed_artifacts"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate V1 datastore artifacts to V2 filesets")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--datastore-url", default=None, help="Datastore HF endpoint (or DATASTORE_URL)")
    common.add_argument("--datastore-token", default=None, help="Datastore token (or DATASTORE_TOKEN)")
    common.add_argument(
        "--files-base-url",
        default=None,
        help="Files service base URL (or NEMO_MICROSERVICES_FILES_URL / NHX_BASE_URL). If omitted, script uses active nemo context/config.",
    )
    common.add_argument(
        "--files-workspace",
        default=None,
        help="Target V2 files workspace for filesets. If omitted, inferred from datastore repo namespace.",
    )
    common.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser(
        "setup",
        help="Validate runtime config and optional connectivity",
        parents=[common],
    )
    p_setup.add_argument("--check", action="store_true", help="Run connectivity checks")
    p_setup.add_argument(
        "--repo-id",
        action="append",
        default=[],
        help="Specific datastore repo_id to validate access for (repeatable).",
    )
    p_setup.add_argument(
        "--repo-prefix",
        default=None,
        help="Datastore prefix to validate access for (e.g., namespace/ or namespace/repo-fragment).",
    )
    p_setup.add_argument(
        "--repo-limit",
        type=int,
        default=None,
        help="Limit discovered repos when validating with --repo-prefix.",
    )
    p_setup.set_defaults(func=run_setup)

    p_plan = sub.add_parser(
        "plan",
        help="Generate migration plan JSON",
        parents=[common],
    )
    p_plan.add_argument(
        "--repo-id",
        action="append",
        default=[],
        help="Explicit repo_id to include (repeatable)",
    )
    p_plan.add_argument(
        "--repo-prefix",
        default=None,
        help="Filter discovered repos by prefix. Supports 'namespace/' (all repos in namespace) or 'namespace/repo-fragment'.",
    )
    p_plan.add_argument("--repo-limit", type=int, default=None, help="Limit number of discovered repos")
    p_plan.add_argument(
        "--output",
        default="./v2_migration_plan.json",
        help="Path to output plan JSON (default: ./v2_migration_plan.json)",
    )
    p_plan.set_defaults(func=run_plan)

    p_apply = sub.add_parser(
        "apply",
        help="Apply a migration plan",
        parents=[common],
    )
    p_apply.add_argument("--plan", required=True, help="Path to plan JSON generated by 'plan'")
    p_apply.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate uploads without writing to Files service",
    )
    p_apply.add_argument("--max-repos", type=int, default=None, help="Apply only first N repos from plan")
    p_apply.add_argument(
        "--result-output",
        default="./v2_migration_apply_result.json",
        help="Path to output apply result JSON (default: ./v2_migration_apply_result.json)",
    )
    p_apply.set_defaults(func=run_apply)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
