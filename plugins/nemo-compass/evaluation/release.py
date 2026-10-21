# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State-version resolution and CSS S3 storage operations."""

import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path

from evaluation.registry import Subject

REGISTRY_PATH = Path(__file__).parent / "evaluations.toml"
ACCESS_KEY_ENV = "INSIGHTS_EVALUATION_S3_ACCESS_KEY"
SECRET_KEY_ENV = "INSIGHTS_EVALUATION_S3_SECRET_KEY"
_ASSET = re.compile(r"^state-v(\d+)\.tar\.zst$")


@dataclasses.dataclass(frozen=True)
class StateStore:
    """The non-secret CSS S3 location committed in ``evaluations.toml``."""

    endpoint: str
    region: str
    bucket: str


class StateRefConflict(RuntimeError):
    """A conditional upload found that another publisher already created the ref."""


def state_store(path: Path = REGISTRY_PATH) -> StateStore:
    """Load the CSS S3 endpoint, region, and bucket from the registry."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    values = {
        "endpoint": data.get("state_s3_endpoint"),
        "region": data.get("state_s3_region"),
        "bucket": data.get("state_s3_bucket"),
    }
    missing = [f"state_s3_{key}" for key, value in values.items() if not value]
    if missing:
        sys.exit(f"{path} is missing required state storage config: {', '.join(missing)}")
    return StateStore(**{key: str(value) for key, value in values.items()})


def latest_ref(names: list[str]) -> str | None:
    """Find the latest state version from a list of object keys."""
    versions = [int(m.group(1)) for n in names if (m := _ASSET.match(n))]
    return f"state-v{max(versions)}" if versions else None


def next_ref(latest: str | None) -> str:
    """Generate the next state version ref after the given latest."""
    return f"state-v{int(latest.removeprefix('state-v')) + 1}" if latest else "state-v1"


def pinned_ref(subject: Subject) -> str | None:
    """Return a subject's pinned state ref from its ``evaluations.toml`` stanza."""
    value = subject.config.get("state")
    return None if value is None else str(value)


def _aws(store: StateStore, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run an authenticated AWS CLI command against the configured CSS endpoint."""
    values = os.environ if env is None else env
    access_key = values.get(ACCESS_KEY_ENV)
    secret_key = values.get(SECRET_KEY_ENV)
    if not access_key or not secret_key:
        missing = [key for key, value in ((ACCESS_KEY_ENV, access_key), (SECRET_KEY_ENV, secret_key)) if not value]
        sys.exit(
            f"missing CSS S3 credentials: {', '.join(missing)} — add them to evaluation/.env "
            "(use the S3 Secret from CSS Portal → Auth Info, not the namespace password)"
        )
    command_env = dict(os.environ)
    command_env["AWS_ACCESS_KEY_ID"] = access_key
    command_env["AWS_SECRET_ACCESS_KEY"] = secret_key
    command_env["AWS_DEFAULT_REGION"] = store.region
    command_env["AWS_EC2_METADATA_DISABLED"] = "true"
    command_env.pop("AWS_SESSION_TOKEN", None)
    command_env.pop("AWS_SECURITY_TOKEN", None)
    try:
        return subprocess.run(
            ["aws", "--endpoint-url", store.endpoint, "--region", store.region, *args],
            check=True,
            capture_output=True,
            text=True,
            env=command_env,
        ).stdout
    except FileNotFoundError:
        sys.exit("AWS CLI not found — install it with `brew install awscli`")
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr, end="")
        raise


def object_names(*, store: StateStore | None = None) -> list[str]:
    """List state object keys in the CSS bucket."""
    store = store or state_store()
    output = _aws(store, "s3api", "list-objects-v2", "--bucket", store.bucket, "--output", "json")
    payload = json.loads(output)
    return [str(item["Key"]) for item in payload.get("Contents", [])]


def resolve_state(state: str | None, *, subject: Subject | None) -> str:
    """Resolve the state ref to restore: an explicit ref or a subject stanza's pin.

    An explicit *state* must be a published ref (``state-v<N>``) — local bundle
    files are the caller's business (the CLI detects existing paths before this
    is reached). ``state=None`` reads the subject's ``state`` key. A missing pin
    or subject-less caller is a hard error.
    """
    if state is not None:
        if not re.fullmatch(r"state-v\d+", state):
            sys.exit(
                f"invalid state ref '{state}' — expected state-v<N> (e.g. state-v6); a local bundle "
                "file goes through --state FILE on analyze, or the positional FILE on restore"
            )
        return state
    if subject is not None and (pinned := pinned_ref(subject)):
        if not re.fullmatch(r"state-v\d+", pinned):
            sys.exit(f"evaluations.toml state for '{subject.name}' is '{pinned}' — expected state-v<N> (e.g. state-v6)")
        return pinned
    name = subject.name if subject is not None else None
    sys.exit(
        f"no state configured in evaluations.toml for subject '{name}' — add state = \"state-vN\" "
        "to its stanza after publishing a fixture, or pass an explicit state "
        "(analyze: --live / --state <state-vN|FILE>; restore: FILE / --state state-vN)"
    )


def download_ref(ref: str, dest_dir: Path, *, store: StateStore | None = None) -> Path:
    """Download and cache a state tarball from CSS S3.

    Published refs are immutable, so an existing complete cache entry is reused.
    Fresh downloads land in a partial file and are atomically promoted so an
    interrupted AWS CLI process cannot poison the cache.
    """
    store = store or state_store()
    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{store.endpoint}__{store.bucket}")
    cache_dir = dest_dir / namespace
    dest = cache_dir / f"{ref}.tar.zst"
    if dest.is_file():
        print(f"using cached {ref}.tar.zst")
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".partial", dir=cache_dir)
    os.close(descriptor)
    partial = Path(partial_name)
    partial.unlink()
    try:
        _aws(
            store,
            "s3",
            "cp",
            f"s3://{store.bucket}/{ref}.tar.zst",
            str(partial),
        )
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def upload_ref(
    ref: str,
    bundle: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    store: StateStore | None = None,
) -> None:
    """Upload one immutable state bundle to CSS S3."""
    store = store or state_store()
    args = [
        "s3",
        "cp",
        str(bundle),
        f"s3://{store.bucket}/{ref}.tar.zst",
        "--no-overwrite",
    ]
    if metadata:
        args += ["--metadata", json.dumps(dict(metadata), separators=(",", ":"))]
    try:
        _aws(store, *args)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if any(marker in stderr for marker in ("PreconditionFailed", "ConditionalRequestConflict", "412", "409")):
            raise StateRefConflict(ref) from exc
        raise
    expected_sha256 = (metadata or {}).get("sha256")
    if expected_sha256 is not None:
        head = json.loads(
            _aws(store, "s3api", "head-object", "--bucket", store.bucket, "--key", f"{ref}.tar.zst", "--output", "json")
        )
        if head.get("Metadata", {}).get("sha256") != expected_sha256:
            raise StateRefConflict(ref)
