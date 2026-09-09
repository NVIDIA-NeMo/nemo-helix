# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the package version to inject into dynamic-versioned wheel builds."""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

Cadence = Literal["nightly", "prerelease", "release"]
SEMVER_CORE_PATTERN = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
# Tags stay SemVer (`1.0.0-rc0`); wheels are PEP 440 (`1.0.0rc0`). The two spellings meet here.
PRERELEASE_SUFFIX_PATTERN = r"(?:a|b|rc)(?:0|[1-9][0-9]*)"
RELEASE_CORE_TAG_PATTERN = re.compile(rf"^({SEMVER_CORE_PATTERN})(?:-{PRERELEASE_SUFFIX_PATTERN})?$")
PRERELEASE_LABEL_PATTERN = re.compile(rf"^({SEMVER_CORE_PATTERN})-({PRERELEASE_SUFFIX_PATTERN})$")


class StampError(Exception):
    """Raised when the SDK version cannot be resolved safely."""


def safe_sdk_id(sdk_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", sdk_id) or sdk_id in {".", ".."}:
        raise StampError(f"selected SDK id must be a safe single path segment: {sdk_id}")
    return sdk_id


def _semver_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def latest_release_core(source_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(source_root), "tag", "--merged", "HEAD", "--list"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None

    release_cores = []
    for tag in result.stdout.splitlines():
        match = RELEASE_CORE_TAG_PATTERN.fullmatch(tag)
        if match:
            release_cores.append(match.group(1))
    if not release_cores:
        return None
    return max(release_cores, key=_semver_key)


def resolve_sdk_version(
    cadence: Cadence,
    release_label: str,
    nightly_timestamp: str,
    source_root: Path,
) -> str:
    if cadence == "nightly":
        base_version = latest_release_core(source_root) or "0.0.0"
        if not re.fullmatch(r"\d{14}", nightly_timestamp):
            raise StampError("nightly timestamp must be YYYYMMDDHHMMSS")
        return f"{base_version}.dev{nightly_timestamp}"

    if cadence == "prerelease":
        match = PRERELEASE_LABEL_PATTERN.fullmatch(release_label)
        if not match:
            raise StampError(f"pre-release label must look like 1.0.0-a1, 1.0.0-b1 or 1.0.0-rc0: {release_label}")
        return f"{match.group(1)}{match.group(2)}"

    if cadence == "release":
        if not re.fullmatch(SEMVER_CORE_PATTERN, release_label):
            raise StampError(f"stable release label must be SemVer core MAJOR.MINOR.PATCH: {release_label}")
        return release_label

    raise StampError(f"unsupported cadence for SDK version stamping: {cadence}")


def stamp_sdk_version(
    source_root: Path,
    sdk_id: str,
    cadence: Cadence,
    release_label: str,
    nightly_timestamp: str,
) -> str:
    safe_sdk_id(sdk_id)
    return resolve_sdk_version(
        cadence=cadence,
        release_label=release_label,
        nightly_timestamp=nightly_timestamp,
        source_root=source_root,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sdk-id", required=True)
    parser.add_argument("--cadence", required=True, choices=["nightly", "prerelease", "release"])
    # --release-label is unused for cadence=nightly; cadence-specific validation
    # in resolve_sdk_version() rejects empty/missing labels for prerelease and release.
    parser.add_argument("--release-label", default="")
    parser.add_argument("--nightly-timestamp", default="")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print only the resolved version on stdout; send the human banner to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        sdk_version = stamp_sdk_version(
            source_root=args.source_root,
            sdk_id=args.sdk_id,
            cadence=args.cadence,
            release_label=args.release_label,
            nightly_timestamp=args.nightly_timestamp,
        )
    except StampError as error:
        print(error, file=sys.stderr)
        return 1

    if args.print_version:
        # Machine-readable mode: human banner to stderr, just the version on stdout.
        print(f"Resolved sdk:{args.sdk_id} version {sdk_version}.", file=sys.stderr)
        print(sdk_version)
    else:
        print(f"Resolved sdk:{args.sdk_id} version {sdk_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
