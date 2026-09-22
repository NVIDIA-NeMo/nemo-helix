# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
from pathlib import Path

RENAME_SCRIPT = Path("tools/rename/rename-to-nemo-helix.sh")
RENAME_IMPL = Path("tools/rename/rename_to_nemo_helix.py")
VERIFY_SCRIPT = Path("tools/rename/verify-nemo-helix-rename.sh")
VERIFY_IMPL = Path("tools/rename/verify_nemo_helix_rename.py")
COMMON_IMPL = Path("tools/rename/rename_common.py")
SCRIPT_PATHS = {RENAME_SCRIPT, RENAME_IMPL, VERIFY_SCRIPT, VERIFY_IMPL, COMMON_IMPL}

PRODUCT_REPLACEMENTS = [
    ("NeMo Platform", "NeMo Helix"),
    ("NeMo platform", "NeMo Helix"),
    ("NeMoPlatform", "NeMoHelix"),
    ("NeMo-Platform", "NeMo-Helix"),
    ("NeMo-platform", "NeMo-Helix"),
    ("Nemo Platform", "Nemo Helix"),
    ("NemoPlatform", "NemoHelix"),
    ("Nemo-Platform", "Nemo-Helix"),
    ("Nemo-platform", "Nemo-Helix"),
    ("nemo platform", "nemo helix"),
    ("nemoplatform", "nemohelix"),
    ("nemoPlatform", "nemoHelix"),
    ("nemo-platform", "nemo-helix"),
    ("nemo_platform", "nemo_helix"),
    ("NEMO PLATFORM", "NEMO HELIX"),
    ("NEMO Platform", "NEMO Helix"),
    ("NEMO-PLATFORM", "NEMO-HELIX"),
    ("NEMO_PLATFORM", "NEMO_HELIX"),
]

ACRONYM_REPLACEMENTS = {
    "NMP": "NHX",
    "Nmp": "Nhx",
    "nmp": "nhx",
}

REPLACEMENTS = [*PRODUCT_REPLACEMENTS, *ACRONYM_REPLACEMENTS.items()]
ACRONYM_PATTERN = re.compile(r"(?<![A-Za-z0-9])(NMP|Nmp|nmp)(?![A-Za-z0-9])")

# These are first-party published image names that predate the common prefix.
UNPREFIXED_IMAGES = [
    "auditor-tasks",
    "guardrails-callout-mock-llm",
    "guardrails-callout",
    "safe-synthesizer-tasks",
]
IMAGE_PREFIX = "nhx-"
IMAGE_PATTERN = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(image) for image in UNPREFIXED_IMAGES) + r")")
BAKE_IMAGE_PATTERN = re.compile(r'(?:sha_and_maybe_latest_tags|base_tags)\("([^"]+)"\)')
LEGACY_PRODUCT_PATTERN = re.compile(r"nemo[ _-]?platform", re.IGNORECASE)
LEGACY_ACRONYM_PATTERN = ACRONYM_PATTERN


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=check, text=True, capture_output=True)


def repo_root(repo_dir: Path = Path(".")) -> Path:
    return Path(
        subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )


def git_paths(*args: str) -> list[Path]:
    output = run_git(*args).stdout
    return [Path(path) for path in output.split("\0") if path]


def tracked_paths() -> list[Path]:
    return git_paths("ls-files", "-z")


def git_file_set() -> list[Path]:
    return git_paths("ls-files", "-z", "--cached", "--others", "--exclude-standard")


def content_paths() -> list[Path]:
    return [path for path in git_file_set() if path not in SCRIPT_PATHS]


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def replace_acronyms(text: str) -> str:
    return ACRONYM_PATTERN.sub(lambda match: ACRONYM_REPLACEMENTS[match.group(1)], text)


def replace_legacy_names(text: str) -> str:
    updated = text
    for old, new in PRODUCT_REPLACEMENTS:
        updated = updated.replace(old, new)
    return replace_acronyms(updated)


def renamed_path(path: Path) -> Path:
    renamed = replace_legacy_names(path.as_posix())
    for image in UNPREFIXED_IMAGES:
        prefixed = f"{IMAGE_PREFIX}{image}"
        if prefixed not in renamed:
            renamed = renamed.replace(image, prefixed)
    return Path(renamed)
