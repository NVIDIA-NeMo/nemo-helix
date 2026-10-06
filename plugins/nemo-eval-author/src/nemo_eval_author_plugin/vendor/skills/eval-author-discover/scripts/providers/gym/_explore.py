# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the NeMo Gym artifacts a repository owns.

Standard library only. Gym holds two kinds of artifact that discovery treats
differently:

- A Gym **manifest** (``manifest.yaml``) describes a resources-server environment
  or benchmark backed by JSONL rows. It declares a ``datasets`` list, so the Harbor
  inventory would otherwise mistake it for a job config. Manifests are inventoried
  and never converted, because Gym has no Harbor mapping for them.
- A Gym **extension task** is a Harbor task directory that uses one of Gym's
  authoring extensions, which Harbor rejects until Gym renders it:
  ``tests/verifier.py`` in place of ``tests/test.sh``, or a parameterized directory
  whose ``tasks.jsonl`` rows fill ``instruction.template.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from providers.harbor._explore import ConfigCandidate

PYTHON_VERIFIER = "gym-python-verifier"
PARAMETERIZED = "gym-parameterized"

_MANIFEST_KEY_PATTERN = re.compile(r"^(?:integration_profile:|\s*-?\s*jsonl_fpath:)", re.MULTILINE)


@dataclass(frozen=True)
class GymManifest:
    """A Gym resources-server environment or benchmark.

    ``task_count`` counts the rows in the data files that exist. ``missing_data``
    names the declared files that do not, such as a benchmark nobody has prepared.
    """

    path: Path
    name: str | None
    kind: str | None
    domain: str | None
    dataset_count: int
    task_count: int = 0
    missing_data: tuple[str, ...] = ()


def is_manifest(candidate: ConfigCandidate) -> bool:
    """Return whether a config candidate is a Gym manifest rather than a Harbor job config."""
    if candidate.parsed:
        data = candidate.data
        datasets = data.get("datasets")
        return "integration_profile" in data or (
            isinstance(datasets, list) and any(isinstance(item, dict) and "jsonl_fpath" in item for item in datasets)
        )
    try:
        text = candidate.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return bool(_MANIFEST_KEY_PATTERN.search(text))


def manifest(candidate: ConfigCandidate, repo_root: Path) -> GymManifest:
    """Describe a Gym manifest from its parsed fields, counting the rows of its data files.

    Gym resolves each ``jsonl_fpath`` against the repository root.
    """
    data: dict[str, Any] = candidate.data
    datasets = data.get("datasets")
    declared = [
        item["jsonl_fpath"]
        for item in (datasets if isinstance(datasets, list) else [])
        if isinstance(item, dict) and isinstance(item.get("jsonl_fpath"), str)
    ]
    present = [path for path in declared if (repo_root / path).is_file()]

    def text(key: str) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) else None

    return GymManifest(
        path=candidate.path,
        name=text("name"),
        kind=text("kind"),
        domain=text("domain"),
        dataset_count=len(datasets) if isinstance(datasets, list) else 0,
        task_count=sum(jsonl_rows(repo_root / path) for path in present),
        missing_data=tuple(path for path in declared if path not in present),
    )


def jsonl_rows(path: Path) -> int:
    """Count the non-blank lines of a JSONL file, or 0 when it cannot be read."""
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeError):
        return 0


def is_cataloged(item: GymManifest) -> bool:
    """Return whether Gym's catalog lists this manifest, so ``gym env test <name>`` can find it.

    Gym catalogs ``manifest.yaml`` files under an ``environments/`` or ``benchmarks/``
    directory that matches their kind. A manifest elsewhere, such as a documentation
    example, is not a workload Gym can test by name.
    """
    if not item.name or item.kind not in {"environment", "benchmark"} or item.path.name != "manifest.yaml":
        return False
    return any(parent.name == "{}s".format(item.kind) for parent in item.path.parents)


def is_environments_root(directory: Path) -> bool:
    """Return whether a directory is where Gym looks for environments, each of which stands alone."""
    return directory.name == "environments"


def extension_kind(task_dir: Path) -> str | None:
    """Return the Gym extension a task directory uses, or None for a plain Harbor task."""
    if (task_dir / "tasks.jsonl").is_file() and (task_dir / "instruction.template.md").is_file():
        return PARAMETERIZED
    if (task_dir / "tests" / "verifier.py").is_file():
        return PYTHON_VERIFIER
    return None
