# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove stale pip metadata while retaining the distribution matching imported code."""

import importlib.metadata
import shutil
from pathlib import Path

import pip


def cleanup_pip_metadata(site_packages: Path, code_version: str) -> None:
    candidates = sorted(site_packages.glob("pip-*.dist-info"))
    matching = []
    for path in candidates:
        distribution = importlib.metadata.PathDistribution(path)
        if distribution.metadata.get("Name", "").lower() == "pip" and distribution.version == code_version:
            matching.append(path)
    # Fail before deleting anything if the installed code has no unambiguous
    # metadata. Removing all metadata or relabeling another version is not repair.
    if len(matching) != 1:
        raise RuntimeError(f"Expected one pip metadata directory matching code {code_version}: {matching}")
    keep = matching[0]
    for path in candidates:
        if path == keep:
            continue
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
        print(f"Removed stale pip metadata: {path}", flush=True)
    remaining = list(site_packages.glob("pip-*.dist-info"))
    if remaining != [keep]:
        raise RuntimeError(f"Expected exactly one pip metadata directory: {remaining}")


def main() -> None:
    # Keep the import location rather than resolving individual package-file
    # symlinks into a shared cache. Only this environment's metadata is cleaned.
    site_packages = Path(pip.__file__).parent.parent
    cleanup_pip_metadata(site_packages, pip.__version__)
    metadata_version = importlib.metadata.version("pip")
    if metadata_version != pip.__version__:
        raise RuntimeError(f"pip code {pip.__version__} disagrees with metadata {metadata_version}")
    print(f"Validated exactly one pip distribution: {pip.__version__}", flush=True)


if __name__ == "__main__":
    main()
