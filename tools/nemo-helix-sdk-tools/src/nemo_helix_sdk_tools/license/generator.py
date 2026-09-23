# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
License report generator using osv-scanner.

This module handles the generation of license reports by:
1. Running osv-scanner on lockfiles
2. Formatting the JSON output into human-readable tables
3. Running scans in parallel for multiple projects
"""

import json
import logging
import shlex
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import requests
from nemo_helix_sdk_tools.license.format_osv_licenses import format_licenses_table
from nemo_helix_sdk_tools.license.formats import get_formatter
from nemo_helix_sdk_tools.license.license_utils import (
    ALLOWED_LICENSES,
    get_local_packages,
    get_override_key_for_package,
    normalize_package_name,
    resolve_license,
)
from packaging.requirements import InvalidRequirement, Requirement

logger = logging.getLogger(__name__)

OVERRIDES_FILE = overrides_file = Path(__file__).parent / "overrides.yaml"
_PYPI_JSON_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
FORMULA_PREFIXES = ("=", "+", "-", "@")
SAFE_URL_SCHEMES = {"http", "https"}
GO_LICENSE_LOCKFILES = (
    Path("services/core/jobs/jobs-launcher/go.mod"),
    Path("services/guardrails/callouts/go.mod"),
)

PROJECT_URL_REPOSITORY_KEYS = (
    "Source",
    "Source Code",
    "Repository",
    "Code",
    "Homepage",
    "Home",
    "Project",
)

PROJECT_URL_LICENSE_KEYS = (
    "License",
    "License File",
    "License URL",
)

SPDX_LICENSE_URLS = {
    "APACHE-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "BSD-2-CLAUSE": "https://opensource.org/license/bsd-2-clause",
    "BSD-3-CLAUSE": "https://opensource.org/license/bsd-3-clause",
    "ISC": "https://opensource.org/licenses/ISC",
    "MIT": "https://opensource.org/licenses/MIT",
    "PSF-2.0": "https://docs.python.org/3/license.html",
    "ZLIB": "https://opensource.org/license/zlib",
}


class LicenseGenerationError(Exception):
    """Raised when license generation fails."""

    pass


def _sanitize_csv_value(value: str) -> str:
    """Escape values that spreadsheet tools may interpret as formulas."""
    if value.lstrip().startswith(FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _safe_url_for_csv(value: Any) -> str:
    """Return a CSV-safe URL if it has an allowed scheme."""
    if not isinstance(value, str):
        return ""

    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in SAFE_URL_SCHEMES or not parsed.netloc:
        return ""

    return _sanitize_csv_value(url)


def _get_pypi_json(package_name: str, version: str) -> dict[str, Any]:
    """Return PyPI JSON metadata for a package, falling back to the unversioned endpoint."""
    cache_key = (package_name, version)
    if cache_key in _PYPI_JSON_CACHE:
        return _PYPI_JSON_CACHE[cache_key]

    normalized_version = version.split("+", 1)[0]
    urls = []
    if normalized_version:
        urls.append(f"https://pypi.org/pypi/{package_name}/{normalized_version}/json")
    urls.append(f"https://pypi.org/pypi/{package_name}/json")

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException as exc:
            logger.debug("Could not fetch PyPI metadata for %s from %s: %s", package_name, url, exc)
            continue

        if response.ok:
            try:
                data = response.json()
                if not isinstance(data, dict):
                    logger.debug("PyPI metadata for %s from %s was not a JSON object", package_name, url)
                    continue
                _PYPI_JSON_CACHE[cache_key] = data
            except ValueError as exc:
                logger.debug("Could not parse PyPI metadata for %s from %s: %s", package_name, url, exc)
                continue
            return data

        logger.debug("Could not fetch PyPI metadata for %s from %s: HTTP %s", package_name, url, response.status_code)

    _PYPI_JSON_CACHE[cache_key] = {}
    return {}


def _github_repository_url(url: str) -> str:
    """Normalize common GitHub project URLs to an owner/repository URL."""
    if not isinstance(url, str):
        return ""

    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return ""

    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(path_parts) < 2:
        return ""

    owner, repo = path_parts[:2]
    repo = repo.removesuffix(".git")
    return f"https://github.com/{owner}/{repo}"


def _license_url_from_pypi_info(info: dict[str, Any], license_str: str) -> str:
    """Resolve a best-effort license URL from PyPI metadata."""
    project_urls = info.get("project_urls") or {}
    if not isinstance(project_urls, dict):
        project_urls = {}

    for key in PROJECT_URL_LICENSE_KEYS:
        if url := _safe_url_for_csv(project_urls.get(key)):
            return url

    for key in PROJECT_URL_REPOSITORY_KEYS:
        if repo_url := _github_repository_url(project_urls.get(key, "")):
            return _safe_url_for_csv(f"{repo_url}/blob/main/LICENSE")

    if repo_url := _github_repository_url(info.get("home_page", "")):
        return _safe_url_for_csv(f"{repo_url}/blob/main/LICENSE")

    return SPDX_LICENSE_URLS.get(license_str.upper(), "")


def resolve_license_url(package_name: str, version: str, license_str: str) -> str:
    """Return a best-effort URL for the package's license text."""
    pypi_data = _get_pypi_json(package_name, version)
    info = pypi_data.get("info", {})
    if not info:
        return SPDX_LICENSE_URLS.get(license_str.upper(), "")

    return _license_url_from_pypi_info(info, license_str)


def get_osv_scanner() -> Path:
    """
    Check if osv-scanner is available.

    Returns:
        Path to osv-scanner executable

    Raises:
        LicenseGenerationError: If osv-scanner is not found
    """
    osv_path = shutil.which("osv-scanner")
    if not osv_path:
        raise LicenseGenerationError("osv-scanner not found. Please install it first.")
    return Path(osv_path)


def _sanitize_osv_json(output_file: Path) -> None:
    """
    Remove machine-specific paths from OSV JSON output.

    The osv-scanner includes absolute paths like /Users/username/code/nhx/uv.lock
    which cause unnecessary diffs when different developers run the tool.
    """

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove path from source objects in results
    for result in data.get("results", []):
        source = result.get("source", {})
        if "path" in source:
            del source["path"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _merge_osv_json_files(output_file: Path, input_files: list[Path]) -> None:
    """Merge OSV JSON scan result arrays into a single JSON artifact."""
    merged: dict[str, Any] = {"results": []}
    license_summary: Counter[str] = Counter()

    for input_file in input_files:
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)

        merged["results"].extend(data.get("results", []))
        for entry in data.get("license_summary", []):
            name = entry.get("name", "")
            count = entry.get("count", 0)
            if name and isinstance(count, int):
                license_summary[name] += count

        if "experimental_config" in data and "experimental_config" not in merged:
            merged["experimental_config"] = data["experimental_config"]

    if license_summary:
        merged["license_summary"] = [
            {"name": name, "count": count}
            for name, count in sorted(license_summary.items(), key=lambda item: (-item[1], item[0]))
        ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def _iter_osv_package_entries(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield package entries across all OSV scanner results."""
    for result in data.get("results", []):
        yield from result.get("packages", [])


def _sanitize_requirements_file(requirements_file: Path, export_command: list[str]) -> None:
    """
    Remove machine-specific absolute paths from requirements file comments.

    The uv export command includes a comment with the full command including
    absolute paths like /Users/username/dev/nhx/third_party/requirements.txt
    which cause unnecessary diffs when different developers run the tool.
    """
    sanitized_command = list(export_command)
    output_file_arg = f"third_party/{requirements_file.name}"
    for index, arg in enumerate(sanitized_command):
        if arg == "--output-file" and index + 1 < len(sanitized_command):
            sanitized_command[index + 1] = output_file_arg
        elif arg.startswith("--output-file="):
            sanitized_command[index] = f"--output-file={output_file_arg}"

    with open(requirements_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Replace absolute paths with relative paths in the header comment
    if lines and lines[0].startswith("# This file was autogenerated by uv"):
        # Just keep the generic comment without the specific command
        lines[0] = "# This file was autogenerated by uv via the following command:\n"
        if len(lines) > 1 and lines[1].startswith("#    uv export"):
            lines[1] = f"#    {shlex.join(sanitized_command)}\n"

    with open(requirements_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def run_osv_scanner(lockfile: Path, output_file: Path, cwd: Optional[Path] = None) -> None:
    """
    Run osv-scanner on a lockfile and save JSON output.

    Args:
        lockfile: Path to uv.lock or other lockfile
        output_file: Where to save the JSON output
        cwd: Working directory to run the command in

    Raises:
        LicenseGenerationError: If osv-scanner fails
    """
    get_osv_scanner()

    cmd = [
        "osv-scanner",
        "scan",
        "source",
        "--licenses",
        "--lockfile",
        str(lockfile),
        "--format",
        "json",
        "--all-packages",
        "--output",
        str(output_file),
    ]

    logger.debug(f"Running: {' '.join(cmd)}")
    logger.debug(f"Working directory: {cwd or Path.cwd()}")

    try:
        # Run osv-scanner, suppress stderr output (vulnerability warnings)
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            check=False,  # osv-scanner returns non-zero when vulnerabilities found
        )

        # osv-scanner returns non-zero exit codes for various reasons
        # We only care if the output file was created
        if not output_file.exists():
            raise LicenseGenerationError(
                f"osv-scanner failed to create output file: {output_file}\n"
                f"Exit code: {result.returncode}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )

        # Remove machine-specific paths from output
        _sanitize_osv_json(output_file)

        logger.info(f"✓ Generated OSV JSON: {output_file}")

    except FileNotFoundError:
        raise LicenseGenerationError("osv-scanner command not found")


def _iter_exported_requirements(requirements_file: Path) -> Iterator[Requirement]:
    """Yield parsed top-level entries from a uv-exported requirements file."""
    if not requirements_file.exists():
        return

    with open(requirements_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or line.startswith((" ", "\t")):
                continue
            if stripped == "-e" or stripped.startswith("-e "):
                continue

            requirement_text = stripped.removesuffix("\\").strip()
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement:
                logger.warning("Could not parse exported requirement: %s", requirement_text)
                continue

            yield requirement


def _get_exported_requirement_names(requirements_file: Path) -> set[str]:
    """Return normalized package names present in an exported requirements file."""
    return {normalize_package_name(requirement.name) for requirement in _iter_exported_requirements(requirements_file)}


def _get_requirements_with_overrides(
    requirements_file: Path, overrides: dict[str, str], local_packages: set[str]
) -> list[dict[str, str]]:
    """Return exported requirements that can be licensed from reviewed overrides."""
    packages = []
    for requirement in _iter_exported_requirements(requirements_file):
        name = requirement.name
        if normalize_package_name(name) in local_packages:
            continue

        version = ""
        for specifier in requirement.specifier:
            if specifier.operator == "==":
                version = specifier.version
                break

        override_key = get_override_key_for_package(name, version)
        if override_key not in overrides:
            continue

        packages.append(
            {
                "name": name,
                "version": version,
                "ecosystem": "PyPI",
                "license": overrides[override_key].upper(),
            }
        )

    return packages


def format_licenses(
    osv_json: Path, output_file: Path, overrides_file: Optional[Path] = None, format_type: str = "table"
) -> None:
    """
    Format OSV JSON output into the desired format.

    Args:
        osv_json: Path to OSV scanner JSON output
        output_file: Where to save the formatted output
        overrides_file: Optional path to license overrides YAML
        format_type: Output format (table, jsonl, json, csv, markdown, text)

    Raises:
        LicenseGenerationError: If formatting fails
    """

    try:
        logger.debug(f"Formatting {osv_json} -> {output_file} (format: {format_type})")

        # Load the JSON
        with open(osv_json) as f:
            import json

            data = json.load(f)

        # Get overrides
        overrides = {}
        if overrides_file and overrides_file.exists():
            import yaml

            with open(overrides_file) as f:
                override_data = yaml.safe_load(f)
                overrides = override_data.get("overrides", {})

        # Get local packages to exclude
        workspace_root = osv_json.parent.parent  # Assuming osv_json is in third_party/
        local_packages = get_local_packages(workspace_root)

        # Use old formatter for backward compatibility
        if format_type == "table":
            formatted = format_licenses_table(data, overrides, local_packages)
        else:
            # Use new formatter
            formatter = get_formatter(format_type)

            requirements_file = osv_json.parent / "requirements-main.txt"
            exported_requirement_names = _get_exported_requirement_names(requirements_file)

            # Extract package info from OSV data
            packages = []
            for pkg_data in _iter_osv_package_entries(data):
                pkg = pkg_data.get("package", {})
                name = pkg.get("name", "")
                version = pkg.get("version", "")
                ecosystem = pkg.get("ecosystem", "")
                licenses = pkg_data.get("licenses", [])
                normalized_name = normalize_package_name(name)

                # Skip local packages
                if normalized_name in local_packages:
                    continue

                # OSV can include conditional transitive Python dependencies that
                # uv did not export for this environment. Other ecosystems are
                # scanned directly from their own lockfiles and should not be
                # filtered through requirements-main.txt.
                if (
                    ecosystem.lower() in {"", "pypi"}
                    and exported_requirement_names
                    and normalized_name not in exported_requirement_names
                ):
                    continue

                # Check overrides (use base name so +cu129 variants match)
                override_key = get_override_key_for_package(name, version)
                if override_key in overrides:
                    license_str = overrides[override_key]
                elif not licenses:
                    continue
                else:
                    # Resolve to a single license, preferring allowed ones
                    license_str = resolve_license(licenses, ALLOWED_LICENSES)

                packages.append(
                    {
                        "name": name,
                        "version": version,
                        "ecosystem": ecosystem,
                        "license": license_str.upper(),
                    }
                )

            # OSV can emit a partial package list when the service is degraded.
            # Keep output stable for reviewed licenses by filling only packages
            # that are present in the exported requirements and overrides.yaml.
            packages.extend(_get_requirements_with_overrides(requirements_file, overrides, local_packages))

            # Deduplicate by ecosystem/name (keep first occurrence)
            seen_names = set()
            unique_packages = []
            for pkg in packages:
                package_key = (str(pkg.get("ecosystem", "")).lower(), normalize_package_name(pkg["name"]))
                if package_key not in seen_names:
                    seen_names.add(package_key)
                    unique_packages.append(pkg)

            # Sort by name
            unique_packages.sort(key=lambda x: x["name"].lower())

            if format_type == "csv":
                for pkg in unique_packages:
                    pkg["license_url"] = _sanitize_csv_value(
                        resolve_license_url(pkg["name"], pkg.get("version", ""), pkg["license"])
                    )

            # Format using selected formatter
            formatted = formatter.format(unique_packages)

        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(formatted)

        logger.debug(f"✓ Generated license {format_type}: {output_file}")

    except Exception as e:
        raise LicenseGenerationError(f"Failed to format licenses: {e}")


def generate_lockfile_without_dev_dependencies(
    lockfile_dir: Path,
    output_lockfile: Path,
    extras: Optional[list[str]] = None,
    packages: Optional[list[str]] = None,
    groups: Optional[list[str]] = None,
) -> Path:
    """
    Generate a lockfile without dev dependencies.

    Args:
        lockfile_dir: Path to the directory containing the lockfile
        output_lockfile: Path to the output lockfile
        extras: Optional list of extras to include (e.g., ["cu129"])
        packages: Optional list of workspace packages to include via --package flag.
            Required when extras reference workspace member dependencies, since
            uv export does not propagate workspace packages' extras' transitive
            dependencies without explicitly specifying --package.
        groups: Optional list of dependency groups to include via --group flag.

    Returns:
        Path to the output lockfile
    """

    cmd = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
    ]

    # Add packages if specified (needed to resolve workspace member extras)
    if packages:
        for package in packages:
            cmd.extend(["--package", package])

    if extras:
        for extra in extras:
            cmd.extend(["--extra", extra])

    # Add non-dev dependency groups used by shipped container images.
    if groups:
        for group in groups:
            cmd.extend(["--group", group])

    cmd.extend(
        [
            "--quiet",
            "--output-file",
            str(output_lockfile),
        ]
    )

    logger.debug(f"Running: {' '.join(cmd)}")
    logger.debug(f"Working directory: {lockfile_dir}")

    try:
        subprocess.run(
            cmd,
            cwd=lockfile_dir,
            text=True,
            check=True,
        )
        # Sanitize the generated requirements file to remove machine-specific paths
        _sanitize_requirements_file(output_lockfile, cmd)
    except Exception as e:
        raise LicenseGenerationError(f"Failed to generate lockfile without dev dependencies: {e}")
    return output_lockfile


def generate_project_licenses(
    project_name: str,
    lockfile_dir: Path,
    output_lockfile: Path,
    osv_json: Path,
    output_file: Path,
    overrides_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    format_type: str = "table",
    extras: Optional[list[str]] = None,
    packages: Optional[list[str]] = None,
    additional_lockfiles: Optional[list[Path]] = None,
    groups: Optional[list[str]] = None,
) -> str:
    """
    Generate licenses for a single project.

    Args:
        project_name: Name of the project (for logging)
        lockfile_dir: Path to the directory containing the lockfile
        output_lockfile: Path to the output lockfile
        osv_json: Where to save OSV JSON output
        output_file: Where to save the formatted license table
        overrides_file: Optional path to license overrides
        cwd: Working directory for osv-scanner
        format_type: Output format (table, jsonl, json, csv, markdown, text)
        extras: Optional list of extras to include (e.g., ["cu129"])
        packages: Optional list of workspace packages to include via --package flag
        additional_lockfiles: Optional lockfiles to scan and merge into osv_json
        groups: Optional list of dependency groups to include via --group flag

    Returns:
        Success message

    Raises:
        LicenseGenerationError: If generation fails
    """
    try:
        logger.info(f"Generating licenses for {project_name}...")

        generate_lockfile_without_dev_dependencies(
            lockfile_dir,
            output_lockfile,
            extras=extras,
            packages=packages,
            groups=groups,
        )

        # Run osv-scanner for the Python export.
        run_osv_scanner(output_lockfile, osv_json, cwd=cwd)

        osv_json_files = [osv_json]
        for lockfile in additional_lockfiles or []:
            scan_output = osv_json.with_name(f"{osv_json.stem}-{lockfile.parent.name}{osv_json.suffix}")
            run_osv_scanner(lockfile, scan_output, cwd=cwd)
            osv_json_files.append(scan_output)

        if len(osv_json_files) > 1:
            _merge_osv_json_files(osv_json, osv_json_files)

        # Format output
        format_licenses(osv_json, output_file, overrides_file, format_type=format_type)

        return f"✓ Generated {output_file}"

    except Exception as e:
        logger.error(f"Failed to generate licenses for {project_name}: {e}")
        raise


def get_projects(workspace_root: Path, output_file: Optional[Path] = None) -> list[dict[str, Any]]:
    import os

    # Check for environment variable overrides (used in CI)
    license_dir_str = os.environ.get("LICENSE_DIR", str(workspace_root / "third_party"))
    license_dir = Path(license_dir_str)

    main_license_name = os.environ.get("LICENSE_NAME", "licenses.jsonl")
    main_output_file = output_file or license_dir / main_license_name

    # Define projects to scan
    projects = [
        {
            "name": "main",
            "lockfile_dir": workspace_root,
            "osv_json": license_dir / "osv-licenses.json",
            "output_lockfile": license_dir / "requirements-main.txt",
            "output_file": main_output_file,
            "cwd": workspace_root,
            "overrides_file": OVERRIDES_FILE,
            # Include shipped task/training image dependency surfaces even when
            # they are intentionally excluded from nhx-api / core service groups.
            "packages": [
                "nemohelix",
                "nhx-automodel",
                "nhx-rl",
                "nhx-unsloth",
                "nemo-safe-synthesizer-plugin",
            ],
            "extras": ["integrations", "unsloth"],
            "groups": [
                "functional-services",
                "cpu-tasks",
                "gpu-tasks",
            ],
            "additional_lockfiles": [workspace_root / lockfile for lockfile in GO_LICENSE_LOCKFILES],
        },
    ]
    return projects


def generate_all_licenses(
    workspace_root: Path, parallel: bool = True, format_type: str = "table", output_file: Optional[Path] = None
) -> None:
    """
    Generate license reports for the main project.

    Args:
        workspace_root: Path to the workspace root
        parallel: Whether to run scans in parallel (default: True)
        format_type: Output format (table, jsonl, json, csv, markdown, text)
        output_file: Optional output path for the formatted license report

    Raises:
        LicenseGenerationError: If generation fails
    """
    projects = get_projects(workspace_root, output_file=output_file)
    if parallel:
        logger.info("Generating licenses for main project...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    generate_project_licenses,
                    proj["name"],
                    proj["lockfile_dir"],
                    proj["output_lockfile"],
                    proj["osv_json"],
                    proj["output_file"],
                    proj["overrides_file"],
                    proj.get("cwd"),
                    format_type=format_type,
                    extras=proj.get("extras"),
                    packages=proj.get("packages"),
                    additional_lockfiles=proj.get("additional_lockfiles"),
                    groups=proj.get("groups"),
                ): proj["name"]
                for proj in projects
            }

            for future in as_completed(futures):
                project_name = futures[future]
                try:
                    message = future.result()
                    logger.info(message)
                except Exception as e:
                    logger.error(f"Error generating licenses for {project_name}: {e}")
                    raise
    else:
        # Sequential execution
        for proj in projects:
            message = generate_project_licenses(
                proj["name"],
                proj["lockfile_dir"],
                proj["output_lockfile"],
                proj["osv_json"],
                proj["output_file"],
                proj["overrides_file"],
                proj.get("cwd"),
                format_type=format_type,
                extras=proj.get("extras"),
                packages=proj.get("packages"),
                additional_lockfiles=proj.get("additional_lockfiles"),
                groups=proj.get("groups"),
            )
            logger.info(message)

    logger.info("\nLicense generation complete!")
