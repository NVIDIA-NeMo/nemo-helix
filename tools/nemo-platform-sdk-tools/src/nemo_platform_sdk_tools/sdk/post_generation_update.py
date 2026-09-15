# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-generation update tool for NeMo Platform SDK.

This script applies customizations to the auto-generated SDK after Stainless generation.
It handles README merging, pyproject.toml updates, and LICENSE file copying.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Tuple

import tomlkit
import typer
from nemo_platform_sdk_tools.sdk.core.common import WRAPPER_DISTRIBUTION_NAME, SdkInfo, get_sdk_info
from nemo_platform_sdk_tools.sdk.post_generation_exist_ok import inject_exist_ok
from nemo_platform_sdk_tools.sdk.source_owned_resources import SOURCE_OWNED_RESOURCE_EXCLUSIONS
from tomlkit.items import AoT, Table

app = typer.Typer(
    name="post-generation", help="Post-generation update tool for NeMo Platform SDK.", no_args_is_help=True
)

SDK_BUILD_SOURCE_PACKAGES: tuple[dict[str, str | list[str]], ...] = (
    {
        "source": "packages/nemo_platform_ext/src/nemo_platform_ext",
        "target": "nemo_platform_ext",
        "include": [
            "**/*.py",
            "skills/**/*.md",
            "skills/**/*.yaml",
            "skills/**/*.yml",
            "skills/**/*.json",
        ],
    },
    {"source": "packages/models/src/models", "target": "models"},
    {"source": "packages/filesets/src/filesets", "target": "filesets"},
    {
        "source": "packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk",
        "target": "nemo_evaluator_sdk",
        "include": [
            "**/*.py",
            "agent_eval/runtimes/fabric/sandbox.Dockerfile",
        ],
    },
)
SOURCE_OWNED_RESOURCE_NAMES = frozenset(resource.resource_name for resource in SOURCE_OWNED_RESOURCE_EXCLUSIONS)
SOURCE_OWNED_JOB_STATUS_TYPE_EXPORTS: tuple[tuple[str, str], ...] = (
    ("platform_job_status_response", "PlatformJobStatusResponse"),
    ("platform_job_step_status_response", "PlatformJobStepStatusResponse"),
    ("platform_job_task_status_response", "PlatformJobTaskStatusResponse"),
)
SOURCE_OWNED_SECRET_TYPE_EXPORTS: tuple[tuple[str, str], ...] = (
    ("platform_secret_access_response", "PlatformSecretAccessResponse"),
    ("platform_secret_admin_rotation_response", "PlatformSecretAdminRotationResponse"),
    ("platform_secret_create_request", "PlatformSecretCreateRequest"),
    ("platform_secret_response", "PlatformSecretResponse"),
    ("platform_secret_responses_page", "PlatformSecretResponsesPage"),
    ("platform_secret_update_request", "PlatformSecretUpdateRequest"),
    ("secret_create_params", "SecretCreateParams"),
    ("secret_list_params", "SecretListParams"),
    ("secret_update_params", "SecretUpdateParams"),
)
SOURCE_OWNED_JOBS_TYPES_INIT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.jobs.spec import (
    PlatformJobSpec as PlatformJobSpec,
    PlatformJobStepSpec as PlatformJobStepSpec,
)
from nemo_platform_plugin.jobs.types import (
    JobLogsQueryParams as JobLogsQueryParams,
    ListJobsQueryParams as ListJobsQueryParams,
    PlatformJobResponse as PlatformJobResponse,
    ListStepsQueryParams as ListStepsQueryParams,
    PlatformJobSortField as PlatformJobSortField,
    PlatformJobTaskUpdate as PlatformJobTaskUpdate,
    JobStatusDetailsUpdate as JobStatusDetailsUpdate,
    PlatformJobLogSortField as PlatformJobLogSortField,
    PlatformJobStepResponse as PlatformJobStepResponse,
    PlatformJobTaskResponse as PlatformJobTaskResponse,
    CreatePlatformJobRequest as CreatePlatformJobRequest,
    PlatformJobListSortField as PlatformJobListSortField,
    ListJobResultsQueryParams as ListJobResultsQueryParams,
    PlatformJobStepWithContext as PlatformJobStepWithContext,
    PlatformJobAttemptSortField as PlatformJobAttemptSortField,
    PlatformJobListTaskResponse as PlatformJobListTaskResponse,
    PlatformJobStatusUpdateRequest as PlatformJobStatusUpdateRequest,
    PlatformJobStatusDetailsUpdateRequest as PlatformJobStatusDetailsUpdateRequest,
)
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobLog as PlatformJobLog,
    FileStorageType as FileStorageType,
    PlatformJobStatus as PlatformJobStatus,
    PlatformJobLogPage as PlatformJobLogPage,
    PlatformJobResultResponse as PlatformJobResultResponse,
    PlatformJobStatusResponse as PlatformJobStatusResponse,
    PlatformJobListResultResponse as PlatformJobListResultResponse,
    PlatformJobStepStatusResponse as PlatformJobStepStatusResponse,
    PlatformJobTaskStatusResponse as PlatformJobTaskStatusResponse,
    PlatformJobResultCreateRequest as PlatformJobResultCreateRequest,
)

PlatformJobStep = PlatformJobStepResponse
"""
SOURCE_OWNED_SECRETS_TYPES_INIT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.secrets.types import (
    SecretListParams as SecretListParams,
    SecretCreateParams as SecretCreateParams,
    SecretUpdateParams as SecretUpdateParams,
    ListSecretsQueryParams as ListSecretsQueryParams,
    PlatformSecretResponse as PlatformSecretResponse,
    PlatformSecretCreateRequest as PlatformSecretCreateRequest,
    PlatformSecretResponsesPage as PlatformSecretResponsesPage,
    PlatformSecretUpdateRequest as PlatformSecretUpdateRequest,
    PlatformSecretAccessResponse as PlatformSecretAccessResponse,
    PlatformSecretAdminRotationResponse as PlatformSecretAdminRotationResponse,
)
"""
SOURCE_OWNED_SECRETS_RESOURCE_INIT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.secrets.compat import (
    SecretsResource as SecretsResource,
    AsyncSecretsResource as AsyncSecretsResource,
    SecretsAdminResource as SecretsAdminResource,
    AsyncSecretsAdminResource as AsyncSecretsAdminResource,
)

AdminResource = SecretsAdminResource
AsyncAdminResource = AsyncSecretsAdminResource
"""
SOURCE_OWNED_SECRETS_ADMIN_RESOURCE = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.secrets.compat import (
    SecretsAdminResource as SecretsAdminResource,
    AsyncSecretsAdminResource as AsyncSecretsAdminResource,
)

AdminResource = SecretsAdminResource
AsyncAdminResource = AsyncSecretsAdminResource
"""
SOURCE_OWNED_SECRETS_RESOURCE = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.secrets.compat import (
    SecretsResource as SecretsResource,
    AsyncSecretsResource as AsyncSecretsResource,
)
"""


def _build_hook_source(sdk_info: SdkInfo) -> Path:
    return sdk_info.overrides_dir / "hatch_build.py"


def _require_build_hook_source(sdk_info: SdkInfo) -> Path:
    source = _build_hook_source(sdk_info)
    if not source.exists():
        raise FileNotFoundError(f"SDK build hook override is required but was not found: {source}")
    return source


def _copy_build_hook(sdk_info: SdkInfo) -> Path:
    source = _require_build_hook_source(sdk_info)
    destination = sdk_info.sdk_dir / "hatch_build.py"
    shutil.copy2(source, destination)
    return destination


def sdk_version_file_content(distribution_name: str) -> str:
    """Render ``_version.py``, which reads the version from installed metadata.

    ``distribution_name`` is the distribution to query at runtime, which is not
    the distribution being built: the SDK ships bundled inside the
    ``nemo-platform`` wheel, so ``nemo-platform-sdk`` metadata only exists in
    the workspace. Both report the same version there.
    """
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version as _package_version

__title__ = "nemo_platform"
try:
    __version__ = _package_version("{distribution_name}")
except PackageNotFoundError:
    __version__ = "0.0.0"
# Injected at release time for non-production builds; None for RC and production releases.
__image_tag__: str | None = None
'''


def merge_readme_files(readme_dir: Path) -> str:
    """Merge README markdown files in numerical order."""
    readme_files = sorted([f for f in readme_dir.glob("*.md") if f.name[0].isdigit()])

    if not readme_files:
        typer.echo(f"Error: No numbered README files found in {readme_dir}", err=True)
        raise typer.Exit(code=1)

    merged_content = []

    for readme_file in readme_files:
        typer.echo(f"  - Merging {readme_file.name}")
        content = readme_file.read_text(encoding="utf-8").strip()
        merged_content.append(content)

    return "\n\n".join(merged_content)


def _toml_array_of_tables(entries: tuple[dict[str, str | list[str]], ...]) -> AoT:
    array = tomlkit.aot()
    for entry in entries:
        table = tomlkit.table()
        for key, value in entry.items():
            table[key] = value
        array.append(table)
    return array


def _source_package_hook_table() -> Table:
    hook = tomlkit.table()
    hook["path"] = "hatch_build.py"
    hook["source-packages"] = _toml_array_of_tables(SDK_BUILD_SOURCE_PACKAGES)
    return hook


def get_string_replacements() -> List[Tuple[str, str]]:
    """Get list of string replacements to apply to source code."""
    stainless_sdk_url = "https://www.github.com/stainless-sdks/nemo-platform-python"
    nemo_docs_url = "https://docs.nvidia.com/nemo/microservices/latest/pysdk/index.html"

    return [
        (
            stainless_sdk_url + "#accessing-raw-response-data-eg-headers",
            nemo_docs_url + "#accessing-raw-response-data-e-g-headers",
        ),
        (stainless_sdk_url + "#with_streaming_response", nemo_docs_url + "#with_streaming_response"),
        # Add more replacements here as needed
        # (old_string, new_string),
    ]


def check_for_stainless_references(sdk_dir: Path) -> List[Tuple[Path, int, str]]:
    """Check for any remaining references to Stainless URLs in the SDK code."""
    # Patterns to look for
    stainless_patterns = ["stainless.com", "github.com/stainless-sdks", "stainless-sdks"]

    # File patterns to process
    patterns = ["**/*.py"]

    # Files to exclude from processing
    exclude_patterns = [
        "**/.*",  # Hidden files
        "**/__pycache__/**",  # Python cache
        "**/node_modules/**",  # Node modules
        "**/venv/**",  # Virtual environments
        "**/.venv/**",  # Virtual environments
    ]

    findings = []

    for pattern in patterns:
        for file_path in sdk_dir.glob(pattern):
            # Skip if file matches exclude patterns
            if any(file_path.match(exclude_pattern) for exclude_pattern in exclude_patterns):
                continue

            # Skip if not a file
            if not file_path.is_file():
                continue

            try:
                # Read file content
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                # Check each line for stainless references
                for line_num, line in enumerate(lines, 1):
                    for stainless_pattern in stainless_patterns:
                        if stainless_pattern.lower() in line.lower():
                            findings.append((file_path, line_num, line.strip()))
                            break  # Only report once per line

            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue

    return findings


def apply_string_replacements(sdk_dir: Path, replacements: List[Tuple[str, str]]) -> None:
    """Apply string replacements to all Python."""
    # File patterns to process
    patterns = ["**/*.py"]

    # Files to exclude from processing
    exclude_patterns = [
        "**/.*",  # Hidden files
        "**/__pycache__/**",  # Python cache
        "**/node_modules/**",  # Node modules
        "**/venv/**",  # Virtual environments
        "**/.venv/**",  # Virtual environments
    ]

    processed_files = 0
    total_replacements = 0

    for pattern in patterns:
        for file_path in sdk_dir.glob(pattern):
            # Skip if file matches exclude patterns
            if any(file_path.match(exclude_pattern) for exclude_pattern in exclude_patterns):
                continue

            # Skip if not a file
            if not file_path.is_file():
                continue

            try:
                # Read file content
                content = file_path.read_text(encoding="utf-8")
                original_content = content

                # Apply replacements
                file_replacements = 0
                for old_string, new_string in replacements:
                    if old_string in content:
                        content = content.replace(old_string, new_string)
                        file_replacements += content.count(new_string) - original_content.count(new_string)

                # Write back if changes were made
                if content != original_content:
                    file_path.write_text(content, encoding="utf-8")
                    processed_files += 1
                    total_replacements += file_replacements
                    typer.echo(f"  - Updated {file_path.relative_to(sdk_dir)} ({file_replacements} replacements)")

            except (UnicodeDecodeError, PermissionError) as e:
                # Skip files that can't be read or written
                typer.echo(f"  - Skipped {file_path.relative_to(sdk_dir)} ({e})", err=True)
                continue

    typer.echo(f"  - Processed {processed_files} files with {total_replacements} total replacements")


def _sdk_type_exports(sdk_info: SdkInfo) -> set[str]:
    types_init = sdk_info.sdk_dir / "src" / sdk_info.module_name / "types" / "__init__.py"
    if not types_init.exists():
        return set()

    exports: set[str] = set()
    for line in types_init.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s+as\s+\1,\s*", line)
        if match:
            exports.add(match.group(1))
    return exports


def _api_index_resource_name(sdk_info: SdkInfo, link_target: str) -> str | None:
    match = re.fullmatch(rf"src/{sdk_info.module_name}/resources/([^/]+)/api\.md", link_target)
    if match:
        return match.group(1)
    return None


def clean_api_index(sdk_info: SdkInfo) -> bool:
    """Remove generated API index entries for artifacts absent after post-processing."""
    api_index = sdk_info.sdk_dir / "api.md"
    if not api_index.exists():
        typer.echo(f"api.md not found at {api_index}. Skipping cleanup.")
        return False

    type_exports = _sdk_type_exports(sdk_info)
    content = api_index.read_text(encoding="utf-8")
    lines = content.splitlines()
    cleaned_lines: list[str] = []
    in_type_import_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"from {sdk_info.module_name}.types import ("):
            in_type_import_block = True
            cleaned_lines.append(line)
            continue
        if in_type_import_block and stripped == ")":
            in_type_import_block = False
            cleaned_lines.append(line)
            continue
        if in_type_import_block:
            type_name = stripped.removesuffix(",")
            if type_name and type_name not in type_exports:
                continue

        link_match = re.fullmatch(r"# \[[^\]]+\]\(([^)]+)\)", stripped)
        resource_name = _api_index_resource_name(sdk_info, link_match.group(1)) if link_match else None
        if (
            resource_name in SOURCE_OWNED_RESOURCE_NAMES
            and link_match
            and not (sdk_info.sdk_dir / link_match.group(1)).exists()
        ):
            continue

        cleaned_lines.append(line)

    cleaned_content = "\n".join(cleaned_lines)
    if content.endswith("\n"):
        cleaned_content += "\n"

    if cleaned_content == content:
        typer.echo(f"  - No changes needed for {api_index}")
        return False

    api_index.write_text(cleaned_content, encoding="utf-8")
    typer.echo(f"  - Updated {api_index}")
    return True


def _source_owned_type_module_content(source_module: str, type_name: str) -> str:
    return f"""\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from {source_module} import (
    {type_name} as {type_name},
)
"""


def _append_missing_lines(path: Path, lines: tuple[str, ...]) -> bool:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = set(content.splitlines())
    missing_lines = [line for line in lines if line not in existing_lines]
    if not missing_lines:
        return False

    if content and not content.endswith("\n"):
        content += "\n"
    if content and not content.endswith("\n\n"):
        content += "\n"
    content += "\n".join(missing_lines)
    content += "\n"
    path.write_text(content, encoding="utf-8")
    return True


def ensure_source_owned_type_aliases(sdk_info: SdkInfo) -> bool:
    """Expose source-owned compatibility objects through legacy SDK import paths."""
    types_dir = sdk_info.sdk_dir / "src" / sdk_info.module_name / "types"
    resources_dir = sdk_info.sdk_dir / "src" / sdk_info.module_name / "resources"
    shared_dir = types_dir / "shared"
    jobs_dir = types_dir / "jobs"
    secrets_types_dir = types_dir / "secrets"
    secrets_resources_dir = resources_dir / "secrets"
    changed = False

    shared_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    secrets_types_dir.mkdir(parents=True, exist_ok=True)
    secrets_resources_dir.mkdir(parents=True, exist_ok=True)

    for module_name, type_name in SOURCE_OWNED_JOB_STATUS_TYPE_EXPORTS:
        module_path = shared_dir / f"{module_name}.py"
        module_content = _source_owned_type_module_content("nemo_platform_plugin.jobs.schemas", type_name)
        if not module_path.exists() or module_path.read_text(encoding="utf-8") != module_content:
            module_path.write_text(module_content, encoding="utf-8")
            changed = True

    for module_name, type_name in SOURCE_OWNED_SECRET_TYPE_EXPORTS:
        module_path = secrets_types_dir / f"{module_name}.py"
        module_content = _source_owned_type_module_content("nemo_platform_plugin.secrets.types", type_name)
        if not module_path.exists() or module_path.read_text(encoding="utf-8") != module_content:
            module_path.write_text(module_content, encoding="utf-8")
            changed = True

    jobs_init = jobs_dir / "__init__.py"
    if not jobs_init.exists() or jobs_init.read_text(encoding="utf-8") != SOURCE_OWNED_JOBS_TYPES_INIT:
        jobs_init.write_text(SOURCE_OWNED_JOBS_TYPES_INIT, encoding="utf-8")
        changed = True

    secrets_types_init = secrets_types_dir / "__init__.py"
    if (
        not secrets_types_init.exists()
        or secrets_types_init.read_text(encoding="utf-8") != SOURCE_OWNED_SECRETS_TYPES_INIT
    ):
        secrets_types_init.write_text(SOURCE_OWNED_SECRETS_TYPES_INIT, encoding="utf-8")
        changed = True

    secrets_resources_init = secrets_resources_dir / "__init__.py"
    if (
        not secrets_resources_init.exists()
        or secrets_resources_init.read_text(encoding="utf-8") != SOURCE_OWNED_SECRETS_RESOURCE_INIT
    ):
        secrets_resources_init.write_text(SOURCE_OWNED_SECRETS_RESOURCE_INIT, encoding="utf-8")
        changed = True

    secrets_admin = secrets_resources_dir / "admin.py"
    if not secrets_admin.exists() or secrets_admin.read_text(encoding="utf-8") != SOURCE_OWNED_SECRETS_ADMIN_RESOURCE:
        secrets_admin.write_text(SOURCE_OWNED_SECRETS_ADMIN_RESOURCE, encoding="utf-8")
        changed = True

    secrets_resource = secrets_resources_dir / "secrets.py"
    if not secrets_resource.exists() or secrets_resource.read_text(encoding="utf-8") != SOURCE_OWNED_SECRETS_RESOURCE:
        secrets_resource.write_text(SOURCE_OWNED_SECRETS_RESOURCE, encoding="utf-8")
        changed = True

    shared_init = shared_dir / "__init__.py"
    shared_imports = tuple(
        f"from .{module_name} import {type_name} as {type_name}"
        for module_name, type_name in SOURCE_OWNED_JOB_STATUS_TYPE_EXPORTS
    )
    changed = _append_missing_lines(shared_init, shared_imports) or changed

    types_init = types_dir / "__init__.py"
    top_level_imports = tuple(
        f"from .shared import {type_name} as {type_name}" for _, type_name in SOURCE_OWNED_JOB_STATUS_TYPE_EXPORTS
    )
    changed = _append_missing_lines(types_init, top_level_imports) or changed
    return changed


def update_pyproject_toml(sdk_info: SdkInfo) -> bool:
    """Update pyproject.toml using regex replacements while preserving formatting."""
    pyproject_path = sdk_info.sdk_dir / "pyproject.toml"

    if not pyproject_path.exists():
        typer.echo(f"pyproject.toml not found at {pyproject_path}. Skipping update.")
        return False

    pyproject_str = pyproject_path.read_text(encoding="utf-8")

    pyproject = tomlkit.loads(pyproject_str)
    project = pyproject.get("project", {})

    authors = project["authors"]
    authors.clear()
    authors.append({"name": "NVIDIA Corporation"})

    project["urls"] = {"Homepage": "https://docs.nvidia.com/nemo/microservices/latest/about/index.html"}

    # Match the supported range of the nemo-platform wheel this SDK ships in.
    version_to_remove = ["3.8", "3.9", "3.10", "3.11", "3.14"]
    for ver in version_to_remove:
        try:
            project["classifiers"].remove(f"Programming Language :: Python :: {ver}")
        except ValueError:
            pass

    project["requires-python"] = ">=3.12,<3.14"
    dependencies = project.get("dependencies")
    if dependencies is not None:
        for dep in list(dependencies):
            if dep.startswith("exceptiongroup"):
                dependencies.remove(dep)
    pyproject["tool"]["ruff"]["target-version"] = "py312"
    pyproject["tool"]["pyright"]["pythonVersion"] = "3.12"

    # Handle versioning
    project.pop("version", None)
    dynamic = project.setdefault("dynamic", [])
    if "version" not in dynamic:
        dynamic.append("version")

    tool = pyproject.setdefault("tool", {})
    hatch_version = tool.setdefault("hatch", {}).setdefault("version", {})
    hatch_version.clear()
    hatch_version["source"] = "nmp-dynamic-versioning"

    tool.pop("uv-dynamic-versioning", None)

    build_requires = pyproject.setdefault("build-system", {}).setdefault("requires", [])
    try:
        build_requires.remove("uv-dynamic-versioning")
    except ValueError:
        pass
    if "nmp-build-tools" not in build_requires:
        build_requires.append("nmp-build-tools")

    # Tweak pytest options
    pytest = pyproject["tool"]["pytest"]["ini_options"]
    pytest_addopts = "-k 'not aiohttp' -Wdefault"
    opts = pytest.get("addopts", "")
    if pytest_addopts not in opts:
        opts += f" {pytest_addopts}"
        pytest["addopts"] = opts.strip()

    # Tweak uv config
    uv_config = pyproject["tool"]["uv"]
    uv_config.pop("conflicts", None)
    uv_config["cache-keys"] = [
        {"file": "pyproject.toml"},
        {"git": {"commit": True, "tags": True}},
    ]
    nmp_build_tools_source = tomlkit.inline_table()
    nmp_build_tools_source["workspace"] = True
    uv_config.setdefault("sources", {})["nmp-build-tools"] = nmp_build_tools_source

    pyproject["dependency-groups"].pop("pydantic-v1", None)

    # Configure wheel build to include the generated SDK package directly and
    # stage extension source packages at build time without import rewriting.
    # Copy first so direct ``update-pyproject`` runs cannot leave Hatch pointing
    # at a missing build hook.
    _copy_build_hook(sdk_info)
    hatch_build = pyproject.setdefault("tool", {}).setdefault("hatch", {}).setdefault("build", {})
    hatch_targets = hatch_build.setdefault("targets", {})
    wheel_target = hatch_targets.setdefault("wheel", {})
    wheel_target["packages"] = ["src/nemo_platform"]
    wheel_target.pop("force-include", None)
    wheel_target.setdefault("hooks", {})["custom"] = _source_package_hook_table()
    sdist_target = hatch_targets.setdefault("sdist", {})
    sdist_target.setdefault("hooks", {})["custom"] = _source_package_hook_table()

    updated_pyproject_str = tomlkit.dumps(pyproject)

    # Only write if changes were made
    if updated_pyproject_str != pyproject_str:
        pyproject_path.write_text(updated_pyproject_str, encoding="utf-8")
        typer.echo(f"  - Updated {pyproject_path}")
    else:
        typer.echo(f"  - No changes needed for {pyproject_path}")

    return True


def get_license_header(file_type: Literal["python"] = "python") -> str:
    """
    Get the standard SPDX license header for NVIDIA files.
    """
    current_year = datetime.now().year
    license_header = f"""\
SPDX-FileCopyrightText: Copyright (c) {current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

    # Use a comment style based on the file type
    if file_type == "python":
        header = "\n".join("# " + line for line in license_header.splitlines()) + "\n\n"
    else:
        raise RuntimeError(f"Unsupported license style: {file_type}")

    # Remove trailing whitespace from all lines
    header = "\n".join(line.rstrip(" \t") for line in header.split("\n"))

    return header


def has_license_header(file_content: str) -> bool:
    """Check if file already has a license header."""
    lines = file_content.splitlines()
    if not lines:
        return False

    # Check first few lines for license header patterns
    first_lines = lines[:10]  # Check first 10 lines
    license_patterns = [
        r"SPDX-FileCopyrightText.*NVIDIA",
        r"Copyright.*NVIDIA",
        r"SPDX-License-Identifier",
    ]

    for line in first_lines:
        for pattern in license_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

    return False


def should_add_license_header(file_path: Path) -> bool:
    """Determine if a file should have a license header added."""
    # Skip certain files
    skip_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".egg-info",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
    ]

    # Skip if file path contains any skip patterns
    file_str = str(file_path)
    for pattern in skip_patterns:
        if pattern in file_str:
            return False

    # Only process Python files
    if file_path.suffix != ".py":
        return False

    # Skip certain specific files
    skip_files = []

    # Allow __init__.py files that are not in the root of the SDK
    if file_path.name in skip_files:
        return False

    return True


def add_license_header_to_file(file_path: Path, license_header: str) -> bool:
    """Add license header to a single file. Returns True if header was added."""
    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Check if license header already exists
        if has_license_header(content):
            return False

        # Handle shebang lines
        lines = content.splitlines(keepends=True)
        insert_pos = 0

        # If file starts with shebang, insert after it
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
            # Add empty line after shebang if there isn't one
            if len(lines) > 1 and not lines[1].strip() == "":
                license_header += "\n"

        # Insert license header
        if insert_pos < len(lines):
            lines.insert(insert_pos, license_header)
        else:
            lines.append(license_header)

        # Write back to file
        file_path.write_text("".join(lines), encoding="utf-8")
        return True

    except (UnicodeDecodeError, PermissionError) as e:
        typer.echo(f"  - Skipped {file_path} ({e})", err=True)
        return False


def process_license_headers(sdk_dir: Path) -> None:
    """Update license headers in all Python files in the SDK."""
    license_header = get_license_header()

    # File patterns to process
    patterns = ["**/*.py"]

    processed_files = 0
    updated_files = 0
    skipped_files = 0

    for pattern in patterns:
        for file_path in sdk_dir.glob(pattern):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if file shouldn't have license header
            if not should_add_license_header(file_path):
                continue

            processed_files += 1

            # Add license header
            if add_license_header_to_file(file_path, license_header):
                updated_files += 1
            else:
                skipped_files += 1

    typer.echo(f"  - Processed {processed_files} files")
    typer.echo(f"  - Updated {updated_files} files with license headers")
    typer.echo(f"  - Skipped {skipped_files} files (already had headers)")


@app.command()
def update_readme() -> None:
    """Merge README override files and replace the generated README."""
    sdk_info = get_sdk_info()

    typer.echo("Updating README...")

    # Check if override directory exists
    if not sdk_info.readme_dir.exists():
        typer.echo(f"README override directory not found: {sdk_info.readme_dir}. Skipping override.")
        return

    # Merge README files
    merged_content = merge_readme_files(sdk_info.readme_dir)

    # Write to SDK README
    readme_path = sdk_info.sdk_dir / "README.md"
    readme_path.write_text(merged_content, encoding="utf-8")

    typer.echo(f"  - Updated {readme_path}")
    typer.echo("README update completed!")


@app.command()
def update_pyproject() -> None:
    """Update pyproject.toml with regex replacements."""
    sdk_info = get_sdk_info()

    typer.echo("Updating pyproject.toml...")

    if update_pyproject_toml(sdk_info):
        typer.echo("pyproject.toml update completed!")


@app.command()
def replace_strings() -> None:
    """Apply string replacements to source code files."""
    sdk_info = get_sdk_info()

    typer.echo("Applying string replacements...")

    # Get replacements
    replacements = get_string_replacements()

    if not replacements:
        typer.echo("  - No replacements configured")
        return

    typer.echo(f"  - Applying {len(replacements)} replacement rules")
    for old_string, new_string in replacements:
        typer.echo(f"    '{old_string[:50]}...' -> '{new_string[:50]}...'")

    # Apply replacements
    apply_string_replacements(sdk_info.sdk_dir, replacements)

    typer.echo("String replacements completed!")


@app.command()
def cleanup_api_index() -> None:
    """Remove API index entries for generated SDK artifacts removed by post-processing."""
    sdk_info = get_sdk_info()

    typer.echo("Cleaning API index...")
    clean_api_index(sdk_info)
    typer.echo("API index cleanup completed!")


@app.command()
def ensure_source_owned_aliases() -> None:
    """Create source-owned compatibility type aliases in the generated SDK tree."""
    sdk_info = get_sdk_info()

    typer.echo("Ensuring source-owned type aliases...")
    if ensure_source_owned_type_aliases(sdk_info):
        typer.echo("  - Updated source-owned type aliases")
    else:
        typer.echo("  - Source-owned type aliases already present")
    typer.echo("Source-owned type alias update completed!")


@app.command()
def copy_license() -> None:
    """Copy LICENSE file from overrides to SDK directory."""
    sdk_info = get_sdk_info()

    typer.echo("Copying LICENSE file...")

    # Check for override LICENSE first
    override_license = sdk_info.overrides_dir / "LICENSE"

    source_license = None
    if override_license.exists():
        source_license = override_license
        typer.echo(f"  - Using override LICENSE from {override_license}")
    else:
        typer.echo("No LICENSE file found in overrides. Skipping override.")
        return

    # Copy to SDK directory
    dest_license = sdk_info.sdk_dir / "LICENSE"
    shutil.copy2(source_license, dest_license)

    typer.echo(f"  - Copied to {dest_license}")
    typer.echo("LICENSE copy completed!")


@app.command()
def copy_build_hook() -> None:
    """Copy SDK build hook from overrides to the generated SDK directory."""
    sdk_info = get_sdk_info()

    typer.echo("Copying build hook...")

    destination = _copy_build_hook(sdk_info)
    typer.echo(f"  - Copied to {destination}")
    typer.echo("Build hook copy completed!")


@app.command()
def copy_source_overrides() -> None:
    """Copy SDK source overrides into the generated SDK directory."""
    sdk_info = get_sdk_info()

    typer.echo("Copying source overrides...")

    source = sdk_info.overrides_dir / "src"
    if not source.exists():
        typer.echo(f"No source overrides found at {source}. Skipping override.")
        return

    destination = sdk_info.sdk_dir / "src"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    typer.echo(f"  - Copied to {destination}")
    typer.echo("Source override copy completed!")


@app.command()
def check_stainless_references() -> None:
    """Check for any remaining references to Stainless URLs in the SDK code."""
    sdk_info = get_sdk_info()

    typer.echo("Checking for Stainless references...")

    findings = check_for_stainless_references(sdk_info.sdk_dir)

    if not findings:
        typer.echo("  ✓ No Stainless references found!")
        return

    typer.echo(f"  ✗ Found {len(findings)} Stainless references:")

    # Group findings by file
    files_with_findings = {}
    for file_path, line_num, line_content in findings:
        rel_path = file_path.relative_to(sdk_info.sdk_dir)
        if rel_path not in files_with_findings:
            files_with_findings[rel_path] = []
        files_with_findings[rel_path].append((line_num, line_content))

    # Display findings
    for file_path, file_findings in files_with_findings.items():
        typer.echo(f"\n  {file_path}:")
        for line_num, line_content in file_findings:
            typer.echo(f"    Line {line_num}: {line_content}")

    typer.echo(f"Error: Found {len(findings)} Stainless references that need to be addressed", err=True)
    raise typer.Exit(code=1)


@app.command()
def remove_stats_file() -> None:
    """
    Remove the stats file if it exists.
    Note: This is important, so the changes to the SDK can be merged without conflicts.
    """
    sdk_info = get_sdk_info()

    typer.echo("Removing stats file...")

    stats_file = sdk_info.sdk_dir / ".stats.yml"
    if stats_file.exists():
        stats_file.unlink()
        typer.echo(f"  - Removed {stats_file}")
    else:
        typer.echo(f"  - No stats file found at {stats_file}. Skipping removal.")

    typer.echo("Stats file removal completed!")


@app.command()
def save_nmp_context() -> None:
    """
    Save the inputs used to generate the current version of the SDK.
    This context is used for checking if the SDK is up to date with the main OpenAPI spec and Stainless config.
    """
    sdk_info = get_sdk_info()

    typer.echo("Saving generation context...")

    nmpcontext_dir = sdk_info.sdk_dir / ".nmpcontext"
    nmpcontext_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(sdk_info.openapi_spec_file, nmpcontext_dir / "openapi.yaml")
    shutil.copy(sdk_info.stainless_config_file, nmpcontext_dir / "stainless.yaml")

    typer.echo(f"  - Copied to {nmpcontext_dir}")


@app.command()
def update_license_headers() -> None:
    """Update license headers in all Python files in the SDK."""
    sdk_info = get_sdk_info()

    typer.echo("Updating license headers in all Python files...")

    process_license_headers(sdk_info.sdk_dir)

    typer.echo("License headers update completed!")


@app.command()
def ensure_api_image_field() -> None:
    """Ensure dynamic version metadata and ``__image_tag__`` are present in ``_version.py``.

    Stainless rewrites ``_version.py`` with a literal version. This command
    restores the dynamic package-metadata lookup and the development default
    image tag field.
    """
    sdk_info = get_sdk_info()

    typer.echo("Ensuring dynamic version metadata in _version.py...")

    version_file = sdk_info.sdk_dir / "src" / sdk_info.module_name / "_version.py"
    content = version_file.read_text(encoding="utf-8")
    expected = sdk_version_file_content(WRAPPER_DISTRIBUTION_NAME)
    if content == expected:
        typer.echo("  - Dynamic version metadata already present, skipping.")
        return

    version_file.write_text(expected, encoding="utf-8")
    typer.echo(f"  - Updated {version_file}")


@app.command()
def update_all() -> None:
    """Run all updates: README, pyproject.toml, LICENSE, string replacements, and license headers."""
    typer.echo("Running all post-generation updates...")

    # Run all update commands
    update_readme()
    typer.echo()
    update_pyproject()
    typer.echo()
    copy_license()
    typer.echo()
    copy_build_hook()
    typer.echo()
    copy_source_overrides()
    typer.echo()
    ensure_source_owned_aliases()
    typer.echo()
    replace_strings()
    typer.echo()
    cleanup_api_index()
    typer.echo()
    update_license_headers()
    typer.echo()
    remove_stats_file()
    typer.echo()
    ensure_api_image_field()
    typer.echo()
    save_nmp_context()
    typer.echo()
    inject_exist_ok()
    typer.echo()

    typer.echo("\nAll post-generation updates completed successfully!")
