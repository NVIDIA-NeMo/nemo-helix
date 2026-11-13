# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for reduced uv workspaces used by Docker image builds."""

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
WORKSPACE_SLICES = ("automodel", "customizer", "rl", "unsloth")
SDK_ALIAS_PACKAGES = ("filesets",)
WANDB_PACKAGE_SPEC_RE = re.compile(r"(?<![\w./-])wandb(?:\[[^\]]+\])?(?:==|~=|!=|<=|>=|<|>)[^\\\s\"']+")
DOCKER_IMAGE_WANDB_CONFIG_PATHS = (
    Path("docker/Dockerfile.nhx-customizer-tasks"),
    Path("docker/Dockerfile.nhx-safe-synthesizer-tasks"),
    Path("docker/automodel/Dockerfile.nhx-automodel-base"),
    Path("docker/automodel/no_override_requirements.txt"),
    Path("docker/rl/Dockerfile.nhx-rl-base"),
    Path("docker/unsloth/no_override_requirements.txt"),
)
CVE_PACKAGE_FLOORS = (
    ("thrift", "0.25.0", (Path("docker/automodel/Dockerfile.nhx-automodel-base"),)),
    ("anyio", "4.14.2", (Path("docker/automodel/Dockerfile.nhx-automodel-base"),)),
    ("httpx2", "2.12.0", (Path("docker/automodel/Dockerfile.nhx-automodel-base"),)),
    ("httpcore2", "2.10.0", (Path("docker/automodel/Dockerfile.nhx-automodel-base"),)),
    (
        "virtualenv",
        "21.14.9",
        (
            Path("docker/automodel/Dockerfile.nhx-automodel-base"),
            Path("docker/rl/Dockerfile.nhx-rl-base"),
        ),
    ),
    (
        "virtualenv",
        "21.14.13",
        (
            Path("docker/locks/nhx-gym-tasks/pyproject.toml"),
            Path("docker/locks/nhx-gym-host/pyproject.toml"),
        ),
    ),
    (
        "notebook",
        "7.6.3",
        (
            Path("docker/automodel/Dockerfile.nhx-automodel-base"),
            Path("docker/Dockerfile.nhx-customizer-tasks"),
            Path("docker/Dockerfile.nhx-unsloth-training"),
        ),
    ),
    (
        "mlflow-skinny",
        "3.15.0",
        (
            Path("pyproject.toml"),
            Path("services/rl/pyproject.toml"),
            Path("services/unsloth/pyproject.toml"),
            Path("docker/automodel/Dockerfile.nhx-automodel-base"),
            Path("docker/Dockerfile.nhx-unsloth-training"),
            Path("docker/rl/Dockerfile.nhx-rl-base"),
            Path("docker/locks/nhx-gym-tasks/pyproject.toml"),
        ),
    ),
    (
        "sqlparse",
        "0.6.0",
        (
            Path("pyproject.toml"),
            Path("docker/automodel/Dockerfile.nhx-automodel-base"),
            Path("docker/Dockerfile.nhx-unsloth-training"),
            Path("docker/rl/Dockerfile.nhx-rl-base"),
            Path("docker/locks/nhx-gym-tasks/pyproject.toml"),
        ),
    ),
)
FULL_MLFLOW_SPEC_RE = re.compile(r"(?<![\w./-])mlflow(?:\[[^\]]+\])?(?:==|~=|!=|<=|>=|<|>)")


def _load_pyproject(path: Path) -> dict:
    """Parse a pyproject.toml into a dict."""
    with open(path, "rb") as pyproject:
        return tomllib.load(pyproject)


def _normalize_package_name(name: str) -> str:
    """Normalize a PEP 503 package name for comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_name_from_spec(spec: str) -> str:
    """Return the normalized package name from a PEP 508 requirement string."""
    match = re.match(r"[A-Za-z0-9_.-]+", spec)
    assert match is not None
    return _normalize_package_name(match.group())


def _notice_wandb_version() -> str:
    """Read the wandb version documented in NOTICE."""
    for line in (ROOT / "NOTICE").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("wandb package version:"):
            return line.split(":", maxsplit=1)[1].strip()
    raise AssertionError("NOTICE does not document the wandb package version")


def _wandb_package_specs(path: Path) -> list[str]:
    """Collect wandb version specs from a Dockerfile or requirements file."""
    specs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_without_comment = line.split("#", maxsplit=1)[0]
        specs.extend(WANDB_PACKAGE_SPEC_RE.findall(line_without_comment))
    return specs


def _package_specs(path: Path, package: str) -> list[str]:
    """Collect version specs for one package from a Dockerfile or manifest."""
    pattern = re.compile(rf"(?<![\w./-]){re.escape(package)}(?:\[[^\]]+\])?(?:==|~=|!=|<=|>=|<|>)[^\\\s\"']+")
    specs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_without_comment = line.split("#", maxsplit=1)[0]
        specs.extend(pattern.findall(line_without_comment))
    return specs


def _workspace_sources(pyproject: dict) -> set[str]:
    """Return workspace-source package names from a uv pyproject table."""
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    return {
        _normalize_package_name(name)
        for name, source in sources.items()
        if isinstance(source, dict) and source.get("workspace") is True
    }


@pytest.mark.parametrize(
    "path",
    [pytest.param(ROOT / path, id=str(path)) for path in DOCKER_IMAGE_WANDB_CONFIG_PATHS],
)
def test_distributed_image_wandb_specs_use_notice_floor(path: Path) -> None:
    """Docker image wandb constraints must use the reviewed wandb-core floor."""
    expected = f"wandb>={_notice_wandb_version()}"
    assert path.exists()

    specs = _wandb_package_specs(path)

    assert specs
    assert all(spec == expected for spec in specs)


@pytest.mark.parametrize(
    ("package", "floor", "path"),
    [
        pytest.param(package, floor, ROOT / path, id=f"{package}-{path}")
        for package, floor, paths in CVE_PACKAGE_FLOORS
        for path in paths
    ],
)
def test_distributed_image_cve_package_specs_use_reviewed_floor(
    package: str,
    floor: str,
    path: Path,
) -> None:
    """CVE-remediated package constraints must retain their reviewed floor."""
    specs = _package_specs(path, package)

    assert specs
    assert all(spec.startswith(f"{package}>={floor}") for spec in specs)


@pytest.mark.parametrize("project", ("nhx-gym-tasks", "nhx-gym-host"))
def test_gym_locks_resolve_fixed_virtualenv(project: str) -> None:
    """Standalone image locks must not retain the vulnerable Ray dependency."""
    from packaging.version import Version

    lock = _load_pyproject(ROOT / "docker/locks" / project / "uv.lock")
    virtualenv = next(package for package in lock["package"] if package["name"] == "virtualenv")
    assert Version(virtualenv["version"]) >= Version("21.14.13")


@pytest.mark.parametrize("versions", [("26.2.2",), ("26.2.2", "26.3"), ("26.3",)])
def test_final_api_pip_guard_rejects_duplicate_or_mismatched_metadata(
    tmp_path: Path, versions: tuple[str, ...]
) -> None:
    text = (ROOT / "docker/Dockerfile.nhx-api").read_text()
    clear = text.index("RUN rm -rf /app/.venv")
    copy = text.index("COPY --chown=1000:1000 --from=builder /app/.venv /app/.venv")
    guard = text.index("RUN /app/.venv/bin/python - <<'PY_PIP_INTEGRITY'")
    assert clear < copy < text.index("sh /bin/cve-cleanup.sh") < guard
    program = text.split("<<'PY_PIP_INTEGRITY'\n", 1)[1].split("\nPY_PIP_INTEGRITY", 1)[0]
    package = tmp_path / "pip"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '26.2.2'\n")
    for version in versions:
        metadata = tmp_path / f"pip-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: pip\nVersion: {version}\n")
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (versions == ("26.2.2",)), result.stderr


def test_ngc_notebook_is_replaced_before_system_copy_cleanup() -> None:
    """Keep the patched venv Notebook while removing scanner-visible NGC copies."""
    removals = "docker/removals/files/base/pytorch-ngc-common.txt"
    patterns = (ROOT / removals).read_text().splitlines()
    assert "/usr/local/lib/python3.12/dist-packages/notebook" in patterns
    assert "/usr/local/lib/python3.12/dist-packages/notebook-*.dist-info" in patterns
    for path in (
        "docker/automodel/Dockerfile.nhx-automodel-base",
        "docker/Dockerfile.nhx-customizer-tasks",
        "docker/Dockerfile.nhx-unsloth-training",
    ):
        text = (ROOT / path).read_text()
        assert text.index('"notebook>=7.6.3,<8"') < text.index(f"source={removals}")


def test_safe_synthesizer_generation_stack_uses_patched_upstream_release() -> None:
    project = _load_pyproject(ROOT / "plugins/nemo-safe-synthesizer/pyproject.toml")
    assert "nemo-safe-synthesizer==0.1.17" in project["project"]["dependencies"]
    dockerfile = ROOT / "docker/Dockerfile.nhx-safe-synthesizer-tasks"
    assert "nemo-safe-synthesizer[engine,cu129]==0.1.17" in dockerfile.read_text()
    assert "https://wheels.vllm.ai/0.32.0/cu129" in dockerfile.read_text()
    assert _package_specs(dockerfile, "vllm") == ["vllm==0.32.0+cu129"]
    assert _package_specs(dockerfile, "setuptools") == ["setuptools>=83.0.0"]
    constraints = ROOT / "plugins/nemo-safe-synthesizer/constraints.txt"
    assert _package_specs(constraints, "setuptools") == ["setuptools>=83.0.0"]
    for name in ("flashinfer-python", "flashinfer-cubin"):
        assert _package_specs(dockerfile, name) == [f"{name}==0.7.0.post1"]
    assert _package_specs(dockerfile, "flashinfer-jit-cache") == ["flashinfer-jit-cache==0.7.0.post1+cu129"]


def test_safe_synthesizer_sentence_transformers_security_pin() -> None:
    """The isolated NSS engine must not retain its upstream vulnerable pin."""
    specs = _package_specs(ROOT / "plugins/nemo-safe-synthesizer/constraints.txt", "sentence-transformers")
    assert specs == ["sentence-transformers==5.6.1"]


def test_rl_generation_security_upgrade_precedes_prefetch_and_survives_source_copy() -> None:
    text = (ROOT / "docker/rl/Dockerfile.nhx-rl-base").read_text()
    upgrade = text.index("RUN uv add --optional vllm --no-sync")
    assert upgrade < text.index("uv sync ${UV_SYNC_MODE} --extra vllm")
    for requirement in (
        "vllm==0.30.0",
        "flashinfer-python==0.6.18.post1",
        "flashinfer-cubin==0.6.18.post1",
        "flashinfer-jit-cache==0.6.18.post1",
        "nvidia-cutlass-dsl[cu13]==4.7.1",
    ):
        assert f"'{requirement}'" in text
    backup = text.index("cp pyproject.toml uv.lock /tmp/nhx-nemo-rl-lock/")
    source_copy = text.index("--exclude=**/.git . /opt/nemo-rl")
    restore = text.index("RUN cp /tmp/nhx-nemo-rl-lock/pyproject.toml /tmp/nhx-nemo-rl-lock/uv.lock /opt/nemo-rl/")
    assert upgrade < backup < source_copy < restore


def test_rl_virtualenv_patch_precedes_unreferenced_cache_cleanup() -> None:
    """Patch runtime environments and remove old archives, not just metadata."""
    text = (ROOT / "docker/rl/Dockerfile.nhx-rl-base").read_text()
    assert 'm.version("virtualenv")' in text
    assert '--directory /tmp --python "${py}" --upgrade-package virtualenv "virtualenv>=21.14.9"' in text
    assert text.index('"virtualenv>=21.14.9"') < text.index('"${d}virtualenv-*.dist-info"')


def test_automodel_cve_layer_does_not_pin_full_mlflow_with_cryptography() -> None:
    """Full MLflow 3.15 requires cryptography<50 and cannot coinstall with the image floor."""
    text = (ROOT / "docker/automodel/Dockerfile.nhx-automodel-base").read_text(encoding="utf-8")
    assert "cryptography>=50.0.0,<51" in text
    assert "mlflow-skinny>=3.15.0" in text
    assert not FULL_MLFLOW_SPEC_RE.search(text)


@pytest.mark.parametrize("slice_name", WORKSPACE_SLICES)
def test_docker_workspace_slice_contains_all_workspace_sources(slice_name):
    """Every workspace source must name a package copied into the image slice."""
    workspace_path = ROOT / "docker" / slice_name / "pyproject.workspace.toml"
    workspace = _load_pyproject(workspace_path)
    member_paths = workspace["tool"]["uv"]["workspace"]["members"]
    member_projects = [
        (ROOT / member / "pyproject.toml", _load_pyproject(ROOT / member / "pyproject.toml")) for member in member_paths
    ]
    member_names = {_normalize_package_name(project["project"]["name"]) for _, project in member_projects}

    for project_path, project in [(workspace_path, workspace), *member_projects]:
        missing_sources = _workspace_sources(project) - member_names
        assert not missing_sources, (
            f"{slice_name} workspace is missing {sorted(missing_sources)} referenced by "
            f"{project_path.relative_to(ROOT)}"
        )


@pytest.mark.parametrize("slice_name", WORKSPACE_SLICES)
@pytest.mark.parametrize("package", SDK_ALIAS_PACKAGES)
def test_sdk_editable_workspace_slice_includes_sdk_alias_package(slice_name: str, package: str) -> None:
    """The SDK imports alias packages lazily when those resources are accessed."""
    workspace_path = ROOT / "docker" / slice_name / "pyproject.workspace.toml"
    workspace = _load_pyproject(workspace_path)
    member_paths = set(workspace["tool"]["uv"]["workspace"]["members"])
    source_names = _workspace_sources(workspace)
    dockerfile_text = (ROOT / "docker" / slice_name / "Dockerfile.platform-workspace").read_text(encoding="utf-8")

    assert f"packages/{package}" in member_paths
    assert package in source_names
    assert f"COPY packages/{package} packages/{package}" in dockerfile_text


def test_gym_host_lock_has_local_source_for_first_party_build_tools() -> None:
    """The standalone gym-host lock must not resolve first-party build hooks from PyPI."""
    lock_project = _load_pyproject(ROOT / "docker/locks/nhx-gym-host/pyproject.toml")
    uv_config = lock_project["tool"]["uv"]
    dockerfile_text = (ROOT / "docker/gym-host/Dockerfile").read_text(encoding="utf-8")

    assert uv_config["sources"]["nhx-build-tools"] == {
        "path": "../../../packages/nhx_build_tools",
        "editable": True,
    }
    assert uv_config["extra-build-dependencies"]["nemo-sandboxed-gym"] == ["nhx-build-tools"]
    assert "COPY packages/nhx_build_tools /app/packages/nhx_build_tools" in dockerfile_text


def test_deployments_plugin_is_optional_for_models_service():
    """The lazily loaded deployments backend must remain an optional package."""
    models = _load_pyproject(ROOT / "services/core/models/pyproject.toml")
    dependency_names = {_package_name_from_spec(dependency) for dependency in models["project"]["dependencies"]}

    assert "nemo-deployments-plugin" not in dependency_names, (
        "nhx-models loads the deployments backend lazily, so nemo-deployments-plugin "
        "must not be an unconditional dependency"
    )
    assert "nemo-deployments-plugin" not in _workspace_sources(models), (
        "an optional package must not be declared as a required workspace source"
    )
