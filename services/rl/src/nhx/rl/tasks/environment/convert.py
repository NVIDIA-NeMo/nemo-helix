# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prime Intellect hub → adapter-wheels-v1 conversion (CLI-first, internet on host)."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from nhx.customization_common.service.constants import SANDBOX_DATASET_PATH
from nhx.rl.tasks.environment.allowlist import DEFAULT_ADAPTER_AGENT
from nhx.rl.tasks.environment.package import (
    ConvertedPackage,
    dataset_row_from_verifiers,
    hub_id_to_package_name,
    hub_id_to_vf_env_id,
    write_adapter_wheels_package,
    write_dataset_jsonl,
)
from nhx.rl.tasks.environment.validate import SANDBOX_VENV_DISTRIBUTIONS, validate_dataset_rows
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

PRIME_HUB_SIMPLE_INDEX = "https://hub.primeintellect.ai/primeintellect/simple/"
DEFAULT_VERIFIERS_SPEC = "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@v0.3.1"

# Training image is linux/amd64; resolve and download for that, not this host.
# pip matches --platform tags literally, so several glibc floors are listed.
TARGET_WHEEL_PLATFORMS = (
    "manylinux_2_39_x86_64",
    "manylinux_2_28_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
)
TARGET_UV_PLATFORM = "x86_64-unknown-linux-gnu"
# Gym 0.7 requires Python>=3.13.14.
TARGET_PYTHON_VERSION = "3.13.14"
_TARGET_PLATFORM_TAG_RE = re.compile(r"^(any|linux_x86_64|manylinux[0-9_.]*_x86_64)$")

# Gym rebuilds each per-server venv from empty and installs the agent before the environment,
# so an offline package has to carry the agent side too. Values mirror scripts/grpo-examples/
# gym_to_env_package.py, which already does this for wheels-v1.
SETUPTOOLS_PKG_RESOURCES_CEILING = "81"  # 81 dropped pkg_resources, which Gym's hydra 1.3 imports
HYDRA_CORE_SPEC = ">=1.3,<1.4"
OMEGACONF_SPEC = ">=2.2,<2.4"
# Resolved for every package: pip is what `uv venv --seed` installs, setuptools/setuptools-scm are
# Gym's build-system.requires for building the agent from the image's source tree.
GYM_VENV_REQUIREMENTS = (
    "pip",
    f"setuptools>=61,<{SETUPTOOLS_PKG_RESOURCES_CEILING}",
    "setuptools-scm",
    f"hydra-core{HYDRA_CORE_SPEC}",
    f"omegaconf{OMEGACONF_SPEC}",
)

# Snapshot of the hub env's HF dataset, mounted at SANDBOX_DATASET_PATH.
HUB_ENV_DATASET_FILENAME = "hub_environment.parquet"
DATASET_PATH_ARG = "dataset_path"
# Isolated Hugging Face cache from convert, for loaders that ignore dataset_path.
HUB_ENV_HF_CACHE_DIRNAME = ".huggingface"


@dataclass(frozen=True)
class ConvertEnvironmentSpec:
    hub_id: str
    out_dir: Path
    hub_version: str | None = None  # pin the hub release; unset follows the index
    dataset_dir: Path | None = None
    vf_env_id: str | None = None
    vf_env_args: dict[str, Any] | None = None  # host-side load args; dataset_path is overwritten
    adapter_agent: str = DEFAULT_ADAPTER_AGENT
    dataset_size: int = -1
    dataset_seed: int | None = None
    validation_fraction: float = 0.0
    verifiers_spec: str = DEFAULT_VERIFIERS_SPEC
    extra_wheels: tuple[str, ...] = ()
    wheels_dir: Path | None = None  # pre-vendored *.whl; required non-empty if set
    # The agent side of the closure. Gym pins each venv to the versions the training image has
    # installed, and nemo-gym is not on an index, so these cannot be inferred from the hub id.
    # A NeMo-RL checkout at the commit docker-bake.hcl pins (NEMO_RL_REF), submodules included.
    # Everything the agent closure needs is in there, so nothing has to be looked up by hand.
    nemo_rl_root: Path | None = None
    gym_root: Path | None = None  # defaults to nemo_rl_root's Gym submodule; built, not downloaded
    nemo_gym_version: str | None = None  # fail unless gym_root builds exactly this
    ray_version: str | None = None  # defaults to the nemo_rl_root lock pin
    openai_version: str | None = None


def _run_build_step(cmd: list[str]) -> None:
    """Run a resolve or build step with its chatter on stderr, never on stdout."""
    # stdout carries this CLI's JSON result. `uv pip compile` echoes the compiled
    # requirements even with --output-file, and pip download prints progress, so an
    # inherited stdout lands both in front of the JSON and breaks a strict consumer.
    subprocess.run(cmd, check=True, stdout=sys.stderr)


def rl_dependency_policy(nemo_rl_root: Path) -> tuple[list[str], list[str]]:
    """Read the uv limits the Gym host applies when it installs the wheelhouse.

    The host starts in the NeMo-RL tree, so ``uv pip install`` honors that checkout's
    ``[tool.uv]`` override-dependencies and constraint-dependencies. A closure resolved
    without them installs versions the host then rejects.
    """
    pyproject = nemo_rl_root / "pyproject.toml"
    if not pyproject.is_file():
        raise ValueError(
            f"--nemo-rl-root has no pyproject.toml: {nemo_rl_root}. "
            "Point it at the NeMo-RL checkout NEMO_RL_REF pins. The training image is built "
            "from that commit, and the Gym host installs this wheelhouse under its uv policy."
        )
    uv_config = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    missing = [key for key in ("override-dependencies", "constraint-dependencies") if not uv_config.get(key)]
    if missing:
        raise ValueError(
            f"{pyproject} has no [tool.uv] {', '.join(missing)}. Those tables are what the Gym "
            "host applies during the offline install."
        )
    return (
        _keep_pkg_resources_setuptools(list(uv_config["override-dependencies"])),
        list(uv_config["constraint-dependencies"]),
    )


def _keep_pkg_resources_setuptools(overrides: list[str]) -> list[str]:
    """Keep a setuptools override inside the range Gym's server venvs accept."""
    ceiling = SpecifierSet(f"<{SETUPTOOLS_PKG_RESOURCES_CEILING}")
    clamped: list[str] = []
    for item in overrides:
        requirement = Requirement(item)
        if canonicalize_name(requirement.name) != "setuptools":
            clamped.append(item)
            continue
        requirement.specifier &= ceiling
        clamped.append(str(requirement))
    return clamped


def _compile_pinned_requirements(
    work_dir: Path,
    packages: list[str],
    *,
    extra_index_url: str | None = None,
    nemo_rl_root: Path | None = None,
) -> Path:
    """Pin ``packages`` before download so ``pip download --no-deps`` cannot vendor duplicates."""
    work_dir.mkdir(parents=True, exist_ok=True)
    requirements_in = work_dir / "requirements.in"
    requirements_in.write_text("\n".join(packages) + "\n", encoding="utf-8")
    pinned = work_dir / "requirements.txt"
    # --no-config ignores this repo's [tool.uv]. The Gym host applies the NeMo-RL
    # checkout's override-dependencies and constraint-dependencies, passed explicitly.
    cmd = [
        "uv",
        "pip",
        "compile",
        str(requirements_in),
        "--output-file",
        str(pinned),
        "--no-header",
        "--no-config",
        "--python-platform",
        TARGET_UV_PLATFORM,
        "--python-version",
        TARGET_PYTHON_VERSION,
    ]
    if nemo_rl_root is not None:
        overrides, constraints = rl_dependency_policy(nemo_rl_root)
        override = work_dir / "override.txt"
        constraint = work_dir / "constraint.txt"
        override.write_text("\n".join(overrides) + "\n", encoding="utf-8")
        constraint.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        cmd.extend(["--override", str(override), "--constraint", str(constraint)])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url, "--index-strategy", "unsafe-best-match"])
    logger.info("Running: %s", " ".join(cmd))
    _run_build_step(cmd)
    logger.info("Resolved closure:\n%s", pinned.read_text(encoding="utf-8").strip())
    return pinned


def _run_pip_download(
    dest: Path,
    *,
    requirements_file: Path,
    extra_index_url: str | None = None,
) -> None:
    """Download the already-pinned closure with ``pip download --no-deps``."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(dest),
        "--no-cache-dir",
        "--no-deps",
        "--python-version",
        TARGET_PYTHON_VERSION,
    ]
    for platform_tag in TARGET_WHEEL_PLATFORMS:
        cmd.extend(["--platform", platform_tag])
    cmd.extend(["-r", str(requirements_file)])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    logger.info("Running: %s", " ".join(cmd))
    _run_build_step(cmd)


def _sdist_build_python() -> str:
    """Major.minor interpreter for building pure-Python sdists on this host.

    Marker resolution uses ``TARGET_PYTHON_VERSION`` so Gym's ``>=3.13.14``
    floor is satisfied. uv 0.10.10 cannot download 3.13.14 or 3.13.15, and
    these sdists are pure Python, so the installed 3.13 emits a compatible wheel.
    """
    major, minor, *_rest = TARGET_PYTHON_VERSION.split(".")
    return f"{major}.{minor}"


def _build_downloaded_sdists(wheels_dir: Path) -> None:
    """Build source artifacts while the host has egress."""
    sdists = sorted(path for path in wheels_dir.iterdir() if path.is_file() and path.suffix != ".whl")
    build_python = _sdist_build_python()
    for sdist in sdists:
        cmd = [
            "uv",
            "run",
            "--no-project",
            "--python",
            build_python,
            "--with",
            "pip",
            "python",
            "-m",
            "pip",
            "wheel",
            "--wheel-dir",
            str(wheels_dir),
            "--no-deps",
            str(sdist),
        ]
        logger.info("Building wheel from source artifact: %s", " ".join(cmd))
        _run_build_step(cmd)
        sdist.unlink()


def _drop_image_gym_wheel(wheels_dir: Path) -> None:
    """Remove the nemo-gym wheel after it has been used to resolve the closure.

    The training image already has Gym, including the patched verifiers pin.
    Installing this wheel puts that copy first on ``PYTHONPATH``, and Gym then
    reads the unpatched ``requirements.txt`` shipped inside it.
    """
    for wheel in wheels_dir.glob("nemo_gym-*.whl"):
        wheel.unlink()


def _assert_complete_wheel_closure(wheels_dir: Path, requirements_file: Path) -> None:
    """Ensure every pinned distribution has a wheel in the package."""
    required = {
        canonicalize_name(Requirement(line).name)
        for raw_line in requirements_file.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith(("#", "-"))
    }
    provided = {canonicalize_name(str(parse_wheel_filename(wheel.name)[0])) for wheel in wheels_dir.glob("*.whl")}
    missing = sorted(required - provided)
    if missing:
        raise RuntimeError(
            "adapter-wheels-v1 requires a complete offline wheel closure; "
            f"no wheel was produced for: {', '.join(missing)}"
        )


def _tag_targets_training_image(interpreter: str, abi: str, platform: str) -> bool:
    """Whether one wheel tag is compatible with the CPython 3.13 linux/amd64 runtime."""
    if not _TARGET_PLATFORM_TAG_RE.fullmatch(platform):
        return False

    # Wheel tags are cp313, not cp31314. The patch level is only for uv's marker check.
    version_parts = TARGET_PYTHON_VERSION.split(".")
    target_major, target_minor = int(version_parts[0]), int(version_parts[1])
    target = f"{target_major}{target_minor}"
    if abi == "none":
        return interpreter in {f"cp{target}", f"py{target}", f"py{target_major}"}
    if interpreter == f"cp{target}" and abi in {f"cp{target}", "abi3"}:
        return True
    if abi != "abi3":
        return False

    match = re.fullmatch(r"cp(\d)(\d+)", interpreter)
    return bool(match and int(match.group(1)) == target_major and int(match.group(2)) <= target_minor)


def assert_wheels_target_platform(wheels_dir: Path) -> None:
    """Reject wheels whose Python, ABI, or platform tags miss the training image."""
    incompatible: dict[str, list[str]] = {}
    for wheel in sorted(wheels_dir.glob("*.whl")):
        tags = parse_wheel_filename(wheel.name)[3]
        if not any(_tag_targets_training_image(tag.interpreter, tag.abi, tag.platform) for tag in tags):
            incompatible[wheel.name] = sorted(str(tag) for tag in tags)
    if incompatible:
        listed = "\n  ".join(f"{name} -> {', '.join(tags)}" for name, tags in incompatible.items())
        raise RuntimeError(
            f"{len(incompatible)} vendored wheel(s) are not installable on the training image "
            f"(expected Python {TARGET_PYTHON_VERSION} linux/amd64):\n  {listed}"
        )


GYM_SUBMODULE_PATH = "3rdparty/Gym-workspace/Gym"


def _lock_pin(nemo_rl_root: Path, distribution: str) -> str | None:
    """Read a distribution's pinned version out of NeMo-RL's ``uv.lock``.

    Gym stamps ``ray``/``openai`` into every sub-venv from what the training image has
    installed, and the image resolves those from this lock -- not from Gym's own, which pins
    different versions.
    """
    lock = nemo_rl_root / "uv.lock"
    if not lock.is_file():
        raise ValueError(f"{nemo_rl_root} has no uv.lock; expected a NeMo-RL checkout")
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    # uv.lock can carry more than one openai (sglang pins another). The version on
    # the nemo-gym package is the one Gym stamps into every per-server venv.
    nemo_gym = next((pkg for pkg in packages if pkg.get("name") == "nemo-gym"), None)
    if nemo_gym is not None:
        pinned = {
            dep["version"]
            for dep in nemo_gym.get("dependencies", [])
            if dep.get("name") == distribution and "version" in dep
        }
        if len(pinned) == 1:
            return pinned.pop()
    versions = {pkg["version"] for pkg in packages if pkg.get("name") == distribution and "version" in pkg}
    if len(versions) != 1:
        return None
    return versions.pop()


def _resolve_image_pins(spec: ConvertEnvironmentSpec) -> tuple[Path | None, str | None, str | None]:
    """Fill gym_root/ray/openai from ``nemo_rl_root``, leaving explicit values untouched."""
    gym_root, ray_version, openai_version = spec.gym_root, spec.ray_version, spec.openai_version
    if spec.nemo_rl_root is None:
        return gym_root, ray_version, openai_version

    if gym_root is None:
        candidate = spec.nemo_rl_root / GYM_SUBMODULE_PATH
        if not (candidate / "pyproject.toml").is_file():
            raise ValueError(
                f"{candidate} is not a Gym checkout. Clone NeMo-RL with --recurse-submodules, "
                "or pass --gym-root explicitly."
            )
        gym_root = candidate
    ray_version = ray_version or _lock_pin(spec.nemo_rl_root, "ray")
    openai_version = openai_version or _lock_pin(spec.nemo_rl_root, "openai")
    return gym_root, ray_version, openai_version


def _write_nemo_gym_metadata_project(gym_root: Path, dest: Path, version: str) -> None:
    """A wheel that carries nemo-gym's dependency metadata and none of its server trees.

    ``uv build`` of the checkout packs every built-in server. The wheel is only an input
    to the resolver, and it is deleted before the package is written.
    """
    project = tomllib.loads((gym_root / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    dependencies = project.get("dependencies", [])
    dev = project.get("optional-dependencies", {}).get("dev", [])

    def array(items: list[str]) -> str:
        return "[\n" + "".join(f"  {json.dumps(item)},\n" for item in items) + "]"

    dest.mkdir(parents=True)
    (dest / "pyproject.toml").write_text(
        f"""\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nemo-gym"
version = {json.dumps(version)}
dependencies = {array(dependencies)}

[project.optional-dependencies]
dev = {array(dev)}

[tool.setuptools]
packages = []
""",
        encoding="utf-8",
    )


def _checkout_nemo_gym_version(gym_root: Path) -> str:
    """Version this Gym checkout builds, from ``package_info.py``."""
    info = gym_root / "nemo_gym" / "package_info.py"
    if not info.is_file():
        raise RuntimeError(f"{gym_root} has no nemo_gym/package_info.py")
    text = info.read_text(encoding="utf-8")

    def grab(name: str) -> str:
        match = re.search(rf"^{name} = (.+)$", text, re.M)
        if not match:
            raise RuntimeError(f"{info} has no {name}")
        value = ast.literal_eval(match.group(1))
        return "" if value is None else str(value)

    return f"{grab('MAJOR')}.{grab('MINOR')}.{grab('PATCH')}{grab('PRE_RELEASE')}"


def _build_gym_fork_wheel(gym_root: Path, dest: Path, expect_version: str | None) -> Path:
    """Build a metadata-only nemo-gym wheel so resolution uses this checkout's dependencies.

    The file list of a real Gym wheel is every built-in server. Nothing in the uploaded
    package installs that wheel; the image's Gym is the one the sandbox imports.
    """
    actual = _checkout_nemo_gym_version(gym_root)
    if expect_version is not None and actual != expect_version:
        raise RuntimeError(
            f"--gym-root builds nemo-gym {actual} but the image reports {expect_version}. "
            "Gym pins each per-server venv to the image's version, so this wheel would be "
            "ignored and uv would resolve from an index instead."
        )
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-meta-") as tmp:
        project = Path(tmp) / "nemo-gym"
        _write_nemo_gym_metadata_project(gym_root, project, actual)
        _run_build_step(["uv", "build", "--wheel", "--no-config", "--out-dir", str(dest), str(project)])
    built = sorted(dest.glob("nemo_gym-*.whl"))
    if len(built) != 1:
        raise RuntimeError(f"expected exactly one nemo_gym wheel, got {[p.name for p in built]}")
    version = built[0].name.split("-")[1]
    if expect_version is not None and version != expect_version:
        raise RuntimeError(
            f"--gym-root builds nemo-gym {version} but the image reports {expect_version}. "
            "Gym pins each per-server venv to the image's version, so this wheel would be "
            "ignored and uv would resolve from an index instead."
        )
    return built[0]


def _agent_closure_requirements(spec: ConvertEnvironmentSpec, *, work_dir: Path) -> list[str]:
    """Requirements for the agent venv Gym builds, on top of the environment's own."""
    gym_root, ray_version, openai_version = _resolve_image_pins(spec)

    requirements = list(GYM_VENV_REQUIREMENTS)
    if gym_root is not None:
        # The [dev] extra, not the bare wheel: the image's verifiers_agent installs
        # `-e nemo-gym[dev] @ ../../`, so the closure has to cover that extra too.
        wheel = _build_gym_fork_wheel(gym_root, work_dir / "gym", spec.nemo_gym_version)
        requirements.append(f"nemo-gym[dev] @ file://{wheel}")
    if ray_version:
        requirements.append(f"ray[default]=={ray_version}")
    if openai_version:
        requirements.append(f"openai=={openai_version}")

    incomplete = [
        name
        for name, supplied in (
            ("--gym-root", gym_root is not None),
            ("--ray-version", bool(ray_version)),
            ("--openai-version", bool(openai_version)),
        )
        if not supplied
    ]
    if incomplete:
        logger.warning(
            "Building without %s. Gym pins each per-server venv to the versions the training "
            "image runs, so without them the package needs sandbox egress to start. Pass "
            "--nemo-rl-root pointing at a NeMo-RL checkout (with submodules) at the commit "
            "NEMO_RL_REF pins, and all of them are derived from it.",
            ", ".join(incomplete),
        )
    return requirements


def _vendor_missing_agent_wheels(wheels_dir: Path, spec: ConvertEnvironmentSpec, *, work_dir: Path) -> None:
    """Top a pre-vendored closure up to what Gym's per-server venv install needs.

    Resolved from the same requirements as the download path rather than from bare
    distribution names: unpinned, setuptools resolves past the pkg_resources ceiling and ray
    and openai land on versions the training image does not run, which fails offline exactly
    like the omission did. A closure that already carries all of them is left alone, so a
    complete pre-vendored directory still needs no network.
    """
    provided = {canonicalize_name(str(parse_wheel_filename(whl.name)[0])) for whl in wheels_dir.glob("*.whl")}
    missing = [name for name in SANDBOX_VENV_DISTRIBUTIONS if canonicalize_name(name) not in provided]
    if not missing:
        return
    logger.info("--wheels-dir omits %s; resolving the agent closure to supply them", ", ".join(missing))
    # No extra index: these come from PyPI, and uv gives --extra-index-url priority for
    # every package it resolves.
    requirements = _agent_closure_requirements(spec, work_dir=work_dir)
    pinned = _compile_pinned_requirements(work_dir / "agent", requirements, nemo_rl_root=spec.nemo_rl_root)
    _run_pip_download(wheels_dir, requirements_file=pinned)
    # omegaconf's antlr4-python3-runtime publishes no wheel, so the closure is incomplete
    # until the source artifacts are built -- same as the download path.
    _build_downloaded_sdists(wheels_dir)
    _assert_complete_wheel_closure(wheels_dir, pinned)


def download_hub_wheels(
    spec: ConvertEnvironmentSpec,
    *,
    work_dir: Path,
) -> Path:
    """Copy ``spec.wheels_dir`` or download the hub env + verifiers closure."""
    wheels_dir = work_dir / "wheels"
    if spec.wheels_dir is not None:
        src_wheels = sorted(spec.wheels_dir.glob("*.whl"))
        if not src_wheels:
            raise ValueError(
                f"--wheels-dir {spec.wheels_dir} has no *.whl files; adapter-wheels-v1 requires a wheel closure"
            )
        wheels_dir.mkdir(parents=True, exist_ok=True)
        for whl in src_wheels:
            shutil.copy2(whl, wheels_dir / whl.name)
        _vendor_missing_agent_wheels(wheels_dir, spec, work_dir=work_dir)
        assert_wheels_target_platform(wheels_dir)
        _drop_image_gym_wheel(wheels_dir)
        return wheels_dir

    package_name = hub_id_to_package_name(spec.hub_id)
    hub_requirement = f"{package_name}=={spec.hub_version}" if spec.hub_version else package_name
    # One resolve for the whole venv, not one per part. Resolving the environment alone is what
    # produced a closure pinning openai 3.14.1 while Gym installs openai==2.6.1: complete on its
    # own terms, unsatisfiable against the venv it has to build.
    packages = [
        spec.verifiers_spec,
        hub_requirement,
        *spec.extra_wheels,
        *_agent_closure_requirements(spec, work_dir=work_dir),
    ]
    pinned = _compile_pinned_requirements(
        work_dir, packages, extra_index_url=PRIME_HUB_SIMPLE_INDEX, nemo_rl_root=spec.nemo_rl_root
    )
    _run_pip_download(wheels_dir, requirements_file=pinned, extra_index_url=PRIME_HUB_SIMPLE_INDEX)
    _build_downloaded_sdists(wheels_dir)
    downloaded = sorted(wheels_dir.glob("*.whl"))
    if not downloaded:
        raise RuntimeError(
            f"pip download produced no wheels for {packages!r}; adapter-wheels-v1 requires a wheel closure"
        )
    _assert_complete_wheel_closure(wheels_dir, pinned)
    assert_wheels_target_platform(wheels_dir)
    _drop_image_gym_wheel(wheels_dir)
    return wheels_dir


def _wheel_version(path: Path) -> Version:
    """Parse the version segment of a wheel filename (``name-version-...whl``)."""
    parts = path.name.split("-")
    try:
        return Version(parts[1]) if len(parts) > 1 else Version("0")
    except InvalidVersion:
        return Version("0")


def _hub_package_wheel(wheels_dir: Path, package_name: str) -> Path:
    """Pick the hub env wheel the cluster will install (highest parsed version)."""
    candidates = sorted(wheels_dir.glob(f"{package_name}-*.whl"))
    if not candidates:
        dashed = package_name.replace("_", "-")
        candidates = sorted(wheels_dir.glob(f"{dashed}-*.whl"))
    if not candidates:
        raise RuntimeError(f"No wheel for hub package {package_name!r} under {wheels_dir}; cannot load dataset")
    return max(candidates, key=_wheel_version)


def _sandbox_hub_dataset_path(filename: str = HUB_ENV_DATASET_FILENAME) -> str:
    return f"{SANDBOX_DATASET_PATH.rstrip('/')}/{filename}"


@contextmanager
def _isolated_hf_home(hf_home: Path) -> Iterator[Path]:
    """Point Hugging Face cache settings at ``hf_home`` for the duration of the block.

    Conversion must not copy the developer's ``~/.cache/huggingface``; loaders
    populate this temp home, which is then snapshotted into the dataset FileSet.
    Hub and datasets libraries cache these settings at import, so update any
    already-imported constants as well as the process environment.
    """
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HF_DATASETS_CACHE": hf_home / "datasets",
    }
    previous_env = {name: os.environ.get(name) for name in cache_paths}
    for name, path in cache_paths.items():
        os.environ[name] = str(path)

    cached_constants: list[tuple[ModuleType, str, object]] = []
    for module_name, names in (
        ("huggingface_hub.constants", ("HF_HOME", "HF_HUB_CACHE")),
        ("datasets.config", ("HF_DATASETS_CACHE",)),
    ):
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            continue
        for name in names:
            if not hasattr(module, name):
                continue
            previous = getattr(module, name)
            cached_constants.append((module, name, previous))
            path = cache_paths[name]
            setattr(module, name, Path(path) if isinstance(previous, Path) else str(path))
    try:
        yield hf_home
    finally:
        for module, name, previous in reversed(cached_constants):
            setattr(module, name, previous)
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def snapshot_huggingface_home(hf_home: Path, dataset_dir: Path) -> Path | None:
    """Copy the isolated Hugging Face home into ``dataset_dir/.huggingface``."""
    if not hf_home.is_dir() or not any(hf_home.iterdir()):
        logger.warning(
            "isolated HF_HOME at %s is empty after load_environment; "
            "not writing %s (parquet dataset_path remains the primary contract)",
            hf_home,
            HUB_ENV_HF_CACHE_DIRNAME,
        )
        return None
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest = dataset_dir / HUB_ENV_HF_CACHE_DIRNAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(hf_home, dest)
    logger.info("Snapshotted isolated Hugging Face cache to %s", dest)
    return dest


def vendor_hub_environment_dataset(env: Any, dataset_dir: Path) -> Path:
    """Write the hub env's Hugging Face dataset into the dataset FileSet as parquet."""
    dataset = getattr(env, "dataset", None)
    if dataset is None:
        raise RuntimeError(
            "hub environment has no .dataset to vendor into the dataset FileSet; "
            "the Gym sandbox would otherwise call Hugging Face at load_environment()"
        )
    if not hasattr(dataset, "to_parquet"):
        raise RuntimeError(f"hub environment dataset type {type(dataset)!r} cannot be written as parquet")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest = dataset_dir / HUB_ENV_DATASET_FILENAME
    dataset.to_parquet(str(dest))
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"failed to vendor hub environment dataset to {dest}")
    logger.info("Vendored hub environment dataset to %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _install_hub_package_from_wheels(wheels_dir: Path, package_name: str) -> None:
    """Install only the hub env wheel so ``load_environment`` can import it.

    ``--no-deps`` because ``verifiers`` comes from the conversion extra, and the
    download dir can still contain sdists (e.g. ``verifiers-*.zip``) that pip would
    otherwise try to build, needing hatchling and failing ``--no-index``.
    """
    whl = _hub_package_wheel(wheels_dir, package_name)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        str(whl),
    ]
    logger.warning(
        "Installing untrusted hub package %s into the active interpreter at %s. This "
        "mutates that environment (it will no longer match uv.lock) and the package stays "
        "importable afterwards. Run pi-to-gym-conversion from a dedicated conversion "
        "environment rather than the repo .venv if that matters.",
        whl.name,
        sys.executable,
    )
    logger.info("Installing hub package for dataset load: %s", " ".join(cmd))
    _run_build_step(cmd)


def _load_verifiers_environment(vf_env_id: str, vf_env_args: dict[str, Any]) -> Any:
    try:
        import verifiers as vf  # ty: ignore[unresolved-import]  # conversion extra, absent from uv.lock
    except ImportError as exc:
        raise RuntimeError(
            "verifiers is required for pi-to-gym-conversion dataset generation; it lives in "
            "the optional `conversion` extra. Sync it into a dedicated environment, not the "
            "repo .venv, which every `flox activate` prunes back to uv.lock: "
            "`UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl "
            "--extra conversion`, then run `.venv-conversion/bin/pi-to-gym-conversion`. "
            "Do not use the project .venv: this installs into sys.executable, and a mutated "
            "project environment makes the next flox activate fail"
        ) from exc

    return vf.load_environment(vf_env_id, **vf_env_args)


def _dataset_rows_from_env(
    env: Any,
    vf_env_id: str,
    *,
    size: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    try:
        dataset = env.get_dataset(n=size, seed=seed)
    except ValueError:
        dataset = env.get_eval_dataset(n=size, seed=seed)

    rows: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        prompt = dataset["prompt"][i]
        example_id = dataset["example_id"][i] if "example_id" in dataset.column_names else i
        answer = dataset["answer"][i] if "answer" in dataset.column_names else ""
        task = dataset["task"][i] if "task" in dataset.column_names else vf_env_id
        info = dataset["info"][i] if "info" in dataset.column_names else {}
        rows.append(
            dataset_row_from_verifiers(
                idx=i,
                prompt=prompt,
                vf_env_id=vf_env_id,
                example_id=example_id,
                answer=answer,
                task=task,
                info=info if isinstance(info, dict) else {},
            )
        )
    return rows


def _load_verifiers_dataset_rows(
    vf_env_id: str,
    vf_env_args: dict[str, Any],
    *,
    size: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    env = _load_verifiers_environment(vf_env_id, vf_env_args)
    return _dataset_rows_from_env(env, vf_env_id, size=size, seed=seed)


def split_train_validation(
    rows: list[dict[str, Any]],
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Split rows into (train, validation). Raises if the split would empty training."""
    if not rows or validation_fraction <= 0:
        return rows, None
    split = max(1, int(len(rows) * validation_fraction))
    if split >= len(rows):
        raise ValueError(
            f"validation_fraction={validation_fraction} leaves no training rows "
            f"({len(rows)} row(s) available, {split} would go to validation). "
            "Use a smaller fraction or generate more rows."
        )
    return rows[split:], rows[:split]


def convert_prime_environment(spec: ConvertEnvironmentSpec) -> ConvertedPackage:
    """Convert a Prime Intellect hub environment to adapter-wheels-v1 + Gym JSONL."""
    vf_env_id = spec.vf_env_id or hub_id_to_vf_env_id(spec.hub_id)
    host_vf_env_args = dict(spec.vf_env_args or {})
    package_vf_env_args = dict(host_vf_env_args)
    env_out = spec.out_dir
    dataset_out = spec.dataset_dir or (spec.out_dir.parent / f"{env_out.name}-dataset")
    package_name = hub_id_to_package_name(spec.hub_id)

    with tempfile.TemporaryDirectory(prefix="nhx-rl-convert-") as tmp:
        wheels_dir = download_hub_wheels(spec, work_dir=Path(tmp))

        if spec.dataset_size == 0:  # package layout only
            all_rows: list[dict[str, Any]] = []
        else:
            hf_home = Path(tmp) / "hf-home"
            with _isolated_hf_home(hf_home):
                _install_hub_package_from_wheels(wheels_dir, package_name)
                env = _load_verifiers_environment(vf_env_id, host_vf_env_args)
                all_rows = _dataset_rows_from_env(
                    env,
                    vf_env_id,
                    size=spec.dataset_size,
                    seed=spec.dataset_seed,
                )
                if not all_rows:
                    raise RuntimeError(
                        f"Dataset generation for {spec.hub_id!r} (vf_env_id={vf_env_id!r}) "
                        f"returned 0 rows (dataset_size={spec.dataset_size})"
                    )
                vendor_hub_environment_dataset(env, dataset_out)
                snapshot_huggingface_home(hf_home, dataset_out)
                package_vf_env_args[DATASET_PATH_ARG] = _sandbox_hub_dataset_path()

        manifest = write_adapter_wheels_package(
            out_dir=env_out,
            hub_id=spec.hub_id,
            vf_env_id=vf_env_id,
            vf_env_args=package_vf_env_args,
            adapter_agent=spec.adapter_agent,
            wheels_src=wheels_dir,
        )

    train_rows, val_rows = split_train_validation(all_rows, spec.validation_fraction)

    if train_rows:
        validate_dataset_rows(
            train_rows,
            expected_vf_env_id=vf_env_id,
            expected_agent=spec.adapter_agent,
        )

    train_path, val_path = write_dataset_jsonl(
        dataset_dir=dataset_out,
        rows=train_rows,
        validation_rows=val_rows,
    )

    return ConvertedPackage(
        environment_root=env_out,
        dataset_dir=dataset_out,
        manifest=manifest,
        training_jsonl=train_path,
        validation_jsonl=val_path,
    )
