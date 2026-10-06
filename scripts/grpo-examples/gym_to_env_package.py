#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package any NeMo Gym server directory as a `wheels-v1` or `native-v1` environment FileSet.

This is an **example helper**, not part of the customizer's functionality. Nothing in the
platform calls it: the supported contract is the environment FileSet layout itself, which
`nhx.rl.tasks.environment.validate` defines and which you can satisfy by hand. Use this to
get a working package quickly, or as a starting point for your own build.

Run it with a dedicated interpreter, not ``uv run`` (that is the project ``.venv``).
``wheels-v1`` calls ``sys.executable -m pip``, and installing into the project
environment makes it diverge from ``uv.lock``. ``flox activate`` then fails trying
to recreate ``.venv``, which blocks commits and uploads.

    UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl
    .venv-conversion/bin/python scripts/grpo-examples/gym_to_env_package.py \\
        --gym-root ~/workspace/Gym \\
        --nemo-rl-root ~/workspace/RL \\
        --server resources_servers/math_with_judge \\
        --format wheels-v1 --arch x86_64 --out-dir /tmp/mwj-env

Then validate and upload, still with that interpreter:

    .venv-conversion/bin/pi-to-gym-conversion --validate-only /tmp/mwj-env
    nemo files filesets create my-env -w default --purpose environment --exist-ok
    nemo files upload /tmp/mwj-env/ my-env -w default

`native-v1` ships no wheels and resolves the server's requirements from a package index when
the job starts, so the cluster needs egress. `wheels-v1` vendors the closure and does not.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename

GYM_REPO = "https://github.com/NVIDIA-NeMo/Gym"

# Gym 0.7 requires Python>=3.13.14. A bare "3.13" resolves as 3.13.0 and rejects
# that wheel. The training image is CPython 3.13.15, so 3.13.14 is a compatible target.
TARGET_PYTHON_VERSION = "3.13.14"
# The training images are published for both linux/amd64 and linux/arm64, so wheel
# architecture is a property of the cluster's nodes. Several glibc floors are listed per arch
# because pip matches these tags literally rather than expanding a compatibility range.
WHEEL_ARCHES = ("x86_64", "aarch64")

SERVER_TYPES = ("resources_servers", "responses_api_agents", "responses_api_models")

# setuptools 81 removed pkg_resources, which Gym's pinned hydra imports at import time.
SETUPTOOLS_PKG_RESOURCES_CEILING = "81"
HYDRA_CORE_SPEC = ">=1.3,<1.4"
OMEGACONF_SPEC = ">=2.2,<2.4"
REQUIRED_WHEEL_SPECS = {
    canonicalize_name("hydra-core"): SpecifierSet(HYDRA_CORE_SPEC),
    canonicalize_name("omegaconf"): SpecifierSet(OMEGACONF_SPEC),
}

# validate_package_layout rejects any *.jsonl anywhere under the package, and prompts ship as
# their own dataset FileSet -- so a server's bundled data never travels with it.
SKIP_DIRS = {"data", "tests", "__pycache__", ".venv", ".pytest_cache"}


def resolve_gym_root(raw: str | None) -> Path:
    """Require an explicit Gym checkout and fail with instructions rather than a stack trace."""
    if not raw:
        raise SystemExit(
            "--gym-root is required.\n\n"
            "This script packages a server out of a NeMo Gym source tree, and Gym is not vendored\n"
            "in nemo-helix. Clone it first, then point --gym-root at the checkout:\n\n"
            f"    git clone {GYM_REPO} ~/workspace/Gym\n"
            "    --gym-root ~/workspace/Gym\n"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--gym-root does not exist: {root}")
    if not any((root / t).is_dir() for t in SERVER_TYPES):
        raise SystemExit(
            f"{root} does not look like a NeMo Gym checkout -- expected at least one of "
            f"{', '.join(SERVER_TYPES)}/ inside it.\n"
            f"Clone it with: git clone {GYM_REPO} ~/workspace/Gym"
        )
    return root


def resolve_server(gym_root: Path, server: str) -> Path:
    """Validate ``<server_type>/<implementation>`` and return it as a relative path."""
    rel = Path(server.strip("/"))
    if len(rel.parts) != 2 or rel.parts[0] not in SERVER_TYPES:
        raise SystemExit(
            f"--server must be '<server_type>/<implementation>', with server_type one of "
            f"{', '.join(SERVER_TYPES)}. Got: {server!r}"
        )
    source = gym_root / rel
    if not (source / "app.py").is_file():
        raise SystemExit(f"no app.py under {source} -- is that the right --server?")
    # Gym only treats a directory as a server if it ships an install marker; without one it
    # silently falls back to a built-in of the same name in the image.
    if not ((source / "requirements.txt").is_file() or (source / "pyproject.toml").is_file()):
        raise SystemExit(
            f"{source} has neither requirements.txt nor pyproject.toml. Gym would not recognise "
            "it as a server, and would silently run a built-in of the same name instead."
        )
    return rel


def copy_server(gym_root: Path, out_dir: Path, rel: Path) -> Path:
    """Copy the server tree, dropping bundled data, tests and any stray JSONL."""
    target = out_dir / rel
    shutil.copytree(
        gym_root / rel,
        target,
        dirs_exist_ok=True,
        ignore=lambda d, names: [n for n in names if n in SKIP_DIRS or n.endswith(".jsonl")],
    )
    return target


def copy_server_configs(gym_root: Path, out_dir: Path, rel: Path, chosen: list[str] | None) -> Path:
    """Copy the selected configs alone, leaving the implementation to the training image.

    Gym resolves a server directory by name against the package's search root first and falls
    back to the built-in of the same name when the package ships no install marker. So a
    package carrying configs alone runs the image's own code, at the version the image was
    built with, which is what a native-v1 package wants whenever the server is one the image
    already provides: nothing can drift between the two copies because there is only one.
    """
    target = out_dir / rel / "configs"
    target.mkdir(parents=True, exist_ok=True)
    for cfg in select_configs(gym_root / rel, rel.parts[1], chosen):
        shutil.copy2(cfg, target / cfg.name)
    return out_dir / rel


def select_configs(server_dir: Path, impl: str, chosen: list[str] | None) -> list[Path]:
    """Select the configs to load -- never all of them.

    A Gym server directory often ships several configs pairing it with different agents
    (math_with_judge alone, or with hermes/opencode/openclaw agents). Loading every one would
    start servers whose implementations this package does not carry. Gym's convention is that
    ``<impl>.yaml`` is the plain pairing, so that is the default; anything else is explicit.
    """
    available = sorted((server_dir / "configs").glob("*.yaml"))
    if not available:
        raise SystemExit(
            f"no configs/*.yaml under {server_dir}. An environment with no config starts no "
            "servers, so at least one is required."
        )
    names = {c.name: c for c in available}
    if chosen:
        missing = [c for c in chosen if c not in names]
        if missing:
            raise SystemExit(f"--config not found: {', '.join(missing)}. Available: {', '.join(sorted(names))}")
        return [names[c] for c in chosen]
    if f"{impl}.yaml" in names:
        return [names[f"{impl}.yaml"]]
    raise SystemExit(
        f"no default config: expected {impl}.yaml under {server_dir}/configs. "
        f"Pass --config explicitly. Available: {', '.join(sorted(names))}"
    )


def strip_inline_datasets(pkg_server_dir: Path) -> list[str]:
    """Drop ``datasets:`` blocks, which point at in-tree JSONL the package cannot carry."""
    touched = []
    for cfg in sorted((pkg_server_dir / "configs").glob("*.yaml")):
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        changed = False
        for instance in doc.values():
            if not isinstance(instance, dict):
                continue
            for server_type, impls in instance.items():
                if server_type not in SERVER_TYPES or not isinstance(impls, dict):
                    continue
                for impl in impls.values():
                    if isinstance(impl, dict) and impl.pop("datasets", None) is not None:
                        changed = True
        if changed:
            cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            touched.append(cfg.name)
    return touched


def policy_model_relpath(fmt: str, server: Path) -> Path:
    """Where ``policy_model.yaml`` is written.

    ``wheels-v1`` allows ``configs/`` at the package root. ``native-v1`` only allows a Gym
    server tree, so the file sits beside the server configs already being packaged.
    """
    if fmt == "wheels-v1":
        return Path("configs") / "policy_model.yaml"
    return server / "configs" / "policy_model.yaml"


def write_policy_model_config(out_dir: Path, fmt: str, server: Path) -> Path:
    """Ship the ``policy_model`` server every config's refs resolve against.

    Reuses the platform's own builder so a hand-built package and pi-to-gym-conversion cannot
    drift on the interpolations, which resolve against the config NeMo-RL injects at spin-up.
    """
    from nhx.rl.tasks.environment.package import build_policy_model_yaml

    target = out_dir / policy_model_relpath(fmt, server)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(build_policy_model_yaml(), sort_keys=False), encoding="utf-8")
    return target


_GYM_SERVER_TREES = (
    "resources_servers",
    "responses_api_agents",
    "responses_api_models",
    "benchmarks",
    "environments",
    "environment_servers",
)


def server_requirement_lines(pkg_server_dir: Path) -> list[str]:
    """The server's own requirements, without the editable Gym checkout line."""
    reqs = pkg_server_dir / "requirements.txt"
    if not reqs.is_file():
        return []
    lines = []
    for raw in reqs.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "../.." in stripped or "nemo-gym" in stripped:
            continue
        lines.append(stripped)
    return lines


def server_closure_requirements(pkg_server_dir: Path, ray_version: str, openai_version: str) -> list[str]:
    """Pins Gym stamps onto the selected server's venv, plus that server's own requirements."""
    return [
        f"ray[default]=={ray_version}",
        f"openai=={openai_version}",
        f"hydra-core{HYDRA_CORE_SPEC}",
        f"omegaconf{OMEGACONF_SPEC}",
        "pip",
        f"setuptools>=61,<{SETUPTOOLS_PKG_RESOURCES_CEILING}",
        "setuptools-scm",
        *server_requirement_lines(pkg_server_dir),
    ]


_NO_WHEEL_RE = re.compile(r"No matching distribution found for (\S+)")


def _build_pure_wheel(requirement: str, wheels: Path) -> None:
    """Build a wheel for a pin that publishes none, e.g. antlr4-python3-runtime.

    Only pure-Python projects are safe to build here: the build host is not the target, so
    anything compiling an extension would produce a wheel for the wrong platform.
    """
    before = set(wheels.glob("*.whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "--wheel-dir", str(wheels), requirement],
        check=True,
    )
    for built in set(wheels.glob("*.whl")) - before:
        if not built.stem.endswith("-py3-none-any") and not built.stem.endswith("-py2.py3-none-any"):
            built.unlink()
            raise SystemExit(
                f"{requirement} has no wheel on the index and does not build a pure-Python one "
                f"({built.name}). Building it here would target the build host, not the cluster."
            )
        print(f"Built from sdist: {built.name}", flush=True)


def _download_with_sdist_fallback(download_cmd: list[str], wheels: Path, max_builds: int = 8) -> None:
    """Run pip download, building any pin that publishes no wheel, then retrying."""
    builds = 0
    while True:
        result = subprocess.run(download_cmd, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return
        sys.stderr.write(result.stderr)
        match = _NO_WHEEL_RE.search(result.stderr)
        if not match:
            raise SystemExit("pip download failed; see the error above.")
        if builds >= max_builds:
            raise SystemExit(f"still missing wheels after building {max_builds} package(s).")
        _build_pure_wheel(match.group(1), wheels)
        builds += 1


def gym_wheel_ships_server_trees(wheel: Path) -> bool:
    with zipfile.ZipFile(wheel) as archive:
        return any(name.split("/", 1)[0] in _GYM_SERVER_TREES for name in archive.namelist())


def drop_image_gym_wheel(wheels: Path) -> None:
    for wheel in wheels.glob("nemo_gym-*.whl"):
        if gym_wheel_ships_server_trees(wheel):
            wheel.unlink()


def write_library_gym_project(gym_root: Path, dest: Path, version: str) -> None:
    project = tomllib.loads((gym_root / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    dependencies = project.get("dependencies", [])
    dev = project.get("optional-dependencies", {}).get("dev", [])
    source = gym_root / "nemo_gym"
    if not source.is_dir():
        raise SystemExit(f"{gym_root} has no nemo_gym package")

    def array(items: list[str]) -> str:
        return "[\n" + "".join(f"  {json.dumps(item)},\n" for item in items) + "]"

    dest.mkdir(parents=True)
    shutil.copytree(
        source,
        dest / "nemo_gym",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
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

[tool.setuptools.packages.find]
where = ["."]
include = ["nemo_gym", "nemo_gym.*"]
""",
        encoding="utf-8",
    )


def build_library_gym_wheel(gym_root: Path, dest: Path) -> Path:
    version = nemo_gym_version(gym_root)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-lib-") as tmp:
        project = Path(tmp) / "nemo-gym"
        write_library_gym_project(gym_root, project, version)
        subprocess.run(
            ["uv", "build", "--wheel", "--no-config", "--out-dir", str(dest), str(project)],
            check=True,
        )
    built = sorted(dest.glob("nemo_gym-*.whl"))
    if len(built) != 1:
        names = ", ".join(path.name for path in built) or "none"
        raise SystemExit(f"expected one nemo-gym library wheel in {dest}, got {names}")
    if gym_wheel_ships_server_trees(built[0]):
        raise SystemExit(f"{built[0].name} contains a Gym server tree")
    return built[0]


def validate_required_wheel_versions(wheels: Path) -> None:
    """Reject a wheelhouse whose resolver backtracked to pre-1.3 Hydra."""
    found: dict[str, set] = {name: set() for name in REQUIRED_WHEEL_SPECS}
    for wheel in wheels.glob("*.whl"):
        try:
            name, version, _, _ = parse_wheel_filename(wheel.name)
        except (InvalidWheelFilename, ValueError):
            continue
        normalized = canonicalize_name(name)
        if normalized in found:
            found[normalized].add(version)

    problems = []
    for name, specifier in REQUIRED_WHEEL_SPECS.items():
        versions = found[name]
        if not versions:
            problems.append(f"missing {name}{specifier}")
            continue
        incompatible = sorted(str(version) for version in versions if version not in specifier)
        if incompatible:
            problems.append(f"{name} has incompatible wheel(s) {', '.join(incompatible)}; expected {specifier}")
    if problems:
        raise SystemExit("invalid wheels-v1 dependency closure: " + "; ".join(problems))


def rl_dependency_policy(nemo_rl_root: Path) -> tuple[list[str], list[str]]:
    """Read the uv limits the Gym host applies when it installs the wheelhouse.

    The host starts in the NeMo-RL tree, so ``uv pip install`` honors that checkout's
    ``[tool.uv]`` override-dependencies and constraint-dependencies. A closure resolved
    without them installs versions the host then rejects.
    """
    pyproject = nemo_rl_root / "pyproject.toml"
    if not pyproject.is_file():
        raise SystemExit(
            f"--nemo-rl-root has no pyproject.toml: {nemo_rl_root}\n"
            "Point it at the NeMo-RL checkout NEMO_RL_REF pins. The training image is built "
            "from that commit, and the Gym host installs this wheelhouse under its uv policy."
        )
    uv_config = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    missing = [key for key in ("override-dependencies", "constraint-dependencies") if not uv_config.get(key)]
    if missing:
        raise SystemExit(
            f"{pyproject} has no [tool.uv] {', '.join(missing)}. Those tables are what the Gym "
            "host applies during the offline install."
        )
    return (
        _keep_pkg_resources_setuptools(list(uv_config["override-dependencies"])),
        list(uv_config["constraint-dependencies"]),
    )


def _keep_pkg_resources_setuptools(overrides: list[str]) -> list[str]:
    """Intersect a setuptools override with Gym's ``setuptools<81`` server-venv dep."""
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


def nemo_gym_version_text(text: str) -> str:
    """Read ``__version__`` from Gym's ``package_info.py`` without importing it."""

    def grab(name: str) -> str:
        match = re.search(rf"^{name} = (.+)$", text, re.M)
        if not match:
            raise SystemExit(f"package_info.py has no {name}")
        value = ast.literal_eval(match.group(1))
        return "" if value is None else str(value)

    return f"{grab('MAJOR')}.{grab('MINOR')}.{grab('PATCH')}{grab('PRE_RELEASE')}"


def nemo_gym_version(gym_root: Path) -> str:
    """Version this Gym checkout builds, from its ``package_info.py``."""
    info = gym_root / "nemo_gym" / "package_info.py"
    if not info.is_file():
        raise SystemExit(f"{gym_root} has no nemo_gym/package_info.py")
    return nemo_gym_version_text(info.read_text(encoding="utf-8"))


def locked_distribution_versions(rl_root: Path) -> tuple[str, str]:
    """Ray and openai versions the image's nemo-gym environment installs.

    ``uv.lock`` can carry more than one openai (sglang pins another). The version on the
    nemo-gym package is the one Gym stamps into every per-server venv. Ray has a single
    locked version, and nemo-gym does not repeat it on the dependency edge.
    """
    packages = tomllib.loads((rl_root / "uv.lock").read_text(encoding="utf-8")).get("package", [])
    nemo_gym = next((pkg for pkg in packages if pkg.get("name") == "nemo-gym"), None)
    if nemo_gym is None:
        raise SystemExit(f"{rl_root / 'uv.lock'} has no nemo-gym package")

    def dep_version(name: str) -> str | None:
        versions = {
            dep["version"] for dep in nemo_gym.get("dependencies", []) if dep.get("name") == name and "version" in dep
        }
        if len(versions) == 1:
            return versions.pop()
        return None

    def unique_version(name: str) -> str | None:
        versions = {pkg["version"] for pkg in packages if pkg.get("name") == name and "version" in pkg}
        if len(versions) == 1:
            return versions.pop()
        return None

    ray = dep_version("ray") or unique_version("ray")
    openai = dep_version("openai") or unique_version("openai")
    missing = [name for name, value in (("ray", ray), ("openai", openai)) if not value]
    if missing:
        raise SystemExit(f"{rl_root / 'uv.lock'} does not pin a single {' and '.join(missing)} version for nemo-gym")
    return ray, openai


def vendor_wheels(
    out_dir: Path,
    pkg_server_dir: Path,
    arch: str,
    ray_version: str,
    openai_version: str,
    nemo_rl_root: Path,
    gym_root: Path,
) -> Path:
    """Resolve and download the offline closure for the selected server.

    Resolution and download are separate steps. ``uv pip compile --python-platform`` picks
    the versions, because environment markers are evaluated during resolution: resolving on
    the build host omits a linux-only dependency (sqlalchemy's greenlet) and can add a
    darwin-only one. ``pip download`` then fetches exactly those pins for the target tags.

    The closure is that server's ``requirements.txt``, the pins Gym adds to every
    sub-venv, and ``nemo-gym[dev]`` from the ``nemo_gym`` library wheel. setuptools is
    capped below 81, the release that removed ``pkg_resources``: Gym pins hydra 1.3,
    which imports it at import time.
    """
    wheels = out_dir / "wheels"
    # Rebuild from empty: pip copies by filename, so a wheel from an earlier run survives
    # whenever the new resolution picks a different version.
    if wheels.exists():
        shutil.rmtree(wheels)
    wheels.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-wheel-") as built:
        gym_wheel = build_library_gym_wheel(gym_root, Path(built))
        staged = wheels / gym_wheel.name
        shutil.copy2(gym_wheel, staged)
    requirements = [
        f"nemo-gym[dev] @ file://{staged}",
        *server_closure_requirements(pkg_server_dir, ray_version, openai_version),
    ]

    overrides, constraints = rl_dependency_policy(nemo_rl_root)
    with tempfile.TemporaryDirectory(prefix="closure-") as tmp:
        req_in = Path(tmp) / "requirements.in"
        req_in.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        override = Path(tmp) / "override.txt"
        constraint = Path(tmp) / "constraint.txt"
        override.write_text("\n".join(overrides) + "\n", encoding="utf-8")
        constraint.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        pinned = Path(tmp) / "requirements.txt"
        # --no-config ignores this repo's [tool.uv]. The host applies the RL checkout's,
        # passed as the override and constraint files.
        compile_cmd = [
            "uv",
            "pip",
            "compile",
            str(req_in),
            "--output-file",
            str(pinned),
            "--no-header",
            "--no-config",
            "--override",
            str(override),
            "--constraint",
            str(constraint),
            "--python-platform",
            f"{arch}-unknown-linux-gnu",
            "--python-version",
            TARGET_PYTHON_VERSION,
        ]
        print("Running:", " ".join(compile_cmd), flush=True)
        subprocess.run(compile_cmd, check=True)

        download_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheels),
            "--no-cache-dir",
            "--only-binary",
            ":all:",
            # Pick up anything built from an sdist below.
            "--find-links",
            str(wheels),
            "--python-version",
            TARGET_PYTHON_VERSION,
            # pip defaults this to the build host's interpreter; the target is CPython
            # regardless of what runs this script. The ABI set is left to pip, which
            # derives it from the implementation and version.
            "--implementation",
            "cp",
        ]
        for tag in (
            f"manylinux_2_39_{arch}",
            f"manylinux_2_28_{arch}",
            f"manylinux_2_17_{arch}",
            f"manylinux2014_{arch}",
        ):
            download_cmd += ["--platform", tag]
        download_cmd += ["-r", str(pinned)]
        print("Running:", " ".join(download_cmd), flush=True)
        _download_with_sdist_fallback(download_cmd, wheels)

    validate_required_wheel_versions(wheels)
    drop_image_gym_wheel(wheels)
    library_wheels = list(wheels.glob("nemo_gym-*.whl"))
    if len(library_wheels) != 1 or gym_wheel_ships_server_trees(library_wheels[0]):
        names = ", ".join(path.name for path in library_wheels) or "none"
        raise SystemExit(f"wheels/ must contain one nemo_gym library wheel, got {names}")
    stray = [f.name for f in wheels.iterdir() if f.is_file() and f.suffix != ".whl"]
    if stray:
        raise SystemExit(f"wheels/ must contain only .whl files, got: {stray}")
    return wheels


def write_manifest(
    out_dir: Path, fmt: str, policy_model: Path, config_paths: list[str], name: str, description: str
) -> Path:
    manifest = out_dir / "nemo-environment.yaml"
    # policy_model first: it defines the server the other configs' refs resolve against.
    entries = "".join(f"  - {p}\n" for p in [policy_model.as_posix(), *config_paths])
    body = f"format: {fmt}\nconfig_paths:\n{entries}metadata:\n  name: {name}\n"
    if description:
        body += f"  description: {description}\n"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gym-root",
        help=f"Local NeMo Gym checkout. Gym is not vendored here: git clone {GYM_REPO}",
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Server to package, as '<server_type>/<implementation>', e.g. resources_servers/math_with_judge",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        help="Config filename to load, repeatable (e.g. math_with_judge.yaml). Defaults to "
        "'<implementation>.yaml'. Other configs in the directory usually pair the server with an "
        "agent whose implementation this package does not carry, so they are not included by default.",
    )
    parser.add_argument(
        "--format",
        choices=("wheels-v1", "native-v1"),
        default="wheels-v1",
        help="wheels-v1 vendors the closure and needs no egress at spin-up. native-v1 ships no "
        "wheels and resolves from a package index, so the cluster must allow internet.",
    )
    parser.add_argument(
        "--arch",
        choices=WHEEL_ARCHES,
        default="x86_64",
        help="Architecture of the nodes that run GRPO training (wheels-v1 only). Check with: "
        "kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}' "
        "-- amd64 maps to x86_64, arm64 to aarch64.",
    )
    parser.add_argument(
        "--nemo-rl-root",
        type=Path,
        help="NeMo-RL checkout the training image is built from (wheels-v1). Its "
        "[tool.uv] limits the wheelhouse, and its uv.lock supplies the ray and openai pins. "
        "The caller fetches the commit NEMO_RL_REF pins.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Ship the selected configs without the server implementation, which then resolves to "
        "the built-in of the same name in the training image. native-v1 only, and only for a "
        "server the image already provides: nothing can drift between package and image because "
        "there is only one copy of the code.",
    )
    parser.add_argument("--name", help="metadata.name. Defaults to the implementation name.")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    if args.reference_only and args.format != "native-v1":
        raise SystemExit(
            f"--reference-only is native-v1 only, got {args.format}. The wheels formats vendor a "
            "closure resolved from the server's own requirements.txt, which a package that ships "
            "no server code does not have."
        )

    gym_root = resolve_gym_root(args.gym_root)
    rel = resolve_server(gym_root, args.server)
    name = args.name or rel.parts[1].replace("_", "-")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_only:
        pkg_server_dir = copy_server_configs(gym_root, args.out_dir, rel, args.config)
    else:
        pkg_server_dir = copy_server(gym_root, args.out_dir, rel)
    stripped = strip_inline_datasets(pkg_server_dir)
    config_paths = [
        c.relative_to(args.out_dir).as_posix() for c in select_configs(pkg_server_dir, rel.parts[1], args.config)
    ]
    policy_relpath = policy_model_relpath(args.format, rel)
    policy_model = write_policy_model_config(args.out_dir, args.format, rel)
    nemo_rl_root = None
    ray_version = openai_version = gym_version = None
    wheels = None
    if args.format == "wheels-v1":
        if args.nemo_rl_root is None:
            raise SystemExit("wheels-v1 needs --nemo-rl-root, the NeMo-RL checkout at the commit NEMO_RL_REF pins.")
        nemo_rl_root = args.nemo_rl_root.expanduser().resolve()
        ray_version, openai_version = locked_distribution_versions(nemo_rl_root)
        gym_version = nemo_gym_version(gym_root)
        print(
            f"image pins: nemo-gym={gym_version} ray={ray_version} openai={openai_version} from {nemo_rl_root}",
            flush=True,
        )
        wheels = vendor_wheels(
            args.out_dir,
            pkg_server_dir,
            args.arch,
            ray_version,
            openai_version,
            nemo_rl_root,
            gym_root,
        )
    manifest = write_manifest(args.out_dir, args.format, policy_relpath, config_paths, name, args.description)

    stray = [str(p.relative_to(args.out_dir)) for p in args.out_dir.rglob("*.jsonl")]
    if stray:
        raise SystemExit(f"validation rejects *.jsonl inside the package, found: {stray}")

    print(
        json.dumps(
            {
                "environment_root": str(args.out_dir),
                "format": args.format,
                "reference_only": args.reference_only,
                "name": name,
                "server": rel.as_posix(),
                "manifest": str(manifest),
                "policy_model_config": str(policy_model),
                "config_paths": [policy_relpath.as_posix(), *config_paths],
                "datasets_blocks_stripped": stripped,
                "arch": args.arch if wheels else None,
                "nemo_rl_root": str(nemo_rl_root) if nemo_rl_root else None,
                "nemo_gym_version": gym_version,
                "ray_version": ray_version,
                "openai_version": openai_version,
                "wheel_count": len(list(wheels.glob("*.whl"))) if wheels else 0,
                "next": [f"{Path(sys.executable).parent / 'pi-to-gym-conversion'} --validate-only {args.out_dir}"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
