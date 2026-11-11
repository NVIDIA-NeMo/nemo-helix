#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory Python packages on the filesystem of an nhx-rl-training container.

--registry and --tag are passed to `docker image inspect` for nhx-rl-training
and nhx-rl-base. They do not select the filesystem that is scanned. Package
files, dpkg, and /tmp/base-dpkg-packages.txt come from the process that runs
this script. The CUDA base image name is read from Dockerfile.nhx-rl-base in
the parent of this script's directory.

Run it from a shell inside the training container. That shell needs the Docker
CLI, a daemon that already has both images at --tag, and this script sitting
next to Dockerfile.nhx-rl-base. Running it on the host records the host's
packages under the inspected image's identity.

    python scan_rl_image_packages.py \\
        --registry nvcr.io/ORG/nemo-platform-dev --tag TAG
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

IMAGE_NAMES = ("nhx-rl-training", "nhx-rl-base")
DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile.nhx-rl-base"

# Paths named by Dockerfile.nhx-rl-base and Dockerfile.nhx-rl-training.
KNOWN_LOCATIONS = [
    {
        "path": "/opt/nemo_rl_venv",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Driver venv (UV_PROJECT_ENVIRONMENT). Training image installs platform glue here and symlinks /opt/venv to it.",
    },
    {
        "path": "/opt/venv",
        "kind": "symlink",
        "dockerfile": "Dockerfile.nhx-rl-training",
        "role": "Symlink to /opt/nemo_rl_venv. Entrypoint is /opt/venv/bin/python.",
    },
    {
        "path": "/opt/ray_venvs",
        "kind": "venv-parent",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "NEMO_RL_VENV_DIR. One prefetched venv per actor.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra automodel. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra mcore --extra nemo_gym. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra vllm --extra nemo_gym. vllm package tree is copied private. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra vllm --extra nemo_gym. vllm package tree is copied private. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra vllm. vllm package tree is copied private. Training image does not add sandboxed_gym here.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra nemo_gym. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/ray_venvs/nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor",
        "kind": "venv",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sync --extra nemo_gym. Training image adds editable sandboxed_gym.",
    },
    {
        "path": "/opt/gym_venvs",
        "kind": "venv-parent",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "NEMO_GYM_VENV_DIR. Populated only when NEMO_GYM_PREFETCH_CONFIGS is set. Default build leaves it empty.",
    },
    {
        "path": "/opt/uv_cache",
        "kind": "uv-cache",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "UV_CACHE_DIR. Extracted downloads live in archive-v0. Venvs symlink into it (UV_LINK_MODE=symlink).",
    },
    {
        "path": "/opt/uv_cache/archive-v0",
        "kind": "uv-cache-archives",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Extracted wheels and sdists. Counted when no venv links to the archive.",
    },
    {
        "path": "/opt/uv_cache/wheels-v7",
        "kind": "uv-cache-wheels",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv wheel cache. Not added again to download_count; archives and venvs are the package records.",
    },
    {
        "path": "/opt/uv_cache/sdists-v10",
        "kind": "uv-cache-sdists",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "uv sdist cache. Built results are in archive-v0, so sdists are not a second download count.",
    },
    {
        "path": "/opt/gym-wheels",
        "kind": "local-wheels",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Local nemo-gym wheel from the gym-wheel stage. Published via UV_FIND_LINKS.",
    },
    {
        "path": "/mnt/wheels",
        "kind": "build-mount",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Build-time rl-cuda-ext-wheel mount. Not present in the published image. Wheels were installed into each Ray venv.",
    },
    {
        "path": "/opt/cpython",
        "kind": "python-prefix",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "CPython copied from python:3.13.16-slim-bookworm, not from the CUDA base image. Interpreter used by every venv.",
    },
    {
        "path": "/opt/nemo-rl",
        "kind": "source-tree",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "NeMo-RL source. Installed editable into the driver and Ray venvs.",
    },
    {
        "path": "/app",
        "kind": "source-tree",
        "dockerfile": "Dockerfile.nhx-rl-training",
        "role": "Platform workspace. Editable glue and sandboxed_gym are installed from here.",
    },
    {
        "path": "/root/.cache/uv",
        "kind": "uv-cache",
        "dockerfile": "Dockerfile.nhx-rl-training",
        "role": "Build-only UV_CACHE_DIR for uv-glue. Cache mounts are not supposed to persist; recorded if anything is present.",
    },
    {
        "path": "/home/ubuntu/.cache/uv",
        "kind": "uv-cache",
        "dockerfile": "Dockerfile.nhx-rl-training",
        "role": "Runtime UV_CACHE_DIR. Empty at build; future uv writes go here instead of /opt/uv_cache.",
    },
    {
        "path": "/usr/share/nemo-helix/source-distributions",
        "kind": "source-archive",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Collected apt sources and Python sdist manifests for nhx-rl-base and nhx-rl-training. Not a second install.",
    },
    {
        "path": "/usr/lib/python3/dist-packages",
        "kind": "system-python",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Debian dist-packages from the CUDA base image, if that image shipped any.",
    },
    {
        "path": "/usr/local/lib",
        "kind": "system-python",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Checked for a system site-packages outside the venvs and /opt/cpython.",
    },
    {
        "path": "/tmp/base-dpkg-packages.txt",
        "kind": "base-image-inventory",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "dpkg name/version snapshot of the CUDA base image, taken before apt-get install/upgrade/purge.",
    },
    {
        "path": "/var/lib/dpkg",
        "kind": "dpkg-database",
        "dockerfile": "Dockerfile.nhx-rl-base",
        "role": "Used to attach Installed-Size to base-image packages that are still present.",
    },
]


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_metadata(text: str):
    name = version = None
    for line in text.splitlines():
        if line == "":
            break
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
    return name, version


def record_size(dist: Path):
    record = dist / "RECORD"
    if not record.is_file():
        return None
    total = 0
    saw = False
    for line in read_text(record).splitlines():
        if not line:
            continue
        parts = line.rsplit(",", 2)
        if len(parts) == 3 and parts[2].isdigit():
            total += int(parts[2])
            saw = True
    return total if saw else 0


def classify(dist: Path):
    direct = dist / "direct_url.json"
    url = None
    if direct.is_file():
        try:
            info = json.loads(read_text(direct))
        except json.JSONDecodeError:
            info = {}
        url = info.get("url") or None
        if isinstance(url, str) and len(url) > 500:
            url = url[:500]
        if info.get("dir_info", {}).get("editable"):
            return "editable", url
        if isinstance(url, str) and (url.startswith("git+") or url.startswith("git://")):
            return "vcs", url
        if isinstance(url, str) and url.startswith("file:"):
            return "local-file", url
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return "download", url
        return "other", url
    return "download", None


def storage_and_archive(dist: Path):
    """Classify every payload file in a distribution, not a sample.

    Returns (storage, archive_id, payload_files). dist-info/egg-info bookkeeping
    files are excluded so the label describes the package payload only.
    payload_files counts symlinks into the uv cache, regular files in the venv,
    and dangling links; regular_bytes is the size of the private copies.
    """
    site = dist.parent
    record = dist / "RECORD"
    counts = {"symlink": 0, "regular": 0, "dangling_symlink": 0, "regular_bytes": 0}
    archive_id = None
    if not record.is_file():
        return "unknown", None, counts
    for line in read_text(record).splitlines():
        rel = line.split(",", 1)[0]
        if not rel or rel.endswith("/"):
            continue
        if ".dist-info/" in rel or ".egg-info/" in rel:
            continue
        candidate = site / rel
        try:
            if candidate.is_symlink():
                if not candidate.exists():
                    counts["dangling_symlink"] += 1
                    continue
                counts["symlink"] += 1
                if archive_id is None:
                    parts = Path(os.path.realpath(candidate)).parts
                    if "archive-v0" in parts:
                        idx = parts.index("archive-v0")
                        if idx + 1 < len(parts):
                            archive_id = parts[idx + 1]
            elif candidate.exists():
                counts["regular"] += 1
                counts["regular_bytes"] += candidate.lstat().st_size
        except OSError:
            counts["dangling_symlink"] += 1
    if counts["symlink"] and counts["regular"]:
        storage = "mixed"
    elif counts["symlink"]:
        storage = "uv-cache-symlink"
    elif counts["regular"]:
        storage = "private-copy"
    else:
        storage = "unknown"
    return storage, archive_id, counts


def dist_name_version(dist: Path):
    meta = dist / "METADATA"
    if not meta.is_file():
        meta = dist / "PKG-INFO"
    if meta.is_file():
        name, version = parse_metadata(read_text(meta))
        if name and version:
            return name, version
    stem = dist.name
    for suffix in (".dist-info", ".egg-info"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "-" in stem:
        name, version = stem.rsplit("-", 1)
        return name.replace("_", "-"), version
    return stem, "unknown"


def iter_dists(site: Path):
    if not site.is_dir():
        return
    for child in site.iterdir():
        if child.name.endswith(".dist-info") or child.name.endswith(".egg-info"):
            yield child


def site_packages_of(venv: Path):
    lib = venv / "lib"
    if not lib.is_dir():
        return []
    return sorted(p for p in lib.glob("python*/site-packages") if p.is_dir())


def scan_site(site: Path, location: str, referenced_archives: set, records: list):
    count = 0
    for dist in iter_dists(site):
        name, version = dist_name_version(dist)
        origin, url = classify(dist)
        size = record_size(dist)
        storage, archive_id, payload_files = storage_and_archive(dist)
        if archive_id:
            referenced_archives.add(archive_id)
        records.append(
            {
                "name": name,
                "version": version,
                "size_bytes": size,
                "origin": origin,
                "url": url,
                "location": location,
                "storage": storage,
                "archive_id": archive_id,
                "payload_files": payload_files,
            }
        )
        count += 1
    return count


def dir_stats(path: Path, max_entries=None):
    if not path.exists():
        return {"exists": False}
    if path.is_file():
        return {"exists": True, "is_file": True, "size_bytes": path.stat().st_size}
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError as exc:
            target = str(exc)
        return {"exists": True, "is_symlink": True, "target": target, "target_exists": path.exists()}
    children = []
    try:
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            children.append(child.name)
    except OSError as exc:
        return {"exists": True, "error": str(exc)}
    # Don't walk huge caches file-by-file here; callers pass counts they computed.
    info = {"exists": True, "child_count": len(children)}
    if len(children) <= 40:
        info["children"] = children
    return info


def cache_only_archives(cache_root: Path, referenced: set, records: list):
    archives = cache_root / "archive-v0"
    if not archives.is_dir():
        return {"archive_count": 0, "cache_only_count": 0}
    archive_count = 0
    cache_only = 0
    for archive in archives.iterdir():
        if not archive.is_dir():
            continue
        archive_count += 1
        if archive.name in referenced:
            continue
        found = False
        try:
            children = list(archive.iterdir())
        except OSError:
            continue
        for child in children:
            if not (child.name.endswith(".dist-info") or child.name.endswith(".egg-info")):
                continue
            found = True
            name, version = dist_name_version(child)
            origin, url = classify(child)
            size = record_size(child)
            records.append(
                {
                    "name": name,
                    "version": version,
                    "size_bytes": size,
                    "origin": origin,
                    "url": url,
                    "location": f"/opt/uv_cache/archive-v0/{archive.name}",
                    "storage": "uv-cache-only",
                    "archive_id": archive.name,
                }
            )
        if found:
            cache_only += 1
        else:
            cache_only += 1
            records.append(
                {
                    "name": archive.name,
                    "version": "unknown",
                    "size_bytes": None,
                    "origin": "cache-archive-without-metadata",
                    "url": None,
                    "location": f"/opt/uv_cache/archive-v0/{archive.name}",
                    "storage": "uv-cache-only",
                    "archive_id": archive.name,
                }
            )
    return {
        "archive_count": archive_count,
        "referenced_archive_count": len(referenced),
        "cache_only_archive_count": cache_only,
    }


def child_dir_count(path: Path):
    if not path.is_dir():
        return None
    try:
        return sum(1 for p in path.iterdir() if p.is_dir())
    except OSError:
        return None


def aggregate(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[canon(rec["name"])].append(rec)
    packages = []
    editable = []
    for key in sorted(grouped):
        recs = grouped[key]
        display = recs[0]["name"]
        for rec in recs:
            if rec["name"] == rec["name"].lower() or "-" in rec["name"]:
                display = rec["name"]
                break
        downloaded = [r for r in recs if r["origin"] != "editable"]
        edits = [r for r in recs if r["origin"] == "editable"]
        if downloaded:
            packages.append(package_object(display, downloaded))
        if edits:
            editable.append(package_object(display, edits, count_key="install_count"))
    return packages, editable


def package_object(display, recs, count_key="download_count"):
    by_version = defaultdict(list)
    for rec in recs:
        by_version[rec["version"]].append(rec)
    versions = []
    for version in sorted(by_version, key=lambda v: (v is None, str(v))):
        items = by_version[version]
        sizes = sorted({i["size_bytes"] for i in items if i["size_bytes"] is not None})
        storages = sorted({i["storage"] for i in items})
        origins = sorted({i["origin"] for i in items})
        urls = sorted({i["url"] for i in items if i["url"]})
        locations = [
            {
                "path": i["location"],
                "storage": i["storage"],
                "origin": i["origin"],
                "payload_files": i.get("payload_files"),
            }
            for i in sorted(items, key=lambda i: i["location"])
        ]
        entry = {
            "version": version,
            "size_bytes": sizes[0] if len(sizes) == 1 else (max(sizes) if sizes else None),
            count_key: len(items),
            "locations": locations,
            "storage": storages[0] if len(storages) == 1 else "mixed",
            "origin": origins[0] if len(origins) == 1 else origins,
        }
        if len(sizes) > 1:
            entry["size_bytes_observed"] = sizes
        if urls:
            entry["urls"] = urls
        versions.append(entry)
    total = sum(v[count_key] for v in versions)
    obj = {
        "name": display,
        count_key: total,
        "versions": versions,
    }
    return obj


def image_ref(registry, name, tag):
    return f"{registry.rstrip('/')}/{name}:{tag}"


def docker_image_metadata(ref):
    """Return id, created, and size from the local Docker daemon."""
    try:
        raw = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}\t{{.Created}}\t{{.Size}}", ref],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.output or "").strip()
        raise SystemExit(f"docker image inspect {ref} failed: {detail}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"docker image inspect {ref} failed: docker is not on PATH") from exc
    image_id, created, size = raw.strip().split("\t")
    return {"id": image_id, "created": created, "size_bytes": int(size)}


def cuda_base_image_ref():
    """Read ARG BASE_IMAGE from Dockerfile.nhx-rl-base, one directory above this script."""
    if not DOCKERFILE.is_file():
        raise SystemExit(f"CUDA base image Dockerfile not found: {DOCKERFILE}")
    for line in DOCKERFILE.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("ARG BASE_IMAGE="):
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise SystemExit(f"ARG BASE_IMAGE=<image> is not set in {DOCKERFILE}")


def image_record(registry, tag):
    registry = registry.rstrip("/")
    images = []
    for name in IMAGE_NAMES:
        ref = image_ref(registry, name, tag)
        metadata = docker_image_metadata(ref)
        images.append({"repository": f"{registry}/{name}", "tag": tag, **metadata})
    training, base = images
    return {
        "registry": registry,
        "repository": training["repository"],
        "tag": tag,
        "id": training["id"],
        "created": training["created"],
        "size_bytes": training["size_bytes"],
        "rl_base": {
            "repository": base["repository"],
            "tag": tag,
            "id": base["id"],
            "created": base["created"],
            "size_bytes": base["size_bytes"],
        },
    }


def base_image_packages(base_image_ref):
    snapshot_path = Path("/tmp/base-dpkg-packages.txt")
    snapshot = []
    if snapshot_path.is_file():
        for line in read_text(snapshot_path).splitlines():
            if not line.strip():
                continue
            name, version = line.split("\t", 1)
            snapshot.append((name, version))
    current = {}
    try:
        raw = subprocess.check_output(
            [
                "dpkg-query",
                "-W",
                "-f",
                "${binary:Package}\t${Version}\t${Installed-Size}\t${db:Status-Status}\n",
            ],
            text=True,
            errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raw = ""
        query_error = str(exc)
    else:
        query_error = None
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, version, size_kb, status = parts
        size_bytes = None
        if size_kb.isdigit():
            size_bytes = int(size_kb) * 1024
        current[name] = {"version": version, "size_bytes": size_bytes, "status": status}
    packages = []
    matched_size = 0
    for name, version in snapshot:
        cur = current.get(name)
        if cur is None:
            state = "removed"
            size_bytes = None
            installed_version = None
        elif cur["version"] == version:
            state = "installed"
            size_bytes = cur["size_bytes"]
            installed_version = cur["version"]
            if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
                matched_size += size_bytes
        else:
            state = "upgraded"
            size_bytes = None
            installed_version = cur["version"]
        item = {
            "name": name,
            "version": version,
            "size_bytes": size_bytes,
            "state": state,
        }
        if installed_version and installed_version != version:
            item["installed_version"] = installed_version
            if cur and cur["size_bytes"] is not None:
                item["installed_size_bytes"] = cur["size_bytes"]
        packages.append(item)
    return {
        "reference": base_image_ref,
        "inventory_source": "/tmp/base-dpkg-packages.txt",
        "size_note": "size_bytes is dpkg Installed-Size in bytes (KiB * 1024) and is set only when the final image still has the base-image version. Upgraded packages keep the base version in version and report the replacement separately. Removed packages were in the base image and are no longer installed.",
        "query_error": query_error,
        "package_count": len(packages),
        "installed_same_version_count": sum(1 for p in packages if p["state"] == "installed"),
        "upgraded_count": sum(1 for p in packages if p["state"] == "upgraded"),
        "removed_count": sum(1 for p in packages if p["state"] == "removed"),
        "installed_same_version_size_bytes": matched_size,
        "packages": packages,
    }


def discover_extra_venvs(known_paths: set):
    roots = [Path("/opt"), Path("/app"), Path("/home"), Path("/usr/local"), Path("/usr/lib")]
    prune = {
        "uv_cache",
        "archive-v0",
        "wheels-v7",
        "sdists-v10",
        "site-packages",
        "__pycache__",
        ".git",
        "node_modules",
        "proc",
        "sys",
    }
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in prune]
            if "pyvenv.cfg" in filenames:
                venv = Path(dirpath)
                if str(venv) not in known_paths:
                    found.append(str(venv))
    return found


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inventory Python packages in an nhx-rl image.")
    parser.add_argument(
        "--registry",
        required=True,
        help="Registry and repository prefix shared by nhx-rl-training and nhx-rl-base. "
        "Example: nvcr.io/0921617854601259/nemo-platform-dev",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag applied to every image under --registry: " + ", ".join(IMAGE_NAMES) + ".",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    image = image_record(args.registry, args.tag)
    cuda_base_image = cuda_base_image_ref()
    referenced = set()
    records = []
    location_results = []
    known_paths = {item["path"] for item in KNOWN_LOCATIONS}

    venv_paths = []
    for item in KNOWN_LOCATIONS:
        path = Path(item["path"])
        entry = {
            "path": item["path"],
            "kind": item["kind"],
            "dockerfile": item["dockerfile"],
            "role": item["role"],
            "exists": path.exists() or path.is_symlink(),
        }
        if item["kind"] == "venv":
            venv_paths.append(item["path"])
            if path.is_dir():
                dist_count = 0
                sites = site_packages_of(path)
                entry["site_packages"] = [str(s) for s in sites]
                for site in sites:
                    dist_count += scan_site(site, item["path"], referenced, records)
                entry["distribution_count"] = dist_count
            else:
                entry["distribution_count"] = 0
        elif item["path"] == "/opt/cpython" and path.is_dir():
            sites = list(path.glob("lib/python*/site-packages"))
            entry["site_packages"] = [str(s) for s in sites]
            dist_count = 0
            for site in sites:
                dist_count += scan_site(site, item["path"], referenced, records)
            entry["distribution_count"] = dist_count
        elif item["path"] == "/opt/gym_venvs" and path.is_dir():
            children = sorted(p.name for p in path.iterdir() if p.is_dir())
            entry["venv_count"] = len(children)
            entry["venvs"] = children
            for child in path.iterdir():
                if not (child / "pyvenv.cfg").is_file():
                    continue
                known_paths.add(str(child))
                for site in site_packages_of(child):
                    scan_site(site, str(child), referenced, records)
        elif item["path"] == "/opt/gym-wheels":
            entry.update(dir_stats(path))
            if path.is_dir():
                wheels = []
                for wheel in sorted(path.glob("*.whl")):
                    wheels.append({"name": wheel.name, "size_bytes": wheel.stat().st_size})
                entry["wheels"] = wheels
        elif item["path"] in {
            "/opt/uv_cache/wheels-v7",
            "/opt/uv_cache/sdists-v10",
            "/root/.cache/uv",
            "/home/ubuntu/.cache/uv",
        }:
            entry["directory_count"] = child_dir_count(path) if path.is_dir() else None
            if path.is_dir():
                try:
                    entry["child_count"] = sum(1 for _ in path.iterdir())
                except OSError as exc:
                    entry["error"] = str(exc)
        elif item["path"] == "/usr/share/nemo-helix/source-distributions" and path.is_dir():
            files = 0
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    files += 1
                    try:
                        total += (Path(dirpath) / filename).stat().st_size
                    except OSError:
                        pass
            entry["file_count"] = files
            entry["size_bytes"] = total
        elif item["kind"] == "symlink" and path.is_symlink():
            entry["target"] = os.readlink(path)
        location_results.append(entry)

    extras = discover_extra_venvs(known_paths)
    for extra in extras:
        path = Path(extra)
        dist_count = 0
        sites = site_packages_of(path)
        for site in sites:
            dist_count += scan_site(site, extra, referenced, records)
        location_results.append(
            {
                "path": extra,
                "kind": "venv",
                "dockerfile": "discovered",
                "role": "pyvenv.cfg found outside the Dockerfile install paths.",
                "exists": True,
                "site_packages": [str(s) for s in sites],
                "distribution_count": dist_count,
            }
        )

    for pattern_root in (Path("/usr/lib"), Path("/usr/local/lib")):
        if not pattern_root.is_dir():
            continue
        for site in pattern_root.glob("python*/dist-packages"):
            count = scan_site(site, str(site), referenced, records)
            location_results.append(
                {
                    "path": str(site),
                    "kind": "system-python",
                    "dockerfile": "base image or apt",
                    "role": "Discovered system dist-packages.",
                    "exists": True,
                    "distribution_count": count,
                }
            )

    cache_info = cache_only_archives(Path("/opt/uv_cache"), referenced, records)
    for entry in location_results:
        if entry["path"] == "/opt/uv_cache/archive-v0":
            entry.update(cache_info)
            entry["exists"] = Path("/opt/uv_cache/archive-v0").is_dir()
        if entry["path"] == "/opt/uv_cache":
            entry["exists"] = Path("/opt/uv_cache").is_dir()
            entry["top_level"] = (
                sorted(p.name for p in Path("/opt/uv_cache").iterdir()) if Path("/opt/uv_cache").is_dir() else []
            )

    packages, editable = aggregate(records)
    packages.sort(key=lambda p: (-p["download_count"], p["name"]))
    editable.sort(key=lambda p: p["name"])

    downloaded_events = sum(p["download_count"] for p in packages)
    version_count = sum(len(p["versions"]) for p in packages)
    largest = sorted(
        (
            {
                "name": p["name"],
                "version": v["version"],
                "size_bytes": v["size_bytes"],
                "download_count": v["download_count"],
                "storage": v["storage"],
            }
            for p in packages
            for v in p["versions"]
            if v["size_bytes"]
        ),
        key=lambda item: item["size_bytes"],
        reverse=True,
    )[:25]
    most_repeated = [
        {
            "name": p["name"],
            "download_count": p["download_count"],
            "versions": [v["version"] for v in p["versions"]],
        }
        for p in packages[:25]
    ]

    # Re-sort the published package list by name for lookup. Keep the ranked
    # views in the summary.
    packages_by_name = sorted(packages, key=lambda p: p["name"].lower())

    result = {
        "image": image,
        "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": {
            "download_count": "Number of virtual environments that have the package installed, plus extracted /opt/uv_cache/archive-v0 copies that no venv file links to.",
            "size_bytes": "Sum of the file sizes in the distribution RECORD. That is the size of one copy of that version. It is not multiplied by download_count.",
            "storage": "Every non-metadata payload file in each install is checked. uv-cache-symlink: all payload files are symlinks into /opt/uv_cache (one shared on-disk copy). private-copy: all payload files are regular files in the venv. mixed: both kinds are present; see payload_files for exact counts and private_copy_bytes_total for the private bytes. uv-cache-only: the archive is in the cache but no venv links to it. unknown: RECORD missing or no payload files.",
            "payload_files": "Per install: symlink = links into uv cache; regular = private files; dangling_symlink = links with no target; regular_bytes = size of the regular files (the actual per-venv private copy).",
            "editable_packages": "Editable installs from /opt/nemo-rl and /app are listed separately. They are not downloads.",
            "base_image_packages": "dpkg packages from "
            + (cuda_base_image or "the CUDA base image")
            + ", read from /tmp/base-dpkg-packages.txt.",
            "locations_searched": "Every install path named in Dockerfile.nhx-rl-base and Dockerfile.nhx-rl-training, plus any extra pyvenv.cfg discovered under /opt, /app, /home, /usr/local, and /usr/lib.",
        },
        "summary": {
            "venv_distribution_records": sum(
                1
                for r in records
                if r["storage"] != "uv-cache-only"
                and not str(r["location"]).startswith("/opt/uv_cache/")
                and r["origin"] != "editable"
                and r["location"] != "/opt/cpython"
            ),
            "unique_downloaded_packages": len(packages_by_name),
            "unique_downloaded_versions": version_count,
            "download_events": downloaded_events,
            "private_copy_bytes_total": sum(
                (r.get("payload_files") or {}).get("regular_bytes", 0) for r in records if r["origin"] != "editable"
            ),
            "editable_packages": len(editable),
            "cache_only_archives": cache_info.get("cache_only_archive_count"),
            "uv_archives": cache_info.get("archive_count"),
            "uv_archives_referenced_by_venvs": cache_info.get("referenced_archive_count"),
            "largest_packages": largest,
            "most_repeated_packages": most_repeated,
        },
        "locations_searched": location_results,
        "base_image_packages": base_image_packages(cuda_base_image),
        "packages": packages_by_name,
        "editable_packages": editable,
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
