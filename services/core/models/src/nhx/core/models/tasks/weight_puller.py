# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download model weights with an explicit remote-access check."""

import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)


def wait_for_proxy(url: str, *, timeout: float = 60) -> None:
    """Wait for the Docker sidecar to join the puller's network namespace."""
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=1) as client:
        while True:
            try:
                client.get(url).raise_for_status()
                return
            except httpx.HTTPError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Auth proxy did not become ready") from None
                time.sleep(min(1, remaining))


def pull_weights(repo_id: str, local_dir: Path, *, revision: str | None = None) -> None:
    """Authorize against the remote manifest before reusing immutable cache entries.

    Avoid snapshot_download's local_dir fallback: remote failures must not turn
    an existing model directory into a successful pull of stale weights.
    """
    endpoint = os.environ.get("HF_ENDPOINT") or None
    token = os.environ.get("HF_TOKEN") or None
    info = HfApi(endpoint=endpoint, token=token).model_info(repo_id, revision=revision)
    if not info.sha or info.siblings is None:
        raise RuntimeError("Remote repository did not provide a commit and file manifest")

    local_dir.mkdir(parents=True, exist_ok=True)
    root = local_dir.resolve()

    def download(filename: str) -> None:
        destination = local_dir / filename
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Repository file path escapes the model directory")
        cached = Path(hf_hub_download(repo_id, filename, revision=info.sha, endpoint=endpoint, token=token)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # HF_HOME and the model directory share a volume by default. Hard links
        # reuse the cache without storing a second copy of large weight files.
        with TemporaryDirectory(prefix=".puller-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "file"
            try:
                os.link(cached, staged)
            except OSError:
                shutil.copyfile(cached, staged)
            os.replace(staged, destination)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, (file.rfilename for file in info.siblings)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--proxy-health-url")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.proxy_health_url:
            wait_for_proxy(args.proxy_health_url)
        pull_weights(args.repo_id, args.local_dir, revision=args.revision)
    except Exception:
        logger.exception("Model weight pull failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
