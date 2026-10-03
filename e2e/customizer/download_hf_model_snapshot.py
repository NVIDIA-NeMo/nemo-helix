# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download a Hugging Face model snapshot for customizer E2E asset publishing.

Used by publish_assets_to_s3.sh (off-CI only). CI must sync pre-published
snapshots from s3://aire-e2e-assets/models/ instead of pulling from HF.
"""

import os
import sys
from pathlib import Path

from e2e.customizer.assets_manifest import model_ignore_globs


def _parse_ignore_patterns(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    patterns = [p.strip() for p in raw.split(",") if p.strip()]
    return patterns or None


def _default_ignore_globs() -> str:
    return model_ignore_globs()


def download_snapshot(
    repo: str,
    local_dir: str | Path,
    *,
    ignore_patterns: str | None = None,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    globs = ignore_patterns or _default_ignore_globs()
    path = snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=str(local_dir),
        ignore_patterns=_parse_ignore_patterns(globs),
        token=token or os.environ.get("HF_TOKEN"),
    )
    return Path(path)


def main() -> int:
    # Invoked by publish_assets_to_s3.sh with env vars (no CLI args).
    repo = os.environ["HF_REPO"]
    local_dir = os.environ["LOCAL_DIR"]
    path = download_snapshot(
        repo,
        local_dir,
        ignore_patterns=os.environ.get("MODEL_IGNORE_GLOBS"),
    )
    print(f"Downloaded {repo} to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
