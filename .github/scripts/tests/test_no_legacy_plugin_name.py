# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: the pre-rename plugin name ("auditor") must not reappear outside the compatibility surface.

The Garak Plugin used to be called NeMo Auditor. Its old name survives only in the one-release
compatibility shims, the migration guide, released history, and a few unrelated English uses.
Anything else that mentions it is a regression. When the shims are deleted, delete their entries
from ``ALLOWED`` as well.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

LEGACY_NAME = re.compile(r"auditor", re.IGNORECASE)
# Plugin-owned nouns from before "audit" became "scan"; checked only inside the plugin's own files.
LEGACY_NOUNS = re.compile(r"Audit(Config|Target|Job|Spec|InputSpec)\b|audit_(config|target|job)\b|jobs/audit\b")
PLUGIN_PREFIXES = (
    "plugins/garak-plugin/",
    "packages/nemo_helix_plugin/src/nemo_helix_plugin/garak_plugin/",
    "e2e/garak_plugin/",
    "docs/garak-plugin/",
)

# Compatibility shims and history. Each entry is a path prefix; a trailing "/" matches a directory.
ALLOWED = (
    # Released history is not rewritten.
    "docs/about/release-notes/",
    # The plugin's one-release shims, their tests, and the migration guide.
    "plugins/garak-plugin/src/garak_plugin/_legacy",
    "plugins/garak-plugin/tests/test_legacy",
    "plugins/garak-plugin/tests/test_authz.py",
    "plugins/garak-plugin/tests/test_config.py",
    "plugins/garak-plugin/pyproject.toml",
    "docs/garak-plugin/migration-guide.mdx",
    # Shim touch points in shared packages.
    "packages/nemo_helix_plugin/src/nemo_helix_plugin/client/client.py",
    "packages/nemo_helix_plugin/tests/client/test_client_resources.py",
    "packages/nemo_helix_ext/src/nemo_helix_ext/cli/manifest.py",
    "packages/nemo_helix_ext/src/nemo_helix_ext/cli/commands/skills/installer.py",
    "packages/nemo_helix_ext/tests/cli/test_app.py",
    "packages/nemo_helix_ext/tests/cli/commands/skills/agents/test_claude.py",
    "packages/nemo_helix/pyproject.toml",
    "services/platform-seed/src/nhx/platform_seed/config.py",
    "services/platform-seed/tests/test_platform_seed_runner.py",
    "services/platform-seed/README.md",
    "docker-bake.hcl",
    # Lock files record the deprecated extras.
    "uv.lock",
    "agents/nemo-studio-assistant/uv.lock",
    # This guard.
    ".github/scripts/tests/test_no_legacy_plugin_name.py",
    # Unrelated uses of the English word.
    "packages/nemo_evaluator_sdk/tests/agent_eval/test_gym_runtime.py",
    "plugins/_temporary-scaled-evals/tests/test_sandbox_k8s_harbor_patch.py",
    "services/jailbreak-detect/scripts/eval_dataset.jsonl",
    "web/packages/studio/public/sample-datasets/",
    "web/packages/studio/src/mocks/",
)


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    return [path for path in out.decode().split("\0") if path]


def _is_allowed(path: str) -> bool:
    return path.startswith(ALLOWED)


def _read(path: str) -> str | None:
    file = REPO_ROOT / path
    if not file.is_file() or file.is_symlink():
        return None
    try:
        return file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_old_plugin_name_is_absent_from_paths_and_contents() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if _is_allowed(path):
            continue
        if LEGACY_NAME.search(path):
            offenders.append(f"{path} (path)")
            continue
        text = _read(path)
        if text is not None and LEGACY_NAME.search(text):
            offenders.append(path)

    assert not offenders, "The pre-rename plugin name reappeared in:\n" + "\n".join(sorted(offenders))


def test_old_audit_nouns_are_absent_from_the_plugin_files() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.startswith(PLUGIN_PREFIXES) or _is_allowed(path):
            continue
        text = _read(path)
        if text is not None and LEGACY_NOUNS.search(text):
            offenders.append(path)

    assert not offenders, "Pre-rename audit nouns (use scan) reappeared in:\n" + "\n".join(sorted(offenders))
