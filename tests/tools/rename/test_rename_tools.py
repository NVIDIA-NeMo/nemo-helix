# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RENAME_SCRIPT = REPO_ROOT / "tools/rename/rename-to-nemo-helix.sh"
VERIFY_SCRIPT = REPO_ROOT / "tools/rename/verify-nemo-helix-rename.sh"
README = REPO_ROOT / "tools/rename/README.md"


def run(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def init_git_repo(path: Path) -> None:
    run(["git", "init"], path)
    run(["git", "config", "user.email", "rename-test@example.com"], path)
    run(["git", "config", "user.name", "Rename Test"], path)


def install_rename_tools(path: Path) -> None:
    tools_dir = path / "tools/rename"
    tools_dir.mkdir(parents=True)
    shutil.copy2(RENAME_SCRIPT, tools_dir / RENAME_SCRIPT.name)
    shutil.copy2(VERIFY_SCRIPT, tools_dir / VERIFY_SCRIPT.name)
    shutil.copy2(README, tools_dir / README.name)


def test_rename_scans_tracked_ignored_files_without_rewriting_itself(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / ".gitignore").write_text("dist/\n")
    (repo / "dist").mkdir()
    (repo / "dist/index.js").write_text("window.product = 'NeMo Platform'; window.acronym = 'NMP';\n")
    (repo / "docs").mkdir()
    (repo / "docs/overview.md").write_text("The nemo-platform repository publishes auditor-tasks.\n")
    (repo / "docker-bake.hcl").write_text('target "images" { tags = sha_and_maybe_latest_tags("auditor-tasks") }\n')

    run(["git", "add", "."], repo)
    run(["git", "add", "-f", "dist/index.js"], repo)
    run(["git", "commit", "-m", "initial"], repo)

    result = run(["tools/rename/rename-to-nemo-helix.sh"], repo)

    assert "tools/rename/verify-nemo-helix-rename.sh" in result.stdout
    assert "NeMo Helix" in (repo / "dist/index.js").read_text()
    assert "NHX" in (repo / "dist/index.js").read_text()
    assert "nemo-helix" in (repo / "docs/overview.md").read_text()
    assert "nhx-auditor-tasks" in (repo / "docker-bake.hcl").read_text()

    script_text = (repo / "tools/rename/rename-to-nemo-helix.sh").read_text()
    assert 'old_product="NeMo Platform"' in script_text
    assert '"auditor-tasks"' in script_text
    assert '"nhx-auditor-tasks"' not in script_text

    verify = run(["tools/rename/verify-nemo-helix-rename.sh"], repo)
    assert "No legacy product" in verify.stdout


def test_verifier_scans_tracked_ignored_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / ".gitignore").write_text("dist/\n")
    (repo / "dist").mkdir()
    (repo / "dist/index.js").write_text("const product = 'NeMo Platform';\n")
    (repo / "docker-bake.hcl").write_text('target "images" { tags = sha_and_maybe_latest_tags("nhx-auditor-tasks") }\n')

    run(["git", "add", "."], repo)
    run(["git", "add", "-f", "dist/index.js"], repo)
    run(["git", "commit", "-m", "initial"], repo)

    result = run(["tools/rename/verify-nemo-helix-rename.sh"], repo, check=False)

    assert result.returncode == 1
    assert "dist/index.js" in result.stdout
    assert "Legacy product names remain" in result.stderr
