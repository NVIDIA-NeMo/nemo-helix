# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RENAME_SCRIPT = REPO_ROOT / "tools/rename/rename-to-nemo-helix.sh"
RENAME_IMPL = REPO_ROOT / "tools/rename/rename_to_nemo_helix.py"
VERIFY_SCRIPT = REPO_ROOT / "tools/rename/verify-nemo-helix-rename.sh"
VERIFY_IMPL = REPO_ROOT / "tools/rename/verify_nemo_helix_rename.py"
COMMON_IMPL = REPO_ROOT / "tools/rename/rename_common.py"
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
    shutil.copy2(RENAME_IMPL, tools_dir / RENAME_IMPL.name)
    shutil.copy2(VERIFY_SCRIPT, tools_dir / VERIFY_SCRIPT.name)
    shutil.copy2(VERIFY_IMPL, tools_dir / VERIFY_IMPL.name)
    shutil.copy2(COMMON_IMPL, tools_dir / COMMON_IMPL.name)
    shutil.copy2(README, tools_dir / README.name)


def test_rename_scans_tracked_ignored_files_without_rewriting_itself(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / ".gitignore").write_text("dist/\n")
    (repo / "dist").mkdir()
    (repo / "dist/index.js").write_text("window.product = 'NeMo Platform'; window.acronym = 'NMP';\n")
    (repo / "dist/binary.dat").write_bytes(b"NeMo Platform NMP should remain in undecodable content: \xff\n")
    (repo / ".claude/skills").mkdir(parents=True)
    (repo / "docs/snmp/nmp-common").mkdir(parents=True)
    (repo / "docs/snmp/nmp-common/NMP_DATA.txt").write_text(
        " ".join(
            [
                "nmp_common",
                "NMP_CONFIG",
                "nmp-common",
                "NMPJobContext",
                "TestNMPJobContextFromEnv",
                "NMPOIDCConfig",
                "NMPModelProvider",
                "NMPSecretResolver",
                "NMPGenerationLog",
                "NmpContext",
                "NmpCliRunner",
                "NmpErrorHandlingMixin",
                "NmpDynamicVersionSource",
                "NmpRun",
                "TestNmpOption",
                "ManifestBackedNmpGroup",
                "nmpBaseURLEnv",
                "nmp-intake",
                r"nhx-intake-clickhouse-one\nnmp-intake-clickhouse-two",
                "nmp2",
                "_xnmp",
                "nmpclient",
                "nmpcontext",
                "NMP",
                "nmp",
                "snmp",
                "abcNMPdef012",
                "abc-nmpXYZ",
                "Nmpnotpascal",
                "nmpclientExtra",
                "sha384-AbCdNMPefghnmpQRST==",
            ]
        )
        + "\n"
    )
    (repo / "docs/overview.md").write_text("The nemo-platform repository publishes auditor-tasks.\n")
    (repo / "docker-bake.hcl").write_text('target "images" { tags = sha_and_maybe_latest_tags("auditor-tasks") }\n')

    run(["git", "add", "."], repo)
    run(["git", "add", "-f", "dist/index.js", "dist/binary.dat"], repo)
    run(["git", "commit", "-m", "initial"], repo)

    result = run([str(repo / "tools/rename/rename-to-nemo-helix.sh"), "--repo-dir", str(repo)], tmp_path)

    assert "tools/rename/verify-nemo-helix-rename.sh" in result.stdout
    index_bytes = (repo / "dist/index.js").read_bytes()
    assert b"NeMo Helix" in index_bytes
    assert b"NHX" in index_bytes
    assert (repo / "dist/binary.dat").read_bytes() == b"NeMo Platform NMP should remain in undecodable content: \xff\n"
    assert "nemo-helix" in (repo / "docs/overview.md").read_text()
    assert "nhx-auditor-tasks" in (repo / "docker-bake.hcl").read_text()

    renamed_path = repo / "docs/snmp/nhx-common/NHX_DATA.txt"
    assert renamed_path.exists()
    renamed_text = renamed_path.read_text()
    assert "nhx_common" in renamed_text
    assert "NHX_CONFIG" in renamed_text
    assert "nhx-common" in renamed_text
    assert "NHXJobContext" in renamed_text
    assert "TestNHXJobContextFromEnv" in renamed_text
    assert "NHXOIDCConfig" in renamed_text
    assert "NHXModelProvider" in renamed_text
    assert "NHXSecretResolver" in renamed_text
    assert "NHXGenerationLog" in renamed_text
    assert "NhxContext" in renamed_text
    assert "NhxCliRunner" in renamed_text
    assert "NhxErrorHandlingMixin" in renamed_text
    assert "NhxDynamicVersionSource" in renamed_text
    assert "NhxRun" in renamed_text
    assert "TestNhxOption" in renamed_text
    assert "ManifestBackedNhxGroup" in renamed_text
    assert "nhxBaseURLEnv" in renamed_text
    assert "nhx-intake" in renamed_text
    assert r"nhx-intake-clickhouse-one\nnhx-intake-clickhouse-two" in renamed_text
    assert "nhx2" in renamed_text
    assert "_xnhx" in renamed_text
    assert "nhxclient" in renamed_text
    assert "nhxcontext" in renamed_text
    assert " NHX " in f" {renamed_text} "
    assert " nhx " in f" {renamed_text} "
    assert "snmp" in renamed_text
    assert "abcNMPdef012" in renamed_text
    assert "abc-nmpXYZ" in renamed_text
    assert "Nmpnotpascal" in renamed_text
    assert "nmpclientExtra" in renamed_text
    assert "sha384-AbCdNMPefghnmpQRST==" in renamed_text

    common_text = (repo / "tools/rename/rename_common.py").read_text()
    assert '("NeMo Platform", "NeMo Helix")' in common_text
    assert '"auditor-tasks"' in common_text
    assert '"nhx-auditor-tasks"' not in common_text

    verify = run([str(repo / "tools/rename/verify-nemo-helix-rename.sh"), "--repo-dir", str(repo)], tmp_path)
    assert "No legacy product" in verify.stdout


def test_include_and_exclude_globs_limit_rename_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / "docs").mkdir()
    (repo / "web").mkdir()
    (repo / "README.md").write_text("NeMo Platform uses NMP.\n")
    (repo / "docs/guide.md").write_text("NeMo Platform uses NMP.\n")
    (repo / "docs/skip.md").write_text("NeMo Platform uses NMP.\n")
    (repo / "web/app.md").write_text("NeMo Platform uses NMP.\n")

    run(["git", "add", "."], repo)
    run(["git", "commit", "-m", "initial"], repo)

    run(
        [
            "tools/rename/rename-to-nemo-helix.sh",
            "--include-glob",
            "docs/**",
            "--exclude-glob",
            "docs/skip.md",
        ],
        repo,
    )

    assert (repo / "README.md").read_text() == "NeMo Platform uses NMP.\n"
    assert (repo / "docs/guide.md").read_text() == "NeMo Helix uses NHX.\n"
    assert (repo / "docs/skip.md").read_text() == "NeMo Platform uses NMP.\n"
    assert (repo / "web/app.md").read_text() == "NeMo Platform uses NMP.\n"

    verify = run(
        [
            "tools/rename/verify-nemo-helix-rename.sh",
            "--include-glob",
            "docs/**",
            "--exclude-glob",
            "docs/skip.md",
        ],
        repo,
    )
    assert "No legacy product" in verify.stdout


def test_root_globs_do_not_match_nested_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / "docs").mkdir()
    (repo / "README.md").write_text("NeMo Platform uses NMP.\n")
    (repo / "docs/guide.md").write_text("NeMo Platform uses NMP.\n")

    run(["git", "add", "."], repo)
    run(["git", "commit", "-m", "initial"], repo)

    run(["tools/rename/rename-to-nemo-helix.sh", "--include-glob", "*.md"], repo)

    assert (repo / "README.md").read_text() == "NeMo Helix uses NHX.\n"
    assert (repo / "docs/guide.md").read_text() == "NeMo Platform uses NMP.\n"


def test_verifier_scans_tracked_ignored_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_git_repo(repo)
    install_rename_tools(repo)

    (repo / ".gitignore").write_text("dist/\n")
    (repo / "dist").mkdir()
    (repo / "dist/index.js").write_text(
        "const product = 'NeMo Platform'; snmp abcNMPdef nmp_common NmpContext nmpclient NMPJobContext\n"
    )
    (repo / "docker-bake.hcl").write_text('target "images" { tags = sha_and_maybe_latest_tags("nhx-auditor-tasks") }\n')

    run(["git", "add", "."], repo)
    run(["git", "add", "-f", "dist/index.js"], repo)
    run(["git", "commit", "-m", "initial"], repo)

    result = run(["tools/rename/verify-nemo-helix-rename.sh"], repo, check=False)

    assert result.returncode == 1
    assert "dist/index.js" in result.stdout
    assert "Legacy product names remain" in result.stderr
    assert "Legacy acronym references remain" in result.stderr


def test_rename_tools_do_not_depend_on_perl_or_ripgrep() -> None:
    checked_files = [RENAME_SCRIPT, RENAME_IMPL, VERIFY_SCRIPT, VERIFY_IMPL, COMMON_IMPL, README]
    combined_text = "\n".join(path.read_text() for path in checked_files)

    assert "perl" not in combined_text.lower()
    assert "ripgrep" not in combined_text.lower()
    assert not re.search(r"\brg\b", combined_text)
