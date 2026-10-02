#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enable the NeMo Helix MCP dev tools server locally.

This script configures your local Claude Code settings to enable the nhx-dev
MCP server without affecting other developers in the repository.

Uses 'claude mcp add' to add the server to local scope (~/.claude.json),
keeping the committed .mcp.json file clean.
"""

import json
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repository root by looking for .git directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory found)")


def add_mcp_server(repo_root: Path) -> bool:
    """Add nhx-dev server using 'claude mcp add' command.

    This adds the server to local scope (~/.claude.json) without modifying
    the committed .mcp.json file.
    """
    try:
        # Check if server already exists by running claude mcp list
        result = subprocess.run(
            ["claude", "mcp", "list", "--json"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            # Parse the JSON output to check if nhx-dev exists
            servers = json.loads(result.stdout)
            if any(server.get("name") == "nhx-dev" for server in servers):
                print("✓ nhx-dev server already configured in local scope")
                return False
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        # If claude command not found or list fails, proceed with add
        pass

    # Add the server using claude mcp add
    try:
        result = subprocess.run(
            [
                "claude",
                "mcp",
                "add",
                "nhx-dev",
                "--",
                "sh",
                "-c",
                "cd tools/mcp-dev-tools && uv run nhx-dev-mcp",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Added nhx-dev server to local config (~/.claude.json)")
        print("   This keeps the committed .mcp.json file clean")
        return True
    except subprocess.CalledProcessError as e:
        # Check if it's just an "already exists" error
        if "already exists" in e.stderr:
            print("✓ nhx-dev server already configured in local scope")
            return False
        print(f"❌ Failed to add MCP server: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 'claude' command not found. Is Claude Code installed?")
        return False


def update_settings_json(repo_root: Path) -> bool:
    """Add nhx-dev permissions to .claude/settings.local.json."""
    claude_dir = repo_root / ".claude"
    settings_path = claude_dir / "settings.local.json"

    # Create .claude directory if needed
    if not claude_dir.exists():
        print(f"📁 Creating {claude_dir}")
        claude_dir.mkdir(exist_ok=True)

    # Read existing settings or create new ones
    if settings_path.exists():
        print(f"📄 Reading existing {settings_path}")
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        print(f"📄 Creating new {settings_path}")
        settings = {}

    changed = False

    # Add permissions
    if "permissions" not in settings:
        settings["permissions"] = {"allow": []}
        changed = True

    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []
        changed = True

    if "mcp__nhx-dev__*" not in settings["permissions"]["allow"]:
        settings["permissions"]["allow"].append("mcp__nhx-dev__*")
        print("✅ Added mcp__nhx-dev__* to permissions.allow")
        changed = True
    else:
        print("✓ mcp__nhx-dev__* already in permissions.allow")

    # Add to enabledMcpjsonServers
    if "enabledMcpjsonServers" not in settings:
        settings["enabledMcpjsonServers"] = []
        changed = True

    if "nhx-dev" not in settings["enabledMcpjsonServers"]:
        settings["enabledMcpjsonServers"].append("nhx-dev")
        print("✅ Added nhx-dev to enabledMcpjsonServers")
        changed = True
    else:
        print("✓ nhx-dev already in enabledMcpjsonServers")

    # Write updated settings
    if changed:
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        print(f"✅ Updated {settings_path}")

    return changed


def update_agents_local(repo_root: Path) -> bool:
    """Add tool preference section to AGENTS.local.md."""
    agents_local_path = repo_root / "AGENTS.local.md"

    tool_preference_section = """
## Tool Preference (Local)

**CRITICAL: Use MCP dev tools instead of bash commands when available.**

The `nhx-dev` MCP server provides dedicated tools for common development tasks with better error handling and focused functionality. Always prefer these over equivalent bash commands:

**Git operations:** Use `git_status`, `git_log`, `git_branch_list`, `git_diff`, `git_diff_summary`, `git_diff_staged`, `git_show` instead of `git` bash commands

**Testing:** Use `run_unit_tests`, `run_integration_tests`, `run_service_tests`, `run_pytest` instead of `make test-*` bash commands

**Linting:** Use `run_ruff_check`, `run_ruff_format`, `run_type_check` instead of `uv run ruff` or `uv run ty`

**Pre-commit:** Use `run_precommit` instead of `uv run pre-commit run -a`

**Project navigation:** Use `list_directory`, `find_files` instead of `ls` or `find`

**Make targets:** Use `make_target` for allowed targets (`build-policy`, `check-licenses`, `check-policy`, `refresh-openapi`, `test-all`, `test-integration`, `test-policy`, `test-unit`, `update-licenses`, `update-sdk`) instead of `make` bash commands

Only fall back to bash commands when no MCP tool equivalent exists.
"""

    # Check if file exists and if it already has the section
    if agents_local_path.exists():
        content = agents_local_path.read_text()
        if "## Tool Preference" in content:
            print(f"✓ Tool Preference section already in {agents_local_path}")
            return False

        # Append to existing file
        print(f"📝 Appending to existing {agents_local_path}")
        with open(agents_local_path, "a") as f:
            f.write(tool_preference_section)
    else:
        # Create new file
        print(f"📝 Creating new {agents_local_path}")
        with open(agents_local_path, "w") as f:
            f.write("# Local Agent Instructions\n")
            f.write(tool_preference_section)

    print(f"✅ Updated {agents_local_path}")
    return True


def main():
    """Main entry point."""
    print("=" * 70)
    print("🔧 Enabling NeMo Helix MCP Dev Tools")
    print("=" * 70)
    print()

    try:
        repo_root = find_repo_root()
        print(f"📍 Repository root: {repo_root}\n")
    except RuntimeError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Update configuration files
    mcp_changed = add_mcp_server(repo_root)
    settings_changed = update_settings_json(repo_root)
    agents_changed = update_agents_local(repo_root)

    print()
    print("=" * 70)

    if mcp_changed or settings_changed or agents_changed:
        print("✅ NeMo Helix MCP Dev Tools Enabled!")
        print("=" * 70)
        print()
        print("📋 Next steps:")
        print("   1. Restart Claude Code to load the MCP server")
        print("   2. The dev tools will be available automatically")
        print()
        print("🔍 Available tools:")
        print("   - git_status, git_log, git_branch_list, git_diff, git_show, etc.")
        print(
            "   - run_unit_tests, run_integration_tests, run_service_tests, run_pytest"
        )
        print("   - run_precommit, run_ruff_check, run_ruff_format, run_type_check")
        print("   - list_directory, find_files")
        print("   - make_target (for allowlisted make commands)")
        print()
        print("📚 See tools/mcp-dev-tools/README.md for full documentation")
    else:
        print("✓ NeMo Helix MCP Dev Tools Already Enabled")
        print("=" * 70)
        print()
        print("All configuration is already in place. If the tools aren't")
        print("working, try restarting Claude Code.")

    print()


if __name__ == "__main__":
    main()
