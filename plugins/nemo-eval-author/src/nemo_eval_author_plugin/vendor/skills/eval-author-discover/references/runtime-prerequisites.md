<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Runtime prerequisites

Read for readiness validation or the Harbor path of **Get the evaluation runtime ready**. A file-only
Explore stays in the parent skill and does not run these probes.

## Contents

- [Resolve and verify the runtime](#runtime-prerequisite-checks)
- [Check optional Harbor assistant skills](#check-for-optional-harbor-skills)

## Runtime prerequisite checks

Identify an interpreter that can import Harbor. Full discovery must later use
that interpreter for Harbor's validators to judge readiness.

Read the repository's setup instructions and Harbor version requirements first.
Use its documented environment when present; otherwise locate an existing one.
A `harbor` command on your `PATH` does not mean Harbor is importable by the Python
you are about to run. A repository with its own virtual environment usually needs
that environment's interpreter. Try these in order until one prints a version:

```bash
harbor_python=""
for py in .venv/bin/python ./venv/bin/python python3; do
  if "$py" -c "import harbor, sys; print(sys.executable, harbor.__version__)" 2>/dev/null; then
    harbor_python="$py"
    break
  fi
done
```

If none prints a version, check an existing uv tool installation before declaring
Harbor unavailable. `uv tool install harbor` isolates Harbor from project Python:

```bash
if [ -z "$harbor_python" ] && command -v uv >/dev/null 2>&1; then
  if harbor_tool_root="$(uv tool dir)" &&
    "$harbor_tool_root/harbor/bin/python" -c \
      "import harbor, sys; print(sys.executable, harbor.__version__)"; then
    harbor_python="$harbor_tool_root/harbor/bin/python"
  fi
fi
```

If these probes fail but a `harbor` executable exists, resolve that executable's
symlink and inspect its launcher to locate the existing environment's Python
(for example, an installed uv tool environment). Verify that interpreter with
the same import check before using it. Do not assume that a project interpreter's
failed import proves Harbor is absent everywhere, and do not run `uv tool run`
to probe availability because it can install a tool.

If no existing interpreter can import Harbor, read
[Help the user get Harbor ready](harbor-setup.md). For prerequisite-only
work, return the missing or broken setup finding to the caller. During full
discovery, still run inventory with an available Python 3.11+ interpreter and use
the [empty-scan conversation](../SKILL.md#when-no-harbor-evals-were-found) when appropriate. Missing Harbor blocks readiness
validation, not that inventory. Include the setup requirement in the full report
even when the inventory finds no Harbor evals.

Keep the verified `harbor_python` path for [Step 1](../SKILL.md#step-1-run-discovery), including across shell sessions.
Check `harbor --help` using the corresponding installation. The Harbor setup path
is complete when that command starts, the selected interpreter imports Harbor
and reports a version consistent with repository requirements, and the invocation
is recorded for reuse. Runtime setup does not prove task-specific backend
readiness, agent access, or successful runs.

Return the verified command, interpreter, version, and unresolved setup needs to
the calling stage. Full discovery later records its runtime mode in
`runtime.harbor_importable` and the top-level `proven` field; prerequisite checks
alone do not produce a suite-readiness verdict or discovery evidence JSON.

## Check for optional Harbor skills

Alongside the runtime check, look for
[Harbor's own skills](https://github.com/harbor-framework/harbor/tree/main/skills).
These are guides for the coding assistant working on evals; they are separate
from the Harbor runtime and from Eval Author. They are recommended, not required.
Check even when Harbor is missing or no Harbor evals were found.

Start with the current assistant's available-skill catalog. Also inspect existing
repository skill directories, such as `.agents/skills/` or `.claude/skills/`, and
skill locations exposed by the host or explicitly supplied by the user. Keep the
search bounded to those locations; do not scan the user's entire home directory.
Use skill metadata and source references to identify Harbor's skills, not a name
match alone. Names such as `create-task`, `create-adapter`, and `harbor-exec` are
examples from the linked collection, not an exhaustive installation checklist.
Do not mistake Eval Author itself or a Harbor source checkout for skills loaded
by the current assistant.

Record which skills are available in the session, which files were found but
whose availability is unconfirmed, and which locations were checked. If none
were found, say “not found in the locations checked.” If access or the host's
catalog is unavailable, report that limitation rather than asserting they are
not installed. A subset is useful; do not require every skill in the collection.

When presenting the runtime findings, briefly explain their purpose and
observed availability. When missing or unconfirmed, recommend them and link the
official collection. For example, when the search found none:

> Harbor also provides skills that guide your coding assistant through creating
> and working with evals. I didn't find them in the locations I checked. They're
> optional, but recommended: see [Harbor's skills](https://github.com/harbor-framework/harbor/tree/main/skills).
> We can continue without them.

When skills are available, name the relevant ones briefly instead of recommending
another installation. Include this advisory with the runtime findings. Do not
install or invoke skills as a detection step, ask the user to install them before
continuing, or add a new approval gate.
Their absence changes no readiness checks, `proven`, `runnable`, or exit code.
The script cannot determine host skill availability. Return these observations
with prerequisite-only findings; include them separately from evidence JSON in
the later full discovery report as described in [Step 6](../SKILL.md#step-6-save-the-report).
