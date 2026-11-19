<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR writer

A standalone Codex agent that implements a supplied plan in a GitHub repository,
runs relevant checks, and produces a patch or draft pull request. The plan can be
written by a person or supplied as a JSON object, including a `ready` plan from
the Linear ticket planner. This agent does not need access to Linear.

## Input

`input.example.json` targets `mmogallapalli/nemo-helix-hackathon`, branch `main`,
with a small documentation task in **patch mode**. It adds a comparison table
for existing agent configurations. Replace its plan with your intended task
before running it. Required fields are `repo`
(`owner/name`), `base_branch`, and `plan` (text or an object).

- `mode: patch` is the default. It edits a fresh job-local checkout and produces
  artifacts without pushing a branch or opening a PR.
- `mode: draft_pr` also requires an unused `branch` and a `commit_author` object
  with `name` and `email`. This explicitly authorizes pushing that new branch and
  opening a draft PR after required checks pass. Change any input constraints
  that prohibit publication when selecting this mode.
- `constraints` is an optional list of scope restrictions.

The writer inspects the actual code before implementing a plan. Missing
requirements produce `needs_clarification`; already implemented requirements
produce `no_changes`. Neither creates a PR. Failed or unavailable required checks
produce `failed`, retaining whatever local artifacts could be generated.

## Outputs

The `output_workdir` job result contains these files at the archive root:

| File | Contents |
| --- | --- |
| `result.json` | Status, repository/base commit, changed files, tests, deviations, errors, and publication details |
| `changes.patch` | Binary-capable diff against the recorded base SHA, including new files and deletions |
| `pr.md` | PR description with behavior and validation evidence |
| `reviewer-input.json` | Only after a verified draft PR: `repo`, `pr_number`, `head_sha`, and `github_token_env` |

The agent also returns the `result.json` object as its final response. Artifacts
can be missing on failures. Check the platform job status and the JSON status;
successful job execution alone does not mean a PR was created. These are prompt
contracts, not platform-enforced input/output schemas.

`reviewer-input.json` matches the existing PR reviewer's input shape. Patch mode
does not produce a PR for that reviewer. A caller must explicitly transfer the
plan/input and consume or validate these outputs.

## Runtime and credentials

- Use a Helix execute-agent image with the Codex adapter/runtime, `git`, `gh`,
  `python3`, a shell, and the target repository's build/test dependencies.
  The platform task image bundles Codex; the GitHub CLI and repository-specific
  dependencies still need to be verified in the selected image. A missing tool
  produces a failure; the agent does not install system tools itself.
- The configuration uses local Helix in workspace `mmogallapall` and
  `azure-openai-gpt-5-6-sol`. The local Inference Hub route has passed both
  Responses API checks and complete writer trials.
- The job's platform identity must be allowed to call that inference route.
  Managed jobs supply model authentication through Helix; no model key belongs
  in these files. A local platform needs its own model access configuration.
- Inject a repository-scoped GitHub token as `GH_TOKEN` through an
  AgentEnvironment secret reference. Patch mode needs repository read access.
  Draft mode needs Contents and Pull requests write permissions for the target
  repo, plus any organization authorization required for that token.
- The sample job injects `mmogallapall/gh-writer-token` as `GH_TOKEN`. The secret
  must exist in the target workspace. The local token was verified in the
  patch and draft PR trials below. The sample remains in patch mode until
  explicitly changed.
- Run this configuration only in an isolated execute-job container without host
  home-directory mounts. Codex uses `danger-full-access` inside that container so
  it can write Git metadata, run repository tools, and access GitHub without
  interactive approval. Container isolation is the boundary; prompt publication
  restrictions are not. Use a read-only token for patch-only trials. Do not invoke
  this configuration directly on your laptop or use a host subprocess job profile.

## Register and submit

These commands target local Helix. Registration and an execute job are separate
operations; no persistent agent deployment is needed. Build the shared local
image and launcher below first.

```bash
NHX_BASE_URL=http://localhost:8080 \
  nemo agents register --workspace mmogallapall \
    --name pr-compose-writer \
    --agent-config plugins/nemo-agents/examples/pr-pipeline/writer/agent.yaml
```

`run.example.json` contains the serialized sample input, a 600-second timeout,
and the GitHub secret reference. After changing `input.example.json`, regenerate
the job input without hand-escaping JSON:

```bash
python3 - <<'PY'
import json
from pathlib import Path

directory = Path("plugins/nemo-agents/examples/pr-pipeline/writer")
spec = json.loads((directory / "run.example.json").read_text())
spec["input"] = json.dumps(json.loads((directory / "input.example.json").read_text()))
spec["image"] = "pr-compose:local"
Path("/tmp/pr-writer-run.json").write_text(json.dumps(spec, indent=2) + "\n")
PY

NHX_BASE_URL=http://localhost:8080 \
  nemo agents execute --workspace mmogallapall --profile gpu \
    --spec-file /tmp/pr-writer-run.json
```

The timeout bounds the requested run duration; older task runtimes may take
longer to cancel an in-flight call. The tested hosted default image lacked the
writer tools. For hosted execution, change the model route in `agent.yaml` and select
a compatible published writer image; `pr-compose-writer:local` is available
only to the local Docker daemon.

## Shared local job image

`Dockerfile.local` extends the existing local `pr-compose-reviewer:local` image.
It retains that image's Git and Helix installation and adds the runtimes needed
by all three standalone agents:

| Agent | Harness runtime | Fabric adapter |
| --- | --- | --- |
| Planner | Deep Agents 0.7.40 | 0.6.1 |
| Writer | Codex 0.154.0 | 0.6.1 |
| Reviewer | Pi SDK packages 1.0.3 | npm `nemo-fabric-adapters-pi` 0.6.1 |

It also includes GitHub CLI and the Relay Pi extension from Relay 0.11.0.
Existing Python dependencies remain pinned except `tenacity`, which changes
from 9.2.1 to 9.1.4 to satisfy Deep Agents' Google client dependency. The build
checks dependency consistency. It contains no API tokens and does not copy
the current Helix source checkout.

```bash
docker build -f plugins/nemo-agents/examples/pr-pipeline/writer/Dockerfile.local \
  -t pr-compose:local -t pr-compose-writer:local \
  plugins/nemo-agents/examples/pr-pipeline/writer
```

Both tags refer to the same shared image. New local jobs can select
`pr-compose:local`; the writer tag keeps existing job specs working.

`run.local.json` selects `pr-compose-writer:local` and includes an inline agent
configuration using the local `mmogallapall` GPT route. No local agent
registration is required. It snapshots `agent.yaml`; regenerate the inline
config after changing the canonical instructions. Submit this file to local
Helix with a **Docker-backed** CPU job profile. Create `gh-writer-token` in the
local `mmogallapall` workspace first; hosted secrets are not copied locally. The hosted cluster cannot
pull this laptop-only tag. The base image is ARM64; this build is intended for
local Docker on Apple Silicon, not an AMD64 cluster.

The personal `run.local.json` is ignored by Git. Generate it from the shared
files (adjust the workspace/model route and secret reference for your setup):

```bash
.venv/bin/python - <<'PYLOCAL'
import json
from pathlib import Path
import yaml

root = Path("plugins/nemo-agents/examples/pr-pipeline/writer")
config = yaml.safe_load((root / "agent.yaml").read_text())
spec = json.loads((root / "run.example.json").read_text())
spec["input"] = json.dumps(json.loads((root / "input.example.json").read_text()))
spec["agent"] = {"config_format": "nemo-agents-spec-v1", "config": config}
spec["image"] = "pr-compose-writer:local"
(root / "run.local.json").write_text(json.dumps(spec, indent=2) + "\n")
PYLOCAL
```

The build checks Git, GitHub CLI, the execute-job module, Codex, Deep Agents,
and Pi SDK imports, plus Python dependency compatibility.

An image-local compatibility patch adds only `GH_TOKEN`
to the Codex adapter 0.6.1 inherited environment allowlist. It fails the build
if the adapter version or patch location changes. No secret value is baked in;
Helix must still inject the token at job runtime. Local model access remains
a separate setup requirement.

The planner and reviewer [agent configs](../README.md#local-planner-and-reviewer-jobs)
now directly contain the local gateway routes and Pi 0.6.1 metadata. The reviewer
also selects the Docker profile for its child jobs. Their job specs select this
shared image and reference local secrets. Local trials verified model/tool calls
and reviewer child-job scheduling; see the trial results in the linked README.

### Local launcher prerequisite

The current local setup remaps the default CPU profile to a host subprocess.
Use `--profile gpu` for its Docker-backed **CPU** profile; the writer does not
request GPU resources. This profile expects a Linux launcher binary at
`services/core/jobs/jobs-launcher/jobs-launcher`, relative to the Helix server's
working directory. Without it, Docker skips secret resolution and job log capture.
Build it from the repository checkout before submitting:

```bash
(cd services/core/jobs/jobs-launcher && bash build-manual.sh linux arm64)

NHX_BASE_URL=http://localhost:8080 \
  nemo agents execute --workspace mmogallapall --profile gpu \
    --spec-file plugins/nemo-agents/examples/pr-pipeline/writer/run.local.json
```

The writer image also excludes `.fabric/codex/custom-provider-home/` from artifact
uploads. Codex updates temporary plugin Git files there asynchronously; uploading
that cache previously failed when files disappeared. Writer deliverables remain
in `output_workdir` and are not excluded.

## Validation status

Validated on 2026-11-19:

- Config parsing, Codex settings schema, Fabric translation/planning, and example
  job parsing passed.
- Responses API checks passed for the hosted GPT route and, as verified by the
  caller, the local Inference Hub GPT route.
- The shared ARM64 image passed offline Fabric plan, doctor, startup, readiness,
  and shutdown checks for Deep Agents, Codex, and Pi. All installed Python
  dependencies are compatible, and both writer compatibility patches remain.
  Local planner and multi-model reviewer trials subsequently completed; see
  the [trial results](../README.md#local-planner-and-reviewer-jobs).
- The original ARM64 writer image passed smoke checks for Git, GitHub CLI, Codex, the
  adapter, and the execute-job module. A dummy-token regression check verified
  `GH_TOKEN` forwarding while unrelated secrets remain filtered.
- Local patch trial `nemo-agents-plugin-e-2vztbcjy` completed with `patch_ready`.
- Shared-image patch trial `nemo-agents-plugin-e-2bbhmorg` completed with
  `patch_ready`, the expected README-only patch, and uploaded artifacts.
- Local draft trial `nemo-agents-plugin-e-uupkn3n4` completed with `pr_created`
  and opened [draft PR #1](https://github.com/mmogallapalli/nemo-helix-hackathon/pull/1).
  Only the requested example README changed. YAML/table consistency, relative
  links, and whitespace checks passed. The generated reviewer input matched
  the published PR's head SHA.

The successful end-to-end trials used the earlier local writer image, a Linux ARM64
launcher, local Helix, and local Inference Hub model routing. Hosted execution
with a published writer image has not been verified. To reproduce the local
build on another machine, first supply the `pr-compose-reviewer:local` base
image or a compatible accessible image via `--build-arg REVIEW_IMAGE=...`.
The Dockerfile currently assumes the base's `/workspace/.venv` layout and
Codex adapter 0.6.1; an arbitrary Helix task image is not interchangeable.
