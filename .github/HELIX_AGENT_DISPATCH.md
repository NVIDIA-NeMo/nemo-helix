<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub → NeMo Helix agent dispatcher

A reusable, fire-and-forget bridge: a `/helix-*` PR comment triggers a mapped NeMo
Helix `agents.execute` job, which acts on the PR itself (reply / review) using a
forwarded GitHub token. Modeled on the forward-merge recovery pattern.

## The two planes

**Plane 1 — trigger (in the TARGET repo, e.g. nemo-helix):**
`.github/workflows/helix-agent-dispatch.yaml` + `.github/scripts/helix-agent-dispatch.cjs`.
On a PR comment starting with `/helix-`, it reacts 👀 and sends a `repository_dispatch`
(`event_type: helix-agent`) to the worker repo with a PR-context `client_payload`.

**Plane 2 — worker (in the runner-privileged repo, e.g. Platform-Deploy):**
`.github/workflows/helix-agent-run.yaml` + `.github/scripts/dispatch_agent.py` +
`.github/dispatch-registry.yaml`. On the dispatch it mints a repo-scoped GitHub App
token, resolves the trigger in the registry, and POSTs the named `agents.execute`
job into the entry's Helix workspace — then STOPS. The agent posts its own reply.

```
PR comment "/helix-review"
      │  (issue_comment workflow, target repo)
      ▼  react 👀 + repository_dispatch{helix-agent, PR context}
Worker repo (self-hosted runner)
      │  mint GH App token · resolve registry · POST agents.execute
      ▼
NeMo Helix  →  pi agent job  →  gh CLI + token  →  reply on the PR thread
```

## The dispatch contract (envelope)

The trigger emits this `client_payload` (`repository_dispatch`). Revision fields are
emitted `null` and the worker backfills them from the PR:

```json
{ "trigger": "/helix-review", "repo": "owner/name", "pr_number": 2653,
  "comment_id": 42, "comment_body": "/helix-review", "comment_author": "benmccown",
  "head_sha": null, "base_ref": null, "head_ref": null }
```

The worker then builds the agent's job `input` from this envelope: it resolves
`head_sha` (and refs) from the PR, and adds `github_token_env` (the env-var NAME the
token lands under). The token VALUE is injected into the job's secret env under that
name — never serialized into the payload. Agents read `GH_TOKEN` to drive the `gh` CLI.

## Onboarding a new GitHub-triggered agent

Follow these in order. Steps 1–3 are the Helix-cluster legwork (done once, out of
band); step 4 is the ~1-line PR to this repo; step 5 installs the trigger.

**1. Create the GitHub-token secret in your Helix workspace** (once per workspace).
The agent reads this to drive the `gh` CLI. The registry entry references it by name
(`github_secret`, default `github-token`):
```bash
nemo secrets create github-token --workspace <workspace> --value "$(gh auth token)"
```

**2. Author the agent config** (`agent.yaml`). For a pi agent on a gateway-served
model, declare the harness + model, and put Pi catalog metadata the gateway can't
express under the model's `extensions` (api is required; declared fields like
`max_tokens` stay top-level):
```yaml
config_format: nemo-agents-spec-v1
name: <your-agent>
default_harness: pi
harnesses:
  pi: { kind: nvidia.fabric.pi }
models:
  default:
    provider: nvidia
    model: <gateway/model-id>
    api_key_env: NVIDIA_API_KEY
    base_url: https://<helix>/apis/inference-gateway/v2/workspaces/<ws>/openai/-/v1
    max_tokens: 8192
    extensions: { api: openai-completions, context_window: 200000 }
instructions:
  system: { content: "…what the agent should do with the PR context…" }
```
The agent's job `input` is a JSON blob of PR context
(`{trigger, repo, pr_number, comment_author, head_sha, github_token_env, …}`) — write
the system prompt to consume it and act via `gh`.

**3. Create the `agent` entity** in the workspace (type `agent`, looked up by
`{workspace, name}`). The dispatcher no-ops gracefully until this exists:
```bash
nemo agents create --agent agent.yaml --workspace <workspace>
```

**4. Add a registry row** in `dispatch-registry.yaml` (this repo) — see the
`/helix-review` entry below as a template:
```yaml
  - trigger: "/helix-<name>"
    description: "<what it does>"
    workspace: <workspace>          # must match where you created the agent
    agent: <your-agent>             # the entity name from step 3
    image: <published task image>   # nemo agents package output (NGC ref)
    allowed_repos: [ "<owner>/<repo>" ]
```

**5. Install the trigger workflow** on the target repo(s): the generic
`.github/workflows/helix-agent-dispatch.yaml` + `.github/scripts/helix-agent-dispatch.cjs`
(from the nemo-helix PR), plus the `CI_DISPATCH_TOKEN` / `CI_DISPATCH_REPO` secrets.

**Test it:** comment `/helix-<name>` on a PR in an allowed repo (as a repo
collaborator — the trigger is author-gated). You should see a 👀 reaction, then the
agent's reply on the thread. If nothing happens, check the worker repo's Actions run:
a missing agent entity logs a `::notice::` and skips (not an error).

## Secrets & vars

**Target repo (trigger side):**
| Secret | Purpose |
| --- | --- |
| `CI_DISPATCH_TOKEN` | token allowed to `repository_dispatch` to the worker repo |
| `CI_DISPATCH_REPO` | `owner/worker-repo` destination |

**Worker repo (dispatch side):**
| Secret / var | Purpose |
| --- | --- |
| `AIRE_OPS_APP_PRIVATE_KEY` | `aire-ops` GitHub App key (App ID 4809653) — mints repo-scoped tokens |
| `INTERNAL_AGENTS_NHX_ACCESS_TOKEN` | Helix API bearer (dispatcher service identity, cross-workspace submit) |

Plus, per workspace, a Helix secret holding the GitHub token the agent reads
(referenced by the registry entry's `github_secret`, default `github-token`).

## Status / open items

- Agent `image` ref in the registry is a TODO (set to the published helix-review image).
- The agent config is a Helix `agent` entity created in the target workspace (NOT a
  file in this repo); the dispatcher skips cleanly until that entity exists.
- Depends on the pi-adapter gateway-model fix (NVIDIA/NeMo-Fabric) for the agent's
  model to resolve; until merged, the task image installs the patched adapter.
- `pull_request_review_comment` (thread-reply) trigger is a fast-follow: same router,
  a second event type + a reply-handler agent.