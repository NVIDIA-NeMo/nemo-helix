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

`client_payload` forwarded trigger→worker and handed to the agent as its job `input`:

```json
{ "trigger": "/helix-review", "repo": "owner/name", "pr_number": 2653,
  "comment_id": 42, "comment_body": "/helix-review", "comment_author": "benmccown",
  "head_sha": "<pinned>", "github_token_env": "GH_TOKEN" }
```

The GitHub token VALUE is injected into the job's secret env under `github_token_env`
(never serialized into the payload). Agents read `GH_TOKEN` to drive the `gh` CLI.

## Onboarding a new GitHub-triggered agent (~3 edits)

1. **Create your agent + a named execute-agent job config** in your Helix workspace.
2. **Add a registry row** in `dispatch-registry.yaml`: `trigger → {workspace, agent, image, allowed_repos}`.
3. **Ensure the trigger workflow is installed** on the repo(s) you want it to fire
   from (the generic `/helix-*` workflow + its `CI_DISPATCH_*` secrets).

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
- First-pass agent (`agents/helix-review/agent.yaml`) only acknowledges on the thread
  ("Let me get started on that for you."); the review fan-out is the next milestone.
- Depends on the pi-adapter gateway-model fix (NVIDIA/NeMo-Fabric) for the agent's
  model to resolve; until merged, the task image installs the patched adapter.
- `pull_request_review_comment` (thread-reply) trigger is a fast-follow: same router,
  a second event type + a reply-handler agent.