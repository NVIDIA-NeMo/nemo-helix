<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub → NeMo Helix agent dispatcher

A reusable bridge: a `/helix-*` PR comment triggers a mapped NeMo Helix
`agents.execute` job that REVIEWS the PR. The agent holds only a read-only token and
returns a typed set of requested actions; the trusted worker validates and performs
those actions with a separate write token. Modeled on the forward-merge recovery
pattern (the agent returns data; the worker does the writes).

## The two planes

**Plane 1 — trigger (in the TARGET repo, e.g. nemo-helix):**
`.github/workflows/helix-agent-dispatch.yaml` + `.github/scripts/helix-agent-dispatch.cjs`.
On a PR comment starting with `/helix-`, it reacts 👀 and sends a `repository_dispatch`
(`event_type: helix-agent`) to the worker repo with a PR-context `client_payload`.

**Plane 2 — worker (in the runner-privileged repo, e.g. Platform-Deploy):**
`.github/workflows/helix-agent-run.yaml` + `.github/scripts/dispatch_agent.py` +
`.github/dispatch-registry.yaml`. On the dispatch it mints TWO repo-scoped `aire-ops`
tokens — a READ-only one forwarded to the agent, and a WRITE one kept in the worker —
resolves the trigger in the registry, submits the `agents.execute` job, then POLLS it
to completion, downloads the agent's action envelope, validates it, and executes the
requested actions against the triggering PR with the write token. The write token
NEVER reaches the agent.

```
PR comment "/helix-review"
      │  (issue_comment workflow, target repo)
      ▼  react 👀 + repository_dispatch{helix-agent, PR context}
Worker repo (runner)
      │  mint READ token (→ agent) + WRITE token (stays here)
      │  resolve registry · submit agents.execute · poll to done
      ▼
NeMo Helix  →  pi agent reads the PR with its READ token
      │  agent writes review-actions.json (typed action envelope)
      ▼
Worker  →  validate envelope · execute actions on the PR with the WRITE token
```

## Why two tokens (the security model)

A GitHub App token can't be scoped below a whole repo, so a `pull_requests: write`
token held by the agent could act on ANY PR in the repo — a prompt-injected agent
could abuse it, and it lives ~1h. So the agent gets only a READ token (on a public
repo that grants nothing beyond anonymous read) and returns its review as data; the
WRITE token stays in the trusted worker, which performs only a closed set of actions
scoped to the triggering PR. (Interim stopgap ahead of OpenShell-managed policy.)

## The dispatch contract (envelope)

The trigger emits this `client_payload` (`repository_dispatch`). Revision fields are
emitted `null` and the worker backfills them from the PR:

```json
{ "trigger": "/helix-review", "repo": "owner/name", "pr_number": 2653,
  "comment_id": 42, "comment_body": "/helix-review", "comment_author": "benmccown",
  "head_sha": null, "base_ref": null, "head_ref": null }
```

The worker builds the agent's job `input` from this envelope (resolving `head_sha`)
and injects the READ-only token into the job's secret env under `github_token_env`.

### The agent's return: a typed action envelope

The agent does NOT call GitHub to post. It writes `review-actions.json` to its output
workdir — a CLOSED, typed set of actions the worker validates (unknown types / extra
fields are rejected) and executes against the triggering PR:

```json
{ "summary": "one-line summary",
  "actions": [
    { "type": "post_review", "event": "COMMENT",
      "body": "…", "comments": [ { "path": "f.py", "line": 3, "body": "…", "side": "RIGHT" } ] },
    { "type": "reply_comment", "comment_id": 42, "body": "…" },
    { "type": "resolve_thread", "thread_id": "PRRT_…" }
  ] }
```

`event` is one of `COMMENT` / `APPROVE` / `REQUEST_CHANGES`. Adding a capability means
adding one action type + its executor branch — nothing else the agent emits is honored.

## Onboarding a new GitHub-triggered agent

Follow these in order. Steps 1–2 are the Helix-cluster legwork (done once, out of
band); step 3 is the ~1-line PR to the WORKER repo's registry; step 4 installs the
trigger. You do NOT create a GitHub-token secret: the dispatcher mints the per-dispatch
READ token and upserts it into a per-dispatch secret the agent reads under
`github_token_env` (default `GH_TOKEN`); the WRITE token stays in the worker.

**1. Author the agent config** (`agent.yaml`). For a pi agent on a gateway-served
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
(`{trigger, repo, pr_number, comment_author, head_sha, github_token_env, …}`). Write
the system prompt to READ the PR with its token and emit `review-actions.json` (the
typed envelope above) — NOT to post directly.

**2. Create the `agent` entity** in the workspace (type `agent`, looked up by
`{workspace, name}`). The dispatcher no-ops gracefully until this exists:
```bash
nemo agents create --agent agent.yaml --workspace <workspace>
```

**3. Add a registry row** in the WORKER repo's `.github/dispatch-registry.yaml`
(the dispatcher only reads the registry there, not the target repo) — see the
`/helix-review` entry below as a template:
```yaml
  - trigger: "/helix-<name>"
    description: "<what it does>"
    workspace: <workspace>          # must match where you created the agent
    agent: <your-agent>             # the entity name from step 2
    image: <published task image>   # nemo agents package output (NGC ref)
    allowed_repos: [ "<owner>/<repo>" ]
```

**4. Install the trigger workflow** on the target repo(s): the generic
`.github/workflows/helix-agent-dispatch.yaml` + `.github/scripts/helix-agent-dispatch.cjs`
(from the nemo-helix PR), plus the `CI_DISPATCH_TOKEN` / `CI_DISPATCH_REPO` secrets.
`CI_DISPATCH_TOKEN` needs `issues: write` on the TARGET repo for the 👀 reaction AND
`repository_dispatch` to the worker repo; if it lacks the former the reaction is
silently skipped (caught) while dispatch still succeeds.

**Test it:** comment `/helix-<name>` on a PR in an allowed repo (as an org
OWNER/MEMBER — the trigger is author-gated). You should see a 👀 reaction, then the
agent's review on the PR. If nothing happens, check the worker repo's Actions run:
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

The agent's GitHub token is NOT a pre-created secret: the dispatcher mints a
repo-scoped READ-only `aire-ops` token per dispatch and upserts it into a per-dispatch
Helix secret (named `dispatch-…`), which the agent reads via its AgentEnvironment. The
WRITE token is minted in the same job but never leaves the worker.

## Status / open items

- Agent `image` ref in the registry is a TODO (set to the published helix-review image).
- The agent config is a Helix `agent` entity created in the target workspace (NOT a
  file in this repo); the dispatcher skips cleanly until that entity exists.
- The pi-adapter gateway-model fix (NVIDIA/NeMo-Fabric) that lets the agent's model
  resolve is merged; the task image installs it from the released adapter.
- Per-dispatch READ-token secrets accumulate; a cleanup/GC sweep is a tracked follow-up.
- `pull_request_review_comment` (thread-reply) trigger is a fast-follow: same router,
  a second event type + a reply-handler agent.