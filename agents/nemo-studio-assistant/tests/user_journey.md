<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Studio Assistant user journey

This is the walk-through for checking the Studio Assistant end to end. Run it
after the platform is up and the assistant is deployed (see
[smoke_test.md](smoke_test.md)). The API half is scripted. The UI half is a
checklist for a person.

## Before you start

- The platform is running with `NEMO_DEFAULT_MODEL` exported in the shell that
  ran `nemo services run`. The topic gate uses that model.
- `agents/nemo-studio-assistant/scripts/deploy-local.sh` finished with a
  successful invocation. The verified model is `azure/openai/gpt-5.4`
  (`default/azure-openai-gpt-5-4`). `nvidia/nvidia/nemotron-3-super-v3` also
  completes the journey, but turns take much longer. On `openai/gpt-oss-120b`,
  follow-up turns often end with an empty answer under the Deep Agents harness.
- Studio is open at `http://localhost:8080/studio`.

## Scripted journey

```bash
uv run python agents/nemo-studio-assistant/scripts/journey.py
```

The script opens one Studio assistant session and sends each message below in
turn, with the Studio page path the browser would send. It answers pickers
with their first option and declines every approval prompt. It prints one PASS
or FAIL line per step, with the tools the assistant used, and exits non-zero
on any failure. Pass `--verbose` to print each answer in full.

| Step | Page | Message | Passes when |
|------|------|---------|-------------|
| Greeting | Dashboard | Hi! What can you help me with here? | The answer is not the off-topic refusal. |
| Page context | Virtual Models | How do I add a guardrail here? | The answer talks about guardrails on Virtual Models. |
| Docs search | Dashboard | How does skill optimization work? | The assistant calls `search_docs`. |
| Platform read | Dashboard | List my workspaces. | The assistant calls `nemo_api` and names the workspace. |
| Short follow-up | Dashboard | yes | The topic gate lets it through. |
| Off-topic | Dashboard | How do I bake a cheesecake? | The fixed refusal comes back with no agent run. |
| Declined write | Dashboard | Create a new workspace named journey-smoke. | An approval prompt appears, and after declining, `journey-smoke` does not exist. |

## In Studio

Walk through the same steps in the browser and check what the script cannot:

1. Open the assistant from the top bar on the dashboard and send the greeting.
   The answer streams in, and no "summary" accordion hides a short answer.
2. Go to **Virtual Models** and ask "How do I add a guardrail here?" The
   answer refers to the Virtual Models page, includes a working Studio link,
   and shows no tool-call syntax such as `nemo_api(...)`.
3. Ask "How does skill optimization work?" A "Searched docs for ..." step
   appears above the answer, and the answer names the doc it used.
4. Ask "List my workspaces." A "Called list workspaces" step appears before
   the answer.
5. Ask "How do I bake a cheesecake?" The refusal appears almost immediately.
   Then ask a NeMo question and check that the assistant does not mention
   the cheesecake.
6. Ask "Create a new workspace named journey-smoke." An approval card
   appears. Decline it. The assistant says nothing changed and does not
   ask again, and the workspace list does not include `journey-smoke`.
7. Open a page with a query string, such as a Virtual Model's chat tab, and
   ask "What am I looking at?" The answer names that Virtual Model.
8. Reload the page and reopen the conversation from history. The earlier
   turns are all there, including the refusal.

## Cleanup

The journey creates nothing when the write is declined. If you approved it
instead, remove the workspace:

```bash
nemo workspaces delete journey-smoke --yes
```
