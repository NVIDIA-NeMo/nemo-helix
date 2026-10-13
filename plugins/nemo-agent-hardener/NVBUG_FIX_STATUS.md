<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Hardener release/0.8 bug bundle

## Scope and patch plan

Investigated the eleven supplied bug summaries against local
`origin/release/0.8` (`d2233d54f437011b6013f5068df93a70714bc30e`).
The relevant hardener plugin and typed-client sources at HEAD matched that ref
before these edits. This is code-based triage, not verification of tracker contents.
During the initial investigation, no branch switch, commit, push, service startup,
or dependency installation was performed.
Existing platform OpenAPI/API edits and concurrently appearing unrelated edits were not touched.

Plan implemented: fix input validation at shared SDK/API/worker boundaries;
correct BYO typed-client access and job collection declarations; keep generated Fabric
commands synchronized with port overrides; update provisioning and the shipped Studio label;
add targeted regressions. Auth transport and persisted-job isolation remain drafts below.

## Status

| Bug | Finding and disposition |
| --- | --- |
| 6914043 | **Implemented:** provisioning did not allow prerelease dependencies. Add command-scoped `uv pip install --prerelease allow`; retain configured index/strategy and post-install binary verification. Exact-command test updated. No live install attempted. |
| 6912901 | **Implemented:** `web/src/Nav.tsx` says `Agent Hardening`, but the committed bundle said `Agent Hardener`. Correct only the emitted navigation label and test source/bundle agreement. This is a surgical bundle edit, not a full web rebuild; frontend dependencies are absent. |
| 6851128 | **Implemented:** sanity-check parsed the mitigations file without handling decoding/read failures. Report a concise CLI error for invalid JSON, unreadable/invalid-text files, and non-object JSON. Regression cases cover malformed syntax and non-object roots. This does not introduce a full mitigation-artifact schema. |
| 6851127 | **Implemented:** shared `SynthBenignSpec` accepted empty ids. Require a nonempty string containing a non-whitespace character. Validate local SDK and worker inputs before provisioning/materialization; server request models use the same schema. |
| 6851126 | **Implemented:** no shared guard required a replay when `validate_only` was true. Reject missing/blank `replay_hitlog_fileset` in the shared spec, SDK sanity-check, compile/API validation, and direct worker execution before any attack can start. |
| 6848701 | **Implemented:** `_apply_manifest_overrides` changed `agent.port`, leaving the generated Fabric `start_command` on its old listener port. Centralize generated command construction and rewrite only an exact generated command; preserve arbitrary BYO commands and the stored manifest. |
| 6848698 | **Draft, unresolved:** event ingest already allows human and service principals, but `_run_service_driven` supplies only `AGENT_HARDENER_EVENT_SINK_URL` to the external process. Allowing caller kinds does not authenticate its HTTP callbacks. No supported external EventBus header/token contract was found in this checkout; do not invent an env knob or exempt ingest from authentication. |
| 6848697 | **Implemented:** BYO init called `ctx.sdk.agent_hardener`, a removed namespace on `NemoClient`, instead of `ctx.agent_hardener`. Use the explicit resource; add a real typed-client/MockTransport regression rather than the old namespace-shaped double. |
| 6848696 | **Draft, unresolved:** both collection routers pass `service_name="agent-hardener"` to `job_route_factory`; list isolation adds only `source == agent-hardener`. `job_type` names schemas, not a persisted-job selector. Neither router declares a stable per-job discriminator. |
| 6848694 | **Implemented:** typed endpoints correctly used `/jobs`, but `NemoJobScheduler` derives `/jobs/{name}` unless overridden. Declare `/jobs` for war-game and `/synth-benign/jobs` for synth. Tests compare real scheduler POSTs with registered service OpenAPI paths. |
| 6848688 | **Implemented:** local SDK checked only that some target existed; remote submissions deferred validation. Require exactly one nonblank `config`/`manifest_id` in the shared model, reject locally before launch/upload, and validate SDK submit before network dispatch. |

## Remaining patch designs

### 6848698: authenticated callback transport

1. Verify the provisioned `nvidia-agent-hardener` EventBus supports configurable headers
   and token renewal. Its source/venv was not available at the default local path.
2. If supported, propagate the job's scoped, renewable platform identity through that
   documented contract; keep callback credentials out of stored manifests and job specs.
   Otherwise implement a host-side authenticated relay using the shared typed client,
   with a private local callback boundary and a bounded lifecycle.
3. Exercise auth-enabled principal and service-principal callbacks, token expiry/refresh,
   unauthorized callers, cross-workspace access, and teardown. Do not weaken existing
   `EVENTS_WRITE`/scope requirements to make anonymous callbacks work.

### 6848696: durable job discrimination

1. Persist a server-owned discriminator for war-game versus synth, separate from user
   input; apply the discriminator as an unavoidable AND predicate before pagination.
2. Support existing `source=agent-hardener` rows: establish a deterministic classification
   from the persisted spec/compiled task, then backfill or provide an explicit legacy
   fallback. Do not simply change `service_name` or filter only newly stamped jobs, which
   would hide existing runs. Do not infer type from job-name prefixes.
3. Apply the same ownership/type checks to get/status/cancel/delete/logs/results, so a
   job name from the sibling collection cannot cross the boundary.
4. Test both job types in one workspace, old rows, total/page counts, `$or`/`$not` filters,
   attempted discriminator overrides, and sibling lookup/mutation rejection.

## Validation and commands

Baseline: 392 tests passed. The first post-patch pass exposed the expected setup-command
assertion change; intermediate new-test issues were corrected. Final main suite:
417 tests passed (25 more than baseline), with two pre-existing pytest temporary-directory
permission warnings. Worker-boundary regressions are included in that count.

Main commands:

```bash
git status --short
git rev-parse origin/release/0.8
git diff origin/release/0.8 HEAD --stat -- plugins/nemo-agent-hardener packages/nemo_helix_plugin/src/nemo_helix_plugin/agent_hardener
uv run --frozen pytest plugins/nemo-agent-hardener/tests/unit packages/nemo_helix_plugin/tests/agent_hardener -q
uv run --frozen ruff check plugins/nemo-agent-hardener/src plugins/nemo-agent-hardener/tests/unit packages/nemo_helix_plugin/src/nemo_helix_plugin/agent_hardener/types.py packages/nemo_helix_plugin/tests/agent_hardener/test_endpoints.py
uv run --frozen ruff format --check plugins/nemo-agent-hardener/src plugins/nemo-agent-hardener/tests/unit packages/nemo_helix_plugin/src/nemo_helix_plugin/agent_hardener/types.py packages/nemo_helix_plugin/tests/agent_hardener/test_endpoints.py
uv run --frozen ty check plugins/nemo-agent-hardener/src/nemo_agent_hardener_plugin/agent_resolver.py plugins/nemo-agent-hardener/src/nemo_agent_hardener_plugin/jobs/manifest.py plugins/nemo-agent-hardener/src/nemo_agent_hardener_plugin/jobs/run.py plugins/nemo-agent-hardener/src/nemo_agent_hardener_plugin/jobs/synth_benign.py plugins/nemo-agent-hardener/src/nemo_agent_hardener_plugin/sdk.py packages/nemo_helix_plugin/src/nemo_helix_plugin/agent_hardener/types.py
git diff --check -- plugins/nemo-agent-hardener packages/nemo_helix_plugin/src/nemo_helix_plugin/agent_hardener/types.py packages/nemo_helix_plugin/tests/agent_hardener/test_endpoints.py
```

Ruff formatted four modified files. Targeted lint, format, and production type checks passed.
The hardener OpenAPI spec was regenerated and validated using the canonical generator's
`extract_plugin_openapi_spec`, `apply_schema_fixes`, `fix_ref_not_allowed_errors`, and
`validate_final_specs` functions for this plugin only. Its diff is six validation-schema lines.
The full `make refresh-openapi` target was deliberately avoided because it would overwrite
the user's modified platform spec and regenerate unrelated plugins.

The plugin-only regeneration command was:

```bash
uv run --frozen python -c 'from pathlib import Path; from script.generate_openapi_spec import extract_plugin_openapi_spec, apply_schema_fixes, fix_ref_not_allowed_errors, validate_final_specs; from script.openapi_helper.plugin_config import PluginConfig; plugin = PluginConfig.from_pyproject(Path("plugins/nemo-agent-hardener/pyproject.toml")); assert plugin is not None; result = extract_plugin_openapi_spec(plugin); assert result[1], result; paths = [plugin.output_path()]; apply_schema_fixes(paths); fix_ref_not_allowed_errors(paths); validate_final_specs(paths)'
```

The final type-check pass also included all modified CLI source files and the new
worker-validation, generated-job, and BYO typed-client regression tests.

No authenticated end-to-end run, Docker/Fabric victim launch, real provisioning,
or Studio browser test was performed. These patches are unit-tested candidates,
not confirmation that the corresponding tracker bugs can be closed.

## Release PR preparation

The bundle was copied, with approval, into an isolated worktree on
`agent-hardener-release-fixes/mkornfield`, based on refreshed `origin/release/0.8`
(`83da059fa05419f4ec220e93f1caaaf977f488fa`). Only this bundle was copied;
the original checkout's source edits remain intact.

The affected Python suite passed again: 417 tests. Python and frontend dependencies
were bootstrapped for the isolated worktree, and the repository's copyright hook
added the required headers to the two new files.

Plugin `pnpm build` and `pnpm test` succeeded (32 web tests). Rebuilding produced
thousands of unrelated bundle lines, so that output was discarded in favor of
the surgical navigation-label correction. `pnpm typecheck` remains blocked by
existing imports of nonexistent `@nemo/sdk/generated/platform/api` and
`@nemo/sdk/generated/agents/api`, with consequent implicit-any errors in
unchanged frontend components. These are not fixed in this bundle.

`make TOOLCHAIN=system refresh-openapi` succeeded under the pinned Flox
toolchain in this worktree. The full platform/plugin regeneration left only
the intended six-line hardener schema difference.

The PR remains a draft: authenticated callbacks and job-collection isolation
still require the separate designs described above.
