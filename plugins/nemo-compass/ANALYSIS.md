<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compass analysis inputs and upgrades

The Helix plugin calls Compass's public analysis pipeline. Compass selects and
orders signals, validates candidate problems, compiles insights, resolves evidence,
and reviews confidence. Helix supplies traces, model access, optional source and
embedding inputs, and persistence.

## Supply source and embeddings

Upload the agent's source as a Helix fileset, then pass its reference and an
embedding Model reference when submitting analysis:

```bash
nemo compass analysis-runs create \
  --agent research-agent \
  --workspace default \
  --ethos ETHOS.md \
  --source-fileset default/research-agent-source \
  --embedding-model default/qwen3-embedding-8b \
  --wait
```

`--source-fileset` uses the generic execute job's working-directory staging.
The adapter passes that staged directory to Compass for candidate validation and
confidence review. The source must describe the agent that produced the traces;
Helix does not infer a matching source revision from telemetry. The job retains
its input workdir for inspection. An explicitly configured source that cannot be
staged or resolved fails the job.

`--embedding-model` must reference a Platform model serving
**Qwen/Qwen3-Embedding-8B with 4096 dimensions**. The sentiment classifier is
trained for that representation; another embedding model with the same vector
width is not interchangeable. Helix routes requests through its authenticated
inference gateway and refreshes credentials per request. Compass owns text
preprocessing, batching, vector normalization, and classification. Credentials
are not saved in analysis options.

Both inputs are optional. Without source, Compass records why code-based
confidence review could not run; candidate validation can still use an Ethos
document. Without an embedding model or a Compass-configured embedding backend,
user sentiment is skipped with a reason in the full analysis result. A configured
but inaccessible embedding model is an error, not a successful skipped signal.

The API and Python SDK expose the same `source_fileset` and `embedding_model`
fields. Periodic analysis stores them when analysis is enabled and uses those
references for subsequent runs. Re-enable analysis to change or clear them.

For a manually authored Fabric agent, `harness.settings.code_base` names a
relative directory within `RuntimeContext.environment.workspace`, and
`models.embedding` supplies the Platform embedding model reference. Stage source
with the execute job's `workdir.base_workdir` or `workdir.artifact_mounts`.
The adapter rejects paths outside the workspace, including symlink escapes.
It does not assume that an otherwise empty Fabric workspace contains source.

## Review an upgrade

Keep saved analysis options limited to explicit overrides. Compass validates
those options and supplies defaults from the installed version. The complete
analysis result records effective options, the package version, stage outcomes,
and diagnostics; Helix stores it independently from its Insight entity projection.

Before merging a Compass dependency upgrade:

1. Run the plugin's tests against the intended Compass build. In particular,
   exercise optional inputs, authenticated model access, complete result
   persistence, and the policy for user-edited Insight fields.
2. Run the standalone CLI and Helix over equivalent canonical trace snapshots,
   prior insights, analysis options, model configurations, Ethos, source, and
   embedding backend. Check which stages ran or were skipped and whether the
   complete outputs survived persistence. Generated insight prose need not be
   identical across model calls.
3. Check new defaults and resource requirements in the Compass release. Signals
   that use existing dependencies should require no new Helix orchestration.
   New external resources, breaking API changes, and new Studio presentation
   still need explicit integration work.

The source and embedding references belong to Helix configuration. Signal names,
thresholds, and stage ordering belong to Compass; adding them to another plugin
registry would restore the maintenance burden the shared API removes.
