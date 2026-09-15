<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Review and publish an exact product

Every export requires a separate publication review, including `no_candidate`.
The earlier trace review and `--human-reviewed` flag do not authorize publication.
First prepare an owner-private preview of the exact files that will be exported:

```bash
python <skill_dir>/scripts/trace_environment.py prepare-publication \
  --task-dir <task-dir>
```

Inspect every file in the returned `preview_dir`: all candidate notes,
uncertainties, software and ground-truth provenance, the generated task and
`reproducibility.json` when present, and the derived `result.json`. Check paths as well as
contents for private identifiers, internal locators, proprietary material, and
other information not authorized for publication. The preview contains only the
publication whitelist; source evidence and review notes are not part of it.
Do not approve a product that still contains private details. Correct the source
artifacts in a new workspace if finalization prevents revision, regenerate any
affected proof, finalize, and prepare a new preview. Never edit just the preview.

After reviewing it, supply the exact returned digest and directory:

```bash
python <skill_dir>/scripts/trace_environment.py review-publication \
  --task-dir <task-dir> \
  --preview-dir <preview-dir> \
  --sha256 <reviewed-sha256> \
  --reviewer-kind <agent|human> \
  --note "<scope reviewed and disposition, without private values>"
```

This records a contextual attestation, not an automatic safety verdict or a
cryptographic proof that a reviewer read the files. It is bound to every exported
byte, path, and entry permission; the private preview root itself stays owner-only.
Staging and export copy file contents and normalized executable bits only, not
source extended attributes or other source filesystem metadata. Directory
metadata is not copied either; keep private context out of publication contents.
Re-review a newly prepared preview after any publication change. Superseded
previews remain private. `check` establishes workspace consistency, not
publication approval. Historical workspaces also require this new review before
export; existing trace-review records are not upgraded into publication approval.

Then export the reviewed product to a new, nonexistent destination:

```bash
python <skill_dir>/scripts/trace_environment.py export \
  --task-dir <task-dir> \
  --output-dir <dataset-product-dir>
```

The command runs `check`, verifies the publication review against a newly staged
snapshot, and copies only `candidate.json`, the generalized `task/` and
`reproducibility.json` when present, and a declassified `result.json`.
It never copies source, canonical, safe, privacy-audit, ground-truth, validation,
or Harbor job files. It preserves task-file executable bits while making the
export readable, so the published task matches its task-tree digest.
Public results report image pinning separately from unverified dependency
closure and distinguish distinct jobs from unverified container freshness.
Publication previews and the versioned review receipt remain private. The public
product schema is unchanged; the new receipt does not imply that older products
were publication-reviewed.
