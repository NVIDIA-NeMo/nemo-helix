<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Candidate record shape

Read this when writing the Step 5 decision. Replace illustrative values with
reviewed evidence; this example is not a ready candidate while required software
availability remains unknown. Validate metadata before building a task:

```bash
python <skill_dir>/scripts/trace_environment.py check-candidate --task-dir <task-dir>
```

This read-only check validates the existing record, prepared safe-trace digest,
evidence step references and retained ground-truth artifacts. It does not edit
values, finalize the workspace, attest privacy review or prove the environment.
Finalization still enforces its full contract.

## Accepted values

Use these exact enum values, not product names, SPDX identifiers or descriptive
synonyms. Put those details in `name`, `version` or `notes` instead.

| Field | Accepted values |
| --- | --- |
| `ground_truth.availability` | `available`, `partial`, `absent`, `unknown` |
| `ground_truth.use` | `none`, `comparison_only`, `verification` |
| Artifact `kind` | `reference_trace`, `expected_output`, `dataset`, `fixture`, `verifier`, `human_feedback`, `other` |
| Software `category` | `library`, `cli`, `desktop_application`, `service`, `hardware`, `other` |
| Software `license` | `open_source`, `proprietary`, `commercial`, `unknown`, `not_applicable` |
| Software `availability` | `available`, `installable`, `unavailable`, `unknown` |
| Provenance `kind` | `atif_step`, `external` |
| `state_basis` | `recorded`, `reconstructed` |

Software `required` is a boolean; `redistributable` is a boolean or null.
`version` is nonempty text or null. Software names must be unique ignoring case.
Every software item and ground-truth artifact needs nonempty `notes` and complete
provenance; use the skill's ATIF-versus-external evidence rules.

Available or partial ground truth must retain at least one hashed artifact and
declare comparison or verification use. Partial ground truth needs an
`absence_reason`; available ground truth sets it to null. Absent or unknown
ground truth needs an empty artifact list, `use: "none"`, and a nonempty reason.

The helper also enforces:

- Every `evidence_steps` list cites real, distinct ATIF `step_id` values from
  the safe trajectory; unknown steps are rejected.
- Provenance is exactly one of `atif_step` (nonempty step IDs, no external
  identifiers) or `external` (empty step IDs, at least one immutable URI,
  revision, or source ID).
- A required software entry with `availability: "unavailable"` blocks
  candidacy.
- `verification_mode` is `"execution"` only.

## Example

Include exactly the keys shown below. Summary fields such as `did_not_work`
belong to `finalize` arguments, not the candidate record.

```json
{
  "schema": "nemo.eval_author.trace_environment_candidate.v2",
  "status": "candidate",
  "decision_basis": "safe_atif_only",
  "instruction": "Observable task instruction without private values",
  "requirements": [
    {"description": "Objectively testable requirement", "evidence_steps": [1, 2]}
  ],
  "verification_mode": "execution",
  "evidence_steps": [1, 2],
  "uncertainties": [],
  "reason_codes": [],
  "ground_truth": {
    "availability": "available",
    "use": "comparison_only",
    "artifacts": [
      {
        "kind": "expected_output",
        "path": "private/ground-truth/expected.json",
        "sha256": "sha256:<64-hex-digest>",
        "provenance": {
          "kind": "external",
          "step_ids": [],
          "uri": "https://example.test/fixture.json",
          "revision": "<immutable-revision>",
          "source_id": null
        },
        "notes": "Expected output attached to the recorded task."
      }
    ],
    "absence_reason": null
  },
  "software_requirements": [
    {
      "name": "ExampleCAD",
      "category": "desktop_application",
      "required": true,
      "version": "2026",
      "license": "proprietary",
      "availability": "unknown",
      "redistributable": false,
      "provenance": {
        "kind": "atif_step",
        "step_ids": [1, 2],
        "uri": null,
        "revision": null,
        "source_id": null
      },
      "notes": "The requested edit and verifier depend on native CAD behavior."
    }
  ]
}
```

## State basis

The optional `state_basis` field declares where the environment's starting
state comes from:

- `"recorded"` (default when the field is absent) — the task reproduces state
  the trace actually recorded.
- `"reconstructed"` — the trace evidences the capability but not the full
  workspace, so the task builds a coherent synthetic world around it.

A reconstructed candidate is a *new task inspired by recorded behavior*, never
a replay claim. Requirements:

- Every capability-defining fact — the request, constraints, correction
  turns, expected outcome — still cites real `step_id` values. Invented state
  never becomes "evidence"; it must be privacy-safe (no real people,
  organizations, identifiers, or file contents from the trace), internally
  consistent, and discoverable by the agent inside the environment.
- The instruction and task must stand alone without the trace. Prefer the
  smallest world in which the recorded behavior is genuinely exercised; do
  not pad with distractors to simulate complexity.
- `instruction.md` and the task README must not represent the world as the
  recorded original. The README `Ground-truth provenance` section names the
  reconstruction.
- `summary.md` and the exported `result.json` carry the basis so dataset
  consumers can filter. Projects that require strictly recorded state filter
  on `state_basis == "recorded"`.
