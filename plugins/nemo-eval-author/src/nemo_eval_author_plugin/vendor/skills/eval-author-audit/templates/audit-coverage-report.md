<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Audit coverage report: <agent name>

<!-- Replace authoring prompts with current findings. Remove these comments in
the produced report. Link only artifacts that exist; explain missing inputs. -->

<Lead with the main finding and whether this is a specification-only, partially
measured, or measured audit. State the scope to which the finding applies.>

## Scope and evidence

- **Audit specification:** <link labeled Audit specification to audit.md>
- **Ethos:** <link to the applicable ETHOS.md>
- **Evaluations inspected:** <suite and case links, or the specific limitation>
- **Run evidence:** <selected source and runs, or why measurement is unavailable>
- **Coverage measurements (JSON):** <link to audit-coverage-report.json when it exists, otherwise explain why unavailable>
- **Measurement methods and evidence:** <methods run and links to applicable measurement details>
- **Exclusions:** <failed, omitted, or stale inputs and their effect on findings>

## Coverage summary

<!-- Populate all three kinds from the specification and applicable measurements.
Use “Unmeasured” instead of numeric measured counts when no method ran for a kind.
A kind with no declared items is “No items declared”, not 100% covered. -->

| Kind | Declared items | Measured covered | Measured but not demonstrated | Unmeasured |
|---|---|---|---|---|
| Tools | <total> | <count or Unmeasured> | <count or Unmeasured> | <count> |
| Capabilities | <total> | <count or Unmeasured> | <count or Unmeasured> | <count> |
| Failure cases | <total> | <count or Unmeasured> | <count or Unmeasured> | <count> |

## Intended coverage

**Scope review:** <Awaiting user review / Agreed with the user; note open decisions>

This is the complete list of what the audit intends to check and the evidence
each check requires. Test mappings describe what existing evaluations intend to
exercise. Measurement status describes what selected recorded runs demonstrate.

<!-- Include every current audit item, optionally grouped by kind. Use per-item
lists if the table becomes too wide. Explain intended behavior and required
evidence in plain language, keeping material criteria from audit.md. For failure
cases, include the trigger and expected safe response. An uninspected test
mapping is “Not inspected”; absence is bounded to the tests actually read. -->

| Audit item | Kind | Intended check and required evidence | Existing tests | Measurement status | Observed evidence and limits |
|---|---|---|---|---|---|
| <stable name> | <kind> | <behavior/trigger, expected result, and evidence needed to demonstrate it> | <case/verifier links or mapping limitation> | <Covered / Measured but not demonstrated / Unmeasured> | <task/run, method, evidence links, or specific missing evidence> |

## Findings and limitations

<Explain the material gaps and observed failures with evidence links. Distinguish
missing scenarios from missing evidence or methods. Include fixture/verifier
limitations and unresolved causes. Do not infer overall quality from coverage.>

## Next actions and open questions

<Recommend the next useful action and explain why. For an existing suite, lead
with proposing new or improved eval tasks for the main gaps, naming the behaviors
to target. Explain when evidence collection, an agent fix, or first-eval creation
should come first. Identify decisions to review with the user. Detailed ranked
task proposals belong to the subsequent proposal workflow.>
