<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Additional skill evaluation metrics

This extension covers the remaining tier 1 and tier 3 candidates from the
SkillEvaluator review. It builds on the focused six-metric SDK change. The
implementations are in `nhx_evals_sdk.metrics`, registered in `MetricsUnion`,
and exported through the SDK's lazy public API. The evals plugin's existing
inline configuration path bundles and reconstructs them.

There are 24 additional concrete metrics and three judge rubric factories.
`skill_outcome_metric` combines the outcome rubrics and the original shared
assertions into one judge. Examples contain fixtures and configuration only.
No additional dependencies are introduced by this extension.

The reference is SkillEvaluator 1.13.2, revision
`6b91545511337c22b7ee88408dfad6ad106a1348`, especially its tier 1 validators,
`constants.py:RUBRIC_CRITERIA`, tier 3 deterministic checks,
`plugin_signals.py`, and Harbor answer-accuracy criteria. These are adaptations
of its useful signals. Scores and deductions are defined below; they do not
claim numerical parity with SkillEvaluator's overall score.

## Tier 1 coverage

| Candidate | SDK implementation | Outputs and interpretation |
| --- | --- | --- |
| Static correctness | `SkillStaticCorrectnessMetric` | Equal-weight mean of portable structure, examples, and local-reference checks; `score` 0–1 and defect count |
| Description quality | `SkillDescriptionQualityMetric` | Equal-weight advisory checks for descriptive length, trigger context, and specific language; `score` 0–1 |
| Reliability readiness | `SkillReliabilityReadinessMetric` | Guidance headings for prerequisites, errors, and limitations; optional required inputs/output-contract sections; equal-weight `score` 0–1 |
| Nine-criterion documentation rubric | `skill_documentation_rubric_metric(model)` | Nine integer 0–10 scores, with importance-weighted `view.documentation_rubric` on the same 0–10 scale |
| Script lint | `SkillScriptQualityMetric` | Python scripts checked and advisory defects; AST parsing only |
| SkillSpector/security | `SkillSecurityScanMetric` | Scanner findings by severity and file, completion, policy cleanliness |
| Sensitive data / secrets | `SkillSensitiveDataMetric` | Redacted credential/PII report metadata with the same coverage accounting |
| Unicode smuggling | `SkillUnicodeSafetyMetric` | Hidden-control indicators, files checked, and `clean`; explicit document/package scope |
| Code risk | `SkillCodeRiskMetric` | Bandit/Semgrep/equivalent report findings, severity, and coverage |
| Bundled tests / coverage | `SkillBundledTestMetric` | Test counts, completed-suite pass rate, reported coverage and coverage-threshold result |
| License compliance | `SkillLicenseMetric` | Auditor findings, severity, coverage, and policy cleanliness |
| Dependency compliance | `SkillDependencyComplianceMetric` | Dependency-license, pinning, banned-package, and vulnerability report findings |
| Package/plugin hygiene, MCP configuration, privileges, hooks | `SkillPackageConfigurationRiskMetric` | Adapter-supplied package/configuration audit findings with explicit scanner identity and completion |

The documentation rubric preserves nine source criteria: description clarity,
instruction clarity, example quality, documentation completeness, scope,
professional tone, trigger simulation, workflow completeness, and error
handling. Their weights are respectively `3, 3, 3, 2, 2, 1, 3, 3, 2`.
Each criterion uses eleven rubric labels (`"0"` through `"10"`) mapped to
numeric scores. Unrecognized labels stay unmeasured, including when the judge
provider does not support structured output.
Trigger simulation is a documentation judgment, separate from observed routing.

Readiness rules are transparent English-language heuristics. Description checks
look for at least 20 characters by default; whole-word trigger language
(`use`, `when`, `for`, `helps`, `allows`); and no generic placeholder words
(`something`, `things`, `stuff`, `various`, `general`). Headings and examples
establish the presence of guidance, not its correctness. Use the judge and live
outcomes to assess meaning, especially for non-English documents.

Script checks cover syntax, function structure, shebang, an input-validation
signal, Usage/Arguments/Output/Exit codes in the module docstring, magic numbers,
deep nesting, nonstandard literal exit codes, and obvious diagnostics sent to
stdout. Each rule contributes at most one finding per file. Only original
`scripts/**/*.py` source is inspected; missing source leaves the metric
unmeasured. Other languages require a scanner report adapter.

Unicode checks include tag characters, bidi controls, zero-width characters,
invisible controls, and variation selectors; confusable spaces are opt-in.
A leading BOM is allowed. These are indicators for review, including potentially
legitimate text and emoji, rather than a malware verdict. Package scope requires
all inventoried text files with supported suffixes (`md`, `mdc`, `py`, `sh`,
`bash`, `toml`, `yaml`, `yml`, `json`, `txt`, `js`, `ts`, `tsx`). Empty text scope
is unmeasured. Diagnostics contain locations and codepoints, without excerpts.

## Tier 3 coverage

| Candidate | SDK implementation | Outputs and interpretation |
| --- | --- | --- |
| Five-criterion answer accuracy | `answer_accuracy_metric(model)` | Correct actions, factual accuracy, request addressed, actionable result, required format; binary criteria and mean/all-met views |
| Task/goal success | `task_success_metric(model)` | Answer-level goal achievement and corresponding view; external state verification remains a runner reward |
| Overactivation | `SkillOveractivationMetric` | Observed activation on an authored negative case; complements the original activation metric |
| Skill routing | `SkillRoutingMetric` | Target, acceptable alternate, wrong, and unknown-component calls; completed-trace routing result |
| Expected script invocation | `ExpectedScriptInvocationMetric` | Invocation flag/count, known success/failure, and unknown outcomes |
| Workflow compliance | `WorkflowComplianceMetric` | Authored precedence checks satisfied and equal-weight `score` 0–1 |
| Error recovery | `ErrorRecoveryMetric` | Failed calls, retries, recovered/unresolved calls, explicit fault ownership, and recovery rate |
| Tool productivity | `ToolProductivityMetric` | Productive/wasted/neutral/unclassified calls and classified productivity ratio |
| Tool-selection precision/recall/F1 | `ToolSelectionMetric` | Distinct called identities, expected patterns matched, unexpected/decoy tools, and precision/recall/F1 |
| Argument compliance | `ToolArgumentComplianceMetric` | Passed/failed call-rule pairs and unmatched rules using authored JSON Schemas |
| Tool execution outcomes (initially MCP) | `ToolExecutionOutcomeMetric` | Known successes/failures, unknown outcomes, known-outcome success rate; default pattern `MCP:*` |
| Component order, handoffs, conflicts | `ComponentCoordinationMetric` | Separate probe counts and an equal-weight coordination score |
| Component usage coverage | `ComponentUsageMetric` | Declared, available, exercised, never activated, unavailable, unknown, and undeclared components; coverage and multi-component flag |
| Agent safety and canaries | `AgentSafetyMetric` | Trial-scoped safety report findings and deduplicated `canary_leaks` |

All ratios are 0–1. Counts have no implied pass threshold. Missing inputs return
no outputs with a diagnostic. Invalid evidence, including another revision's
digest or another trial's ID, raises and uses Helix's metric-failure accounting.
Every trace metric exposes `complete` except overactivation, which uses the
original explicit activation observation.

Important scoring details:

- Tool/component patterns use case-sensitive `fnmatch` on `component`, falling
  back to the normalized tool identity when component is absent. Routing uses
  exact `Skill:` identities. Its expected and acceptable skill values include
  that prefix. Unknown component attribution prevents a final routing result.
- A complete trace means all relevant calls were captured, including skill
  activation events. Every `after` event needs at least one strictly earlier
  `before` event. Both patterns must occur; a missing workflow is not a pass.
  Same-step calls are unordered. These rules test invocation order, not success
  of prerequisite work; execution outcomes report that separately.
- Script paths match exactly. Invocation can be established in a partial trace;
  absence requires a complete trace. A failed script still counts as invoked.
- Recovery follows explicit `retry_of` chains. A failed call is recovered when
  a descendant retry has known success. Each failed attempt is one recovery
  opportunity. The final rate requires a complete trace, known outcomes, and at
  least one failure. Attribution is supplied explicitly, never guessed from text.
- Productivity labels are producer/judge evidence. Its score is
  `productive / (productive + wasted)`, excluding neutral calls. A complete,
  fully classified trace and a nonzero denominator are required.
- Selection precision is accepted distinct identities / all called identities;
  recall is expected patterns matched / expected patterns. Acceptable extras
  contribute to precision without satisfying a missing expected pattern. Decoys
  are always unexpected, even if another pattern matches them. F1 is the harmonic
  mean. No calls yields zero for a complete trace, and no ratio for a partial one.
- Argument rules use self-contained JSON Schema (required fields, types,
  constants, patterns, etc.). External references and `$id` changes are rejected;
  no schema resources are fetched. All matching call/rule pairs are scored. No
  matching calls means no ratio. An unmatched rule remains visible and is covered
  by selection expectations rather than being counted as a compliant call.
- Outcome success rate includes known outcomes only, and can describe a partial
  trace. Interpret it with `unknown` and `complete` beside it.
- Value handoffs require a successful producer, a strictly later consumer, and
  a string of at least eight characters present in the producer observation and
  consumer arguments. The value must be absent from the task instruction and
  the producer-supplied initial context. Missing initial-context coverage or
  unknown producer outcomes/observations make that probe unmeasured. Artifact probes require an observed successful write
  followed by a matching observed read. These are evidence of transfer, not a
  causal proof that coordination improved task quality.
- Conflicts are authored pairs that must not both activate. Coordination's
  aggregate requires a complete trace and all probe inputs. Component usage
  excludes declared unavailable components from coverage while reporting them;
  unknown attribution prevents negative coverage conclusions. Multi-component
  use can be established positively in a partial trace.

## Experimental adapter interfaces

Adapters supply these inputs alongside the first PR's `SkillSnapshot`:

| Location | Public model | Producer responsibility |
| --- | --- | --- |
| `reference.skill_contents` | `SkillContentBundle` | Original package text, canonical relative paths, and artifact digest matching the snapshot |
| `reference.skill_scans[scan_key]` | `SkillScanReport` | Actual scanner/version, redacted findings, subject digest, completion and coverage |
| `reference.skill_test_report` | `SkillTestReport` | Isolated test execution results bound to the artifact digest |
| `trial.metadata.skill_tool_trace` | `SkillToolTrace` | Correlated ordered calls bound to `trial.id`, component/script attribution, explicit outcomes and completeness |
| `trial.metadata.skill_scans[scan_key]` | `SkillScanReport` | Runtime safety verifier report bound to the trial ID |

The default artifact scan keys are `security`, `sensitive_data`, `code_risk`,
`license`, `dependencies`, and `package_configuration`. Runtime safety uses
`safety`. `scan_key` is configurable. Findings contain rule/category/severity
and optional file/line; raw secret values and source excerpts are outside the
report contract. Repeated findings are deduplicated for counts. A canary leak
uses category `canary_exfiltration`.

`SkillScanReport.status` is `completed`, `partial`, `failed`, or `unavailable`.
Only completed reports emit `clean`, using configurable failure severities
(default critical/high). Thus `clean` means no findings at the configured
failure levels, not no findings at all. If expected file coverage is provided,
a completed report must cover every expected file. Completion remains a
producer assertion; it is not inferred from an empty findings list.

Bundled-test pass rate is `passed / (passed + failed + errors)` for a completed,
nonempty measured suite. Skipped tests are reported separately. Coverage must be
explicitly measured; missing coverage is not zero, and a partial or skipped-only
suite cannot pass the default 80% coverage gate.

`SkillToolCall` carries an ID, nonnegative step, normalized tool identity,
optional component, arguments, observation, tri-state outcome, script path,
artifact reads/writes, retry correlation, explicit fault owner, and optional
productivity label. IDs are unique and steps nondecreasing. Retries must point
to a known failed call in an earlier step. Calls at the same step have no
causal order. `SkillToolTrace.initial_context` is null when coverage is unknown;
an explicit empty list asserts that there is no additional preexisting context
relevant to the handoff. Producers must include relevant prior messages,
initial files, and other context there when assessing value transfer.

Report metrics **consume** evidence; this change does not install SkillSpector,
Semgrep, secret scanners, or a bundled-test sandbox, or implement a Fabric/Harbor
trace normalizer. Those producer interfaces are mocked in the preview so
runtime work can proceed in parallel. A digest-bound report is an adapter's
claim of provenance, not cryptographic verification of the scanner itself.

## Judge configuration and skill revision comparisons

Factories return `(LLMJudgeMetric, views)`. Documentation review requires
`reference.skill_snapshot.document`. Outcome review requires the task's
`inputs.instruction`, held-out `reference.ground_truth`, and final answer.
Missing prerequisite fields can fail template rendering; supply them explicitly
and inspect metric coverage. Missing/invalid verdicts never satisfy a view.

An `AgentEvalTask` permits one metric of each type, including one `llm-judge`.
Use `skill_outcome_metric(model, assertions)` to combine five-criterion
accuracy, goal success, and the first PR's assertions in one request, keeping
their separate views. Run documentation review in a separate artifact task.
Judge task success concerns the observable answer. Use existing Harbor/Gym
verifier rewards when success depends on changed files, tests, or external state.

For v1 vs v2, hold tasks, ground truth, assertions, declared expectations,
model/harness settings, and judge configuration fixed. Supply each revision's
own snapshot, reports, trial IDs, and trace. Compare outcome quality first;
activation, routing, recovery, and cost explain differences. Artifact checks
apply to each revision once, rather than being treated as independent samples
for every repeated live attempt. Use authored negative tasks for overactivation
and separate no-skill arms for baseline lift.

Native token/latency/cost measurements, trial status, metric coverage, runtime
failures, rewards, pass@k, and existing run comparisons stay with Helix's
measurement/reporting layer. Paired lift, uncertainty intervals, aggregate
component coverage across a suite, and compositional ablation are run-level
analyses; these per-trial metrics supply their inputs. They do not implement a
new significance test or an overall SkillEvaluator score.

The existing metric/view rendering path can consume these named outputs.
This change adds no frontend controls or dedicated skill-version comparison
page. Runtime evidence adapters and any future skill-specific UI remain
separate integration work.

## Offline preview

With the workspace Evals SDK installed, run from the repository root:

```bash
uv run --no-sync python -m packages.nhx_evals_sdk.examples.skill_eval.run_additional_metric_preview
```

This scores all 24 metrics and the documentation/outcome rubrics through the
SDK evaluator for two synthetic revisions. Scanner/test reports and judge
replies are fixtures, labeled as such. It needs no scanner binaries, services,
agent execution, or model credentials. The original six-metric preview remains
`run_metric_preview`.
