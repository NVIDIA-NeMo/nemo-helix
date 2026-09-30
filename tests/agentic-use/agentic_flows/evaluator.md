<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluator Service Agentic Flows

The Evaluator service provides comprehensive evaluation capabilities for language models, including standard metrics, LLM-as-a-Judge, academic benchmarks, and tool calling evaluation.

**PIC**: Sandy Chapman
**Priority**: Medium

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 12 | Simple Custom Evaluation Job | 3 | No | No | Launch a custom evaluation with a simple metric (BLEU, ROUGE, or string-check) against a dataset stored in Files service, targeting a model via IGW. | POR; tests/e2e/evaluator/test_metric_jobs.py |
| 13 | LLM-as-a-Judge Evaluation | 3 | No | No | Configure and run an LLM-as-a-judge evaluation job that uses an LLM (via IGW) to score model outputs based on rubric criteria. | POR; tests/e2e/evaluator/test_llm_judge_jobs.py |
| 14 | Zero-Config LLM-as-a-Judge | 3 | No | No | Run the new zero-config LLM-as-a-Judge flow that requires minimal configuration, using sensible defaults. | POR (new v2 feature) |
| 15 | Academic Benchmark Evaluation | 3 | No | No | Trigger an academic benchmark evaluation using lm_eval_harness or MMLU against a model. Verify results with expected score ranges. | POR |
| 16 | Tool Calling Evaluation (BFCL) | 3 | No | No | Run a BFCL-style evaluation for function calling/tool use. Verify function_name_accuracy and function_name_and_args_accuracy metrics. | POR |

---

## Flow Details

### 12. Simple Custom Evaluation Job

**Complexity**: 3 (Moderate)

**Operations**:
1. Prepare dataset in Files service
2. Create evaluation configuration with metric
3. Launch evaluation job targeting model via IGW
4. Monitor job status
5. Retrieve results

**Supported Metrics**:
- BLEU score
- ROUGE score
- String-check (equals, startswith, contains)

**Prerequisites**:
- Dataset in Files service
- Model accessible via IGW
- Workspace exists

**Success Criteria**:
- Evaluation job created and runs
- Job completes successfully
- Results contain aggregate scores
- Results contain per-row scores

---

### 13. LLM-as-a-Judge Evaluation

**Complexity**: 3 (Moderate)

**Operations**:
1. Configure judge model (via IGW)
2. Define evaluation rubric/criteria
3. Prepare dataset with model outputs
4. Launch LLM-as-a-Judge job
5. Retrieve scored results

**Prerequisites**:
- Judge LLM accessible via IGW
- Target model outputs to evaluate
- Evaluation criteria defined

**Success Criteria**:
- Judge LLM scores outputs correctly
- Scores align with rubric criteria
- Results are reproducible

---

### 14. Zero-Config LLM-as-a-Judge

**Complexity**: 3 (Moderate)

**Operations**:
1. Provide minimal configuration (dataset, target model)
2. Let system use default judge and criteria
3. Run evaluation
4. Retrieve results

**New in v2**: This flow requires minimal setup using sensible defaults.

**Prerequisites**:
- Dataset available
- Target model accessible

**Success Criteria**:
- Evaluation runs with minimal config
- Default judge produces meaningful scores
- Results comparable to full configuration

---

### 15. Academic Benchmark Evaluation

**Complexity**: 3 (Moderate)

**Operations**:
1. Select benchmark (MMLU, HellaSwag, TruthfulQA, etc.)
2. Configure evaluation with lm_eval_harness
3. Run benchmark against model
4. Verify scores against expected ranges

**Supported Benchmarks**:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag
- TruthfulQA
- Custom lm_eval_harness tasks

**Prerequisites**:
- Model accessible via IGW
- Benchmark configuration

**Success Criteria**:
- Benchmark runs to completion
- Scores within expected ranges for model
- Results formatted correctly

---

### 16. Tool Calling Evaluation (BFCL)

**Complexity**: 3 (Moderate)

**Operations**:
1. Prepare BFCL-format evaluation dataset
2. Configure tool calling evaluation
3. Run evaluation against model
4. Verify metrics:
   - function_name_accuracy
   - function_name_and_args_accuracy

**Prerequisites**:
- Model with tool calling capability
- BFCL-format test dataset

**Success Criteria**:
- Evaluation completes
- function_name_accuracy reported
- function_name_and_args_accuracy reported
- Metrics align with model capability

---

## Documentation References

- Evaluation template: docs/evaluator/flows/template.md
- Run an evaluation: docs/evaluator/tutorials/run-an-evaluation.md
- LLM-as-a-Judge: docs/evaluator/flows/llm-as-a-judge.md
- LLM Judge tutorial: docs/evaluator/tutorials/run-llm-judge-evaluation.md
- Academic benchmarks: docs/evaluator/flows/academic-benchmarks/
- BFCL: docs/evaluator/flows/academic-benchmarks/bfcl.md
