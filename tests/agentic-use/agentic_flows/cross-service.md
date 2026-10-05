<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Service Integration Flows

These flows explicitly test the integration between multiple services and cannot be attributed to a single service. They represent complete ML pipelines and workflows that demonstrate the full value of the NeMo Helix.

**PIC**: Aaron Gabow
**Priority**: Low (but high value for demonstrating platform capabilities)

---

## Flows

| # | Flow Name | Services | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|----------|------------|----------|----------|-------------|--------|
| 34 | Customized Model Inference via IGW | Customizer, IGW | 4 | No | No | After customization job completes, verify the LoRA adapter is picked up by NIM and inference with the customized model returns expected results. | POR |
| 35 | Customization + Evaluation Loop | Customizer, Evaluator | 4 | No | No | Full loop: Upload dataset → Run base model evaluation → Run customization → Run customized model evaluation → Compare metrics to show improvement. | POR |
| 36 | Tool Calling Fine-Tuning + Evaluation | Customizer, Evaluator | 4 | No | No | Fine-tune a model on xLAM-format tool calling data, then verify improved tool calling performance through evaluation. | POR |
| 37 | Guardrails Evaluation Flow | Guardrails, Evaluator | 4 | No | No | Evaluate model responses with and without guardrails applied. Verify guardrails improve safety metrics on content safety test dataset. | POR |
| 38 | Full E2E Flow | Files, IGW, Evaluator, Customizer, Guardrails | 5 | No | No | Dataset upload → Base model inference → Base model evaluation → LoRA customization → Customized model inference → Customized model evaluation → Compare metrics → Guardrails inference check → Guardrails evaluation. | POR |
| 39 | Data Flywheel (Full Cycle) | Intake, Files, Evaluator, Customizer, Guardrails | 5 | No | No | Intake collection → Dataset creation → Evaluation → Customization → Deployment → Guardrails → Re-evaluation. Full lifecycle demonstrating the NeMo Helix value proposition. | POR; docs/notebooks/data-flywheel-bp-tutorial.ipynb |

---

## Flow Details

### 34. Customized Model Inference via IGW

**Complexity**: 4 (Complex)

**Services**: Customizer, IGW

**Operations**:
1. Complete a customization job (LoRA)
2. Verify LoRA adapter saved to Files service
3. Configure NIM deployment to use LoRA adapter
4. Run inference through IGW
5. Verify customized model behavior

**Prerequisites**:
- Completed customization job
- NIM deployment with LoRA support

**Success Criteria**:
- LoRA adapter picked up by NIM
- Inference reflects customized behavior
- Response quality matches training objective

---

### 35. Customization + Evaluation Loop

**Complexity**: 4 (Complex)

**Services**: Customizer, Evaluator, Files, IGW

**Operations**:
1. Upload dataset to Files service
2. Run base model evaluation (BLEU, accuracy)
3. Note baseline metrics
4. Run customization job
5. Run customized model evaluation
6. Compare metrics to show improvement

**Expected Improvements**:
- Initial BLEU >= 2, customized >= 35
- Initial accuracy >= 0, customized >= 0.45
- Customized model significantly outperforms base

**Prerequisites**:
- Training dataset
- Evaluation dataset
- Base model accessible

**Success Criteria**:
- Both evaluations complete
- Customized model shows improvement
- Metrics quantify the improvement

---

### 36. Tool Calling Fine-Tuning + Evaluation

**Complexity**: 4 (Complex)

**Services**: Customizer, Evaluator

**Operations**:
1. Upload xLAM-format tool calling training dataset
2. Upload BFCL evaluation dataset
3. Evaluate base model on tool calling
4. Fine-tune on tool calling data
5. Evaluate customized model
6. Compare function_name_accuracy and function_name_and_args_accuracy

**Expected Improvements**:
- function_name_accuracy improvement >= 0.4
- function_name_and_args_accuracy improvement >= 0.2

**Prerequisites**:
- xLAM training dataset
- BFCL evaluation dataset
- Tool-calling capable base model

**Success Criteria**:
- Tool calling accuracy improves after fine-tuning
- Both metrics show measurable improvement

---

### 37. Guardrails Evaluation Flow

**Complexity**: 4 (Complex)

**Services**: Guardrails, Evaluator

**Operations**:
1. Prepare content safety test dataset
2. Run evaluation WITHOUT guardrails
3. Note safety-related metrics
4. Run evaluation WITH guardrails applied
5. Compare metrics to show guardrails improvement

**Expected Results**:
- BLEU difference (with guardrails) >= 20
- Safety metrics improve with guardrails

**Prerequisites**:
- Content safety test dataset
- Guardrails configuration
- Model accessible via IGW

**Success Criteria**:
- Guardrails block unsafe content
- Safety metrics improve
- Acceptable impact on valid requests

---

### 38. Full E2E Flow

**Complexity**: 5 (Advanced)

**Services**: Files, IGW, Evaluator, Customizer, Guardrails

**Full Pipeline**:
1. **Dataset Upload**: Upload train/val/test data to Files
2. **Base Inference**: Test base model inference
3. **Base Evaluation**: Run evaluation on base model
4. **Customization**: Fine-tune with LoRA (2 epochs, SFT)
5. **Customized Inference**: Test customized model
6. **Customized Evaluation**: Evaluate improvement
7. **Compare Metrics**: Quantify improvement
8. **Chat Dataset**: Repeat with chat-format data
9. **Guardrails Inference**: Test safety filtering
10. **Guardrails Evaluation**: Verify safety improvement

**Prerequisites**:
- All services running
- GPU resources for customization
- Multiple dataset formats prepared

**Success Criteria**:
- All pipeline stages complete
- Customization shows improvement
- Guardrails improve safety
- Full loop demonstrates platform value

---

### 39. Data Flywheel (Full Cycle)

**Complexity**: 5 (Advanced)

**Services**: Intake, Files, Evaluator, Customizer, Guardrails

**Full Lifecycle**:
1. **Intake Collection**: Collect production LLM interactions
2. **Dataset Creation**: Export Intake data to Files service
3. **Evaluation**: Assess current model performance
4. **Customization**: Fine-tune on collected data
5. **Deployment**: Deploy customized model
6. **Guardrails**: Apply safety controls
7. **Re-evaluation**: Measure improvement
8. **Repeat**: Continue flywheel cycle

**Value Proposition**:
This flow demonstrates the core NeMo Helix value proposition:
- Continuous improvement from production data
- Automated model enhancement pipeline
- Safety-first deployment

**Prerequisites**:
- Full NeMo Helix deployed
- Production data collection active
- Complete pipeline infrastructure

**Success Criteria**:
- Full cycle completes end-to-end
- Model improves with each iteration
- Safety maintained throughout
- Metrics tracked across cycles

---

## Documentation References

- E2E tutorial: docs/get-started/tutorials/customize-eval-loop.md
- Safety tutorial: docs/get-started/tutorials/add-safety-checks.md
- Data flywheel: docs/notebooks/data-flywheel-bp-tutorial.ipynb
