<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Safe Synthesizer Service Agentic Flows

The Safe Synthesizer service generates privacy-preserving synthetic versions of sensitive tabular data using differential privacy techniques.

**PIC**: Matt Kornfield
**Priority**: Low

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 29 | Safe Synthesizer Flow | 5 | No | No | Upload tabular data → Run PII detection → Run synthetic data generation → Evaluate privacy metrics → Download synthetic dataset. | POR; tests/e2e/test_safe_synthesizer.py |

---

## Flow Details

### 29. Safe Synthesizer Flow

**Complexity**: 5 (Advanced)

**Operations**:
1. Upload original tabular dataset to Files service
2. Configure PII detection settings
3. Run PII detection/removal phase
4. Configure synthetic generation parameters
5. Run synthetic data generation (with optional differential privacy)
6. Evaluate privacy metrics against original
7. Download synthetic dataset

**Pipeline Stages**:
```
Original Data → PII Removal → Fine Tune/Generate → Evaluation → Synthetic Output
```

**Available Notebooks/Tutorials**:
- PII only flow
- 101 Tutorial (basic synthesis)
- Advanced Privacy (differential privacy)

**Prerequisites**:
- Tabular dataset in supported format
- GPU resources for generation
- Privacy parameters defined

**Configuration Options**:
- PII detection sensitivity
- Differential privacy epsilon
- Generation parameters
- Evaluation metrics

**Success Criteria**:
- PII successfully detected and removed
- Synthetic data generated
- Statistical similarity to original maintained
- Privacy metrics meet requirements
- Synthetic dataset downloadable

**Privacy Metrics Evaluated**:
- Statistical similarity scores
- Differential privacy guarantees
- Re-identification risk assessment

---

## Documentation References

- 101 Tutorial: docs/safe-synthesizer/tutorials/safe-synthesizer-101.md
- Synthesize: docs/safe-synthesizer/synthesize/
- Evaluate: docs/safe-synthesizer/evaluate/

**Note**: Additional permutations exist in OSS repository; NeMo Helix includes top user flows.
