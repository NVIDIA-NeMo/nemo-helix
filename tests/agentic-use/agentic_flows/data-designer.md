<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Designer Service Agentic Flows

The Data Designer service generates synthetic datasets using LLMs. It supports schema-based generation, seed data expansion, and preview capabilities.

**PIC**: Mike Knepper
**Priority**: Medium

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 26 | Configure Models | 2 | No | `data-designer-config-cli` | Set up model configurations for Data Designer including model aliases, provider settings, and inference parameters (temperature, max_tokens, etc.). | POR |
| 27 | Preview Synthetic Data | 3 | No | `data-designer-preview` | Configure columns (sampler + LLM-generated), run a preview to generate a small sample dataset (10 records). Validate configuration before full generation. | POR |
| 28 | Full Batch Generation Job | 4 | No | No | After validating with preview, run a full batch generation job to create a larger dataset. Monitor job status, download results. | POR |

---

## Flow Details

### 26. Configure Models

**Complexity**: 2 (Simple)

**Operations**:
1. Set up model aliases
2. Configure provider settings
3. Set inference parameters:
   - temperature
   - max_tokens
   - top_p
   - etc.
4. Save configuration

**Prerequisites**:
- NeMo Helix running
- Model provider accessible (via IGW)

**Success Criteria**:
- Model configuration saved
- Configuration can be used for generation
- Inference parameters applied correctly

---

### 27. Preview Synthetic Data

**Complexity**: 3 (Moderate)

**Operations**:
1. Define column schema:
   - Sampler columns (static values, distributions)
   - LLM-generated columns (prompts, constraints)
2. Configure relationships between columns
3. Run preview (generates ~10 records)
4. Validate output format and quality
5. Iterate on configuration if needed

**Column Types**:
- Sampler: Random values from distributions
- LLM-generated: Content created by language model
- Derived: Computed from other columns

**Prerequisites**:
- Model configuration set up
- Column schema defined

**Success Criteria**:
- Preview generates sample records
- Output matches expected schema
- LLM-generated content is coherent
- Relationships between columns maintained

---

### 28. Full Batch Generation Job

**Complexity**: 4 (Complex)

**Operations**:
1. Finalize column configuration (from preview)
2. Set batch size and record count
3. Launch full generation job
4. Monitor job progress
5. Handle any errors/retries
6. Download generated dataset

**Job Parameters**:
- Total records to generate
- Batch size
- Parallelism settings
- Output format

**Prerequisites**:
- Preview validated
- Model configuration finalized
- Sufficient API credits/resources

**Success Criteria**:
- Job runs to completion
- Correct number of records generated
- Output quality matches preview
- Dataset downloadable from Files service

---

## Documentation References

- Configure models: docs/data-designer/configure-models.md
- Preview: docs/data-designer/generate-data/manage-jobs/preview.md
- Define columns: docs/data-designer/define-your-data-columns/
- Generate data: docs/data-designer/generate-data/generating-data.md
- Create job: docs/data-designer/generate-data/manage-jobs/create-job.md

**Note**: Documentation may need updates as SDK vendoring strategy changes (GitLab 3187).
