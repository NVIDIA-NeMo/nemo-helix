<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor CLI Eval Comparison: Standard vs Easy

**Batch directory:** `jobs/batch-2026-02-12__10-59-58`
**Generated:** 2026-02-12 13:54:04
**Model:** `aws/anthropic/bedrock-claude-sonnet-4-5-v1`
**Agent:** `claude-code`

This compares the current CLI evals vs "easy" versions. The easy versions all provide pointers to the needed CLI commands. 

The question I'm answering is: if we stick the right CLI commands in the context, does this significantly reduce time/tokens for Claude Code to achieve the exact same tasks.

The answer is *mostly* yes. The one counter example is inference-chat-completions, where Claude Code is more efficient simply running curls against the API. Telling it to use the CLI slows it down.

## Summary

| Variant  | Pass | Total | Rate |
|----------|------|-------|------|
| Standard | 13 | 13 | 100% |
| Easy     | 13 | 13 | 100% |

### Timing (Agent Execution)

| Variant  | Mean   | Median | Min    | Max    |
|----------|--------|--------|--------|--------|
| Standard | 3:37 | 3:17 | 0:44 | 9:18 |
| Easy     | 2:31 | 2:23 | 0:36 | 5:39 |

## Pair-by-Pair Comparison

| Eval | Std Result | Std Time | # Std Tokens | # Std Tools | Easy Result | Easy Time | # Easy Tokens | # Easy Tools | Time Diff | # Token Diff | # Tool Diff |
|------|-----------|----------|--------------|-------------|-------------|-----------|---------------|--------------|-----------|--------------|-------------|
| garak-plugin-config-crud | PASS | 5:11 | 1.9M | 34 | PASS | 3:46 | 1.0M | 23 | -27% | -46% | -32% |
| garak-plugin-target-crud | PASS | 4:09 | 974.1k | 25 | PASS | 2:23 | 529.0k | 15 | -43% | -46% | -40% |
| auth-authorization | PASS | 4:17 | 1.3M | 30 | PASS | 2:25 | 602.6k | 19 | -44% | -53% | -37% |
| data-designer-config | PASS | 2:35 | 943.2k | 20 | PASS | 1:23 | 484.6k | 11 | -46% | -49% | -45% |
| entities-basic | PASS | 4:02 | 1.1M | 25 | PASS | 2:58 | 758.6k | 19 | -27% | -31% | -24% |
| evaluator-simple-job | PASS | 9:18 | - | 84 | PASS | 5:39 | 2.3M | 41 | -39% | - | -51% |
| files-crud | PASS | 3:34 | 1.1M | 30 | PASS | 2:29 | 839.8k | 21 | -30% | -25% | -30% |
| files-upload-dataset | PASS | 3:05 | 944.3k | 24 | PASS | 1:55 | 578.9k | 18 | -38% | -39% | -25% |
| guardrails-content-safety | PASS | 1:56 | 690.4k | 15 | PASS | 0:53 | 252.2k | 6 | -54% | -63% | -60% |
| inference-chat-completions | PASS | 3:17 | 1.6M | 29 | PASS | 4:26 | - | 67 | +35% | - | +131% |
| inference-provider-reg | PASS | 2:38 | 1.1M | 28 | PASS | 2:04 | 776.1k | 19 | -21% | -31% | -32% |
| secrets-crud | PASS | 2:16 | 978.1k | 22 | PASS | 1:53 | 747.0k | 19 | -16% | -24% | -14% |
| workspace-basic | PASS | 0:44 | 325.1k | 8 | PASS | 0:36 | 234.3k | 5 | -18% | -28% | -38% |

## Outcome Matrix

```
Eval                                       Standard       Easy
--------------------------------------------------------------
garak-plugin-config-crud-cli                        PASS       PASS
garak-plugin-target-crud-cli                        PASS       PASS
auth-authorization-cli                         PASS       PASS
data-designer-config-cli                       PASS       PASS
entities-basic-cli                             PASS       PASS
evaluator-simple-job-cli                       PASS       PASS
files-crud-cli                                 PASS       PASS
files-upload-dataset-cli                       PASS       PASS
guardrails-content-safety-cli                  PASS       PASS
inference-chat-completions-cli                 PASS       PASS
inference-provider-reg-cli                     PASS       PASS
secrets-crud-cli                               PASS       PASS
workspace-basic-cli                            PASS       PASS
```
