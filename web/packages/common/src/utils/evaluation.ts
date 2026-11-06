// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure utility functions for generating evaluation templates.
 *
 * IMPORTANT: This file is used by Playwright e2e tests which run in Node.js (not Vite),
 * so any imports must be Node-compatible (no import.meta.env or heavy dependencies).
 */
import {
  DEFAULT_LLM_JUDGE_DEFAULTS,
  EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING,
} from '@nemo/common/src/constants/metrics';
import type { ChatCompletionRole } from 'openai/resources/index.mjs';

interface InferenceRequestTemplate {
  messages: Array<{ role: ChatCompletionRole; content: string }>;
}

/**
 * Generates an inference request template from template selector inputs
 */
export function generateInferenceRequestTemplate(
  templateSelectorInputPrompt: string
): InferenceRequestTemplate | undefined {
  if (!templateSelectorInputPrompt) {
    return undefined;
  }

  return {
    messages: [{ role: 'user', content: templateSelectorInputPrompt }],
  };
}

/**
 * Generates the default LLM Judge user message template
 */
export function generateLLMJudgeUserMessage(
  templateSelectorInputGroundTruth: string,
  outputText: string = EVALUATION_DEFAULT_OUTPUT_TEMPLATE_STRING
): string {
  return DEFAULT_LLM_JUDGE_DEFAULTS.userMessage
    .replace('{{response1}}', templateSelectorInputGroundTruth)
    .replace('{{response2}}', outputText);
}
