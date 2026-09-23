// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DATA_DESIGNER_JOB_GENERATOR_SYSTEM_PROMPT } from '@studio/components/NewDataDesignerJobForm/constants';
import type { ChatCompletionMessageParam } from 'openai/resources/index.mjs';

export interface FixRequestInput {
  /** The description the user originally wrote, so the retry keeps the same intent. */
  prompt: string;
  /** The config the model produced last time, verbatim (its tool-call arguments). */
  config: string;
  /** Issues that block loading; empty when the draft was merely lossy. */
  errors: string[];
  /** Issues that don't block loading but lost something (skipped columns, substitutions). */
  warnings: string[];
}

const bulletList = (items: string[]): string => items.map((item) => `- ${item}`).join('\n');

/**
 * Builds the follow-up conversation that asks the model to repair its own draft: the original
 * request, the config it returned, and what the builder found wrong with it.
 *
 * The previous config is replayed as a plain assistant turn rather than a tool call — a
 * `tool_calls` message would need a matching `tool` response, which providers enforce
 * inconsistently, and the model only needs to see the JSON it wrote.
 */
export const buildFixMessages = ({
  prompt,
  config,
  errors,
  warnings,
}: FixRequestInput): ChatCompletionMessageParam[] => {
  const sections = [
    errors.length > 0
      ? `Errors — the builder cannot load this config until these are fixed:\n${bulletList(errors)}`
      : '',
    warnings.length > 0
      ? `Warnings — the config loads, but something was lost or substituted:\n${bulletList(warnings)}`
      : '',
  ].filter(Boolean);

  return [
    { role: 'system', content: DATA_DESIGNER_JOB_GENERATOR_SYSTEM_PROMPT },
    { role: 'user', content: prompt },
    { role: 'assistant', content: config },
    {
      role: 'user',
      content: [
        'That config was checked against the visual builder and these issues came back:',
        '',
        sections.join('\n\n'),
        '',
        'Call the tool again with a corrected job config. Keep the intent, column names, and',
        'anything already working unchanged — change only what is needed to resolve the issues',
        'above. If a column type was skipped because the builder cannot edit it, replace it with',
        'the closest supported column type rather than dropping the data it produced.',
      ].join('\n'),
    },
  ];
};
