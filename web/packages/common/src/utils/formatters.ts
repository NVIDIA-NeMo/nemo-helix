// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Returns a formatted string with count and properly pluralized text.
 * @param text - The singular form of the word (e.g., "entry", "file")
 * @param count - The number to display
 * @param plural - Optional custom plural form for irregular words (e.g., "entries", "children").
 *                 If not provided, defaults to appending 's' to the text.
 * @returns Formatted string like "1 entry" or "3 entries"
 * @example
 * getTextWithCount('file', 1)           // "1 file"
 * getTextWithCount('file', 3)           // "3 files"
 * getTextWithCount('entry', 2, 'entries') // "2 entries"
 */
export const getTextWithCount = (text: string, count: number, plural?: string) => {
  const pluralForm = plural ?? `${text}s`;
  return `${count} ${count !== 1 ? pluralForm : text}`;
};

import type { FinetuningType } from '@nemo/sdk/generated/platform/schema';

/** Maps FinetuningType enum values to their correct display labels. */
const FINETUNING_TYPE_LABELS: Record<FinetuningType, string> = {
  lora: 'LoRA',
  lora_merged: 'LoRA Merged',
  all_weights: 'All Weights',
  last_layer: 'Last Layer',
  top_layers: 'Top Layers',
  gradual_unfreezing: 'Gradual Unfreezing',
  bias_only: 'Bias Only',
  attention_only: 'Attention Only',
  qlora: 'QLoRA',
  adalora: 'AdaLoRA',
  dora: 'DoRA',
  lora_plus: 'LoRA+',
  prompt_tuning: 'Prompt Tuning',
  prefix_tuning: 'Prefix Tuning',
  p_tuning: 'P-Tuning',
  p_tuning_v2: 'P-Tuning v2',
  soft_prompt: 'Soft Prompt',
  ppo: 'PPO',
  dpo: 'DPO',
  cdpo: 'cDPO',
  ipo: 'IPO',
  orpo: 'ORPO',
  kto: 'KTO',
  rrhf: 'RRHF',
  grpo: 'GRPO',
};

/** Format a FinetuningType enum value for display. Falls back to the raw value for unknown types. */
export const formatFinetuningType = (type: FinetuningType): string =>
  FINETUNING_TYPE_LABELS[type] || type;

export const kebabCaseToTitleCase = (str: string) => {
  return str.replace(/-/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
};

export const snakeCaseToTitleCase = (str: string) => {
  return str.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
};

/**
 * Converts a number to scientific notation string if it is very small or large.
 */
export const toScientificNotation = (value: number) => {
  if ((Math.abs(value) < 1e-6 && value !== 0) || Math.abs(value) >= 1e15) {
    return value.toExponential();
  }
  return value.toString();
};

/**
 * Formats an evaluator score for display as a raw number. Evaluator values are stored as bare
 * floats with no scale metadata, so we cannot know whether one is a ratio, a pass/fail flag, or a
 * point count — rendering everything as a percentage (the old behavior) turned an integer `1` into
 * a misleading `100%`. Show the number as-is instead: `toFixed(3)` bounds the precision, then
 * trailing zeros are stripped so integers stay integers (`1`) and fractions read cleanly (`0.603`).
 */
export const formatEvaluatorScore = (value: number | null | undefined): string => {
  if (value == null || !Number.isFinite(value)) return '-';
  return value.toFixed(3).replace(/\.?0+$/, '');
};
