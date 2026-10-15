// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileFormat, InputFileSchemaType } from '@nemo/common/src/types';
import type { DatasetInputFileResult } from '@studio/types/dataset';

/**
 * In-Studio sample prompt sets surfaced in the Run Prompts dataset picker.
 * Demo bridge — when the platform-seed work lands, these graduate to real
 * Filesets shipped alongside their agents and this constant goes away.
 */

const CALCULATOR_AGENT_PROMPTS: readonly string[] = [
  'What is 27 + 45?',
  'I have $250 and I spend 32% of it on groceries. How much do I have left?',
  'A train travels 240 miles in 3 hours. What is its average speed in miles per hour?',
  'A stock starts at $50 per share and grows 8% per year. What will it be worth after 5 years? Show your work.',
  "What is today's date?",
  'How many days are there between today and December 31st of this year?',
  'If I save $200 every month starting today, how much will I have saved by the end of this year?',
  'What is 7 divided by 0?',
  'What is the capital of France?',
  'If I invest $10,000 at 4.5% annual interest compounded monthly for 3 years, how much will I have at the end? Round to the nearest cent.',
];

/**
 * Synthesize a `DatasetInputFileResult` that looks exactly like what the
 * upload-from-disk path produces, so `ModelComparePrompts` can consume it via
 * its existing `onChange` without any branch on "is this a sample?".
 */
function buildCalculatorAgentSample(): DatasetInputFileResult {
  const parsedRows = CALCULATOR_AGENT_PROMPTS.map((prompt) => ({ prompt }));
  const firstRow = parsedRows[0];
  return {
    fileUrl: 'sample://calculator-agent-prompts.jsonl',
    format: FileFormat.JSONL,
    validationResult: { isValid: true, format: FileFormat.JSONL },
    detectionResult: {
      schemaType: InputFileSchemaType.COMPLETION,
      detectedFields: { prompt: 'prompt' },
      isComplete: true,
      firstRow,
    },
    keyMapping: { promptKey: 'prompt', completionKey: null, idealResponseKey: null },
    availableKeys: [{ label: 'prompt', value: 'prompt' }],
    firstRow,
    parsedRows,
    rowCount: parsedRows.length,
  };
}

export interface SampleDataset {
  id: string;
  label: string;
  description: string;
  build: () => DatasetInputFileResult;
}

export const SAMPLE_DATASETS: readonly SampleDataset[] = [
  {
    id: 'calculator-agent',
    label: 'Calculator agent — sample prompts (10)',
    description: 'Vibe-check prompts spanning arithmetic, datetime, and edge cases',
    build: buildCalculatorAgentSample,
  },
];
