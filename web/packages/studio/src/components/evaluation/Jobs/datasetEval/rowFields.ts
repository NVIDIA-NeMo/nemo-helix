// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DatasetEvalRow } from '@studio/components/evaluation/Jobs/datasetEval/DatasetEvalRowResultsPanel';

const EXPECTED_FIELDS = ['label', 'expected', 'reference', 'answer'];

export const expectedValue = (item?: Record<string, unknown>): string | null => {
  if (!item) return null;
  for (const field of EXPECTED_FIELDS) {
    const value = item[field];
    if (typeof value === 'string' && value) return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  }
  return null;
};

export const inputText = (row: DatasetEvalRow): string => {
  const request = row.requests?.[0]?.request;
  // `input_message` is the pre-chat-completions body; jobs submitted then still render.
  const rendered = request?.messages?.at(-1)?.content ?? request?.input_message;
  if (typeof rendered === 'string' && rendered) return rendered;
  return row.item ? JSON.stringify(row.item, null, 2) : '';
};

export const scoreCells = (
  row: DatasetEvalRow
): { label: string; value: number | string | undefined }[] =>
  Object.entries(row.metrics ?? {}).flatMap(([metricType, outputs]) =>
    (Array.isArray(outputs) ? outputs : []).map((output) => ({
      label: output?.name ? `${metricType}.${output.name}` : metricType,
      value: output?.value,
    }))
  );
