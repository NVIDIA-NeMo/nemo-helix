// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataDesignerConfig } from '@nemo/sdk/generated/data-designer/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { asRecord } from '@studio/util/guards';
import { readLineDelimitedStream } from '@studio/util/lineStream';

/** Request body for the data designer preview stream endpoint */
export interface PreviewRequestBody {
  config: DataDesignerConfig;
  num_records?: number;
}

/**
 * Format preview log text for display: lines that are JSON are pretty-printed for readability.
 */
export function formatPreviewLogsForDisplay(raw: string): string {
  if (!raw.trim()) return raw;
  return raw
    .split('\n')
    .map((line) => {
      const trimmed = line.trim();
      const isJson =
        (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
        (trimmed.startsWith('[') && trimmed.endsWith(']'));
      if (!isJson) return line;
      try {
        return JSON.stringify(JSON.parse(trimmed), null, 2);
      } catch {
        return line;
      }
    })
    .join('\n');
}

export function isAbortError(err: unknown): boolean {
  return (
    typeof err === 'object' && err !== null && 'name' in err && (err as Error).name === 'AbortError'
  );
}

export type PreviewRecord = Record<string, unknown>;

export interface PreviewStreamHandlers {
  onLog: (line: string) => void;
  onRecords: (records: PreviewRecord[]) => void;
  onError: (message: string) => void;
}

/** Record keys in first-seen order, with the config's columns first in config order. */
export const orderPreviewColumns = (
  records: readonly PreviewRecord[],
  columnOrder: readonly string[]
): string[] => {
  const seen = new Set<string>();
  for (const record of records) {
    for (const key of Object.keys(record)) seen.add(key);
  }
  const configured = columnOrder.filter((name) => seen.has(name));
  const configuredSet = new Set(configured);
  return [...configured, ...[...seen].filter((key) => !configuredSet.has(key))];
};

const asRecordList = (value: unknown): PreviewRecord[] =>
  Array.isArray(value)
    ? value.flatMap((entry) => {
        const row = asRecord(entry);
        return row ? [row] : [];
      })
    : [];

/**
 * Route one NDJSON frame: `dataset` frames carry the generated records, frames with a
 * `message` (log, error) become log lines, and anything else is logged verbatim.
 */
export function dispatchPreviewLine(line: string, handlers: PreviewStreamHandlers): void {
  const trimmed = line.trim();
  if (!trimmed) return;

  let decoded: unknown;
  try {
    decoded = JSON.parse(trimmed);
  } catch {
    handlers.onLog(trimmed);
    return;
  }

  const frame = asRecord(decoded);
  if (frame?.kind === 'dataset') {
    handlers.onRecords(asRecordList(frame.records));
    return;
  }

  const message = frame?.message;
  if (typeof message !== 'string') {
    handlers.onLog(trimmed);
    return;
  }
  if (frame?.kind === 'error') handlers.onError(message);
  handlers.onLog(message);
}

/**
 * Stream the data designer preview endpoint, routing each frame to `handlers`.
 */
export async function streamPreview(
  path: string,
  requestBody: PreviewRequestBody,
  accessToken: string | undefined,
  signal: AbortSignal,
  handlers: PreviewStreamHandlers
): Promise<void> {
  const response = await fetch(`${PLATFORM_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      'X-Source': 'NeMo Studio',
    },
    body: JSON.stringify(requestBody),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Preview failed: ${response.status}`);
  }

  const body = response.body;
  if (!body) throw new Error('No response body');

  await readLineDelimitedStream(body, (line) => dispatchPreviewLine(line, handlers));
}
