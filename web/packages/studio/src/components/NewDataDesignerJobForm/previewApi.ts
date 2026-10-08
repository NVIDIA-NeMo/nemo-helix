// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataDesignerConfig } from '@nemo/sdk/generated/data-designer/schema';
import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
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

/**
 * Parse a single SSE line: expect JSON with optional "message" field; otherwise treat as plain text.
 */
function parsePreviewLine(trimmed: string): string | null {
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed) as { message?: string };
    if (typeof parsed.message === 'string') return parsed.message;
  } catch {
    /* fall through to return raw line */
  }
  return trimmed;
}

/**
 * Stream the data designer preview endpoint and invoke onLine for each log line.
 */
export async function streamPreview(
  path: string,
  requestBody: PreviewRequestBody,
  accessToken: string | undefined,
  signal: AbortSignal,
  onLine: (line: string) => void
): Promise<void> {
  const response = await platformFetch(`${PLATFORM_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
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

  await readLineDelimitedStream(body, (line) => {
    const msg = parsePreviewLine(line.trim());
    if (msg) onLine(msg);
  });
}
