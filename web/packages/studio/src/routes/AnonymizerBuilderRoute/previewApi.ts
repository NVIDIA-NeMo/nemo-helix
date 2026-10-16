// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LogFrameLevel,
  ModelDownloadFrameStatus,
  type Done,
  type Error as ErrorFrame,
  type FailedRecordsFrame,
  type Heartbeat,
  type LogFrame,
  type ModelDownloadFrame,
  type PreviewDatasetFrame,
  type PreviewRequest,
  type TraceDatasetFrame,
} from '@nemo/sdk/generated/anonymizer/schema';
import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { asRecord } from '@studio/util/guards';
import { readLineDelimitedStream } from '@studio/util/lineStream';

export type PreviewFrame =
  | LogFrame
  | ModelDownloadFrame
  | PreviewDatasetFrame
  | TraceDatasetFrame
  | FailedRecordsFrame
  | Heartbeat
  | Done
  | ErrorFrame;

const LOG_LEVELS: readonly string[] = Object.values(LogFrameLevel);
const DOWNLOAD_STATUSES: readonly string[] = Object.values(ModelDownloadFrameStatus);

const asRecordList = (value: unknown): Record<string, unknown>[] =>
  Array.isArray(value)
    ? value.flatMap((entry) => {
        const row = asRecord(entry);
        return row ? [row] : [];
      })
    : [];

/** Frames arrive as NDJSON. Anything unrecognised is dropped rather than surfaced as an error. */
export const parsePreviewFrame = (line: string): PreviewFrame | undefined => {
  const trimmed = line.trim();
  if (!trimmed) return undefined;

  let decoded: unknown;
  try {
    decoded = JSON.parse(trimmed);
  } catch {
    return undefined;
  }

  const frame = asRecord(decoded);
  const kind = frame?.kind;
  if (!frame || typeof kind !== 'string') return undefined;

  switch (kind) {
    case 'log': {
      const { level, message } = frame;
      return {
        kind,
        level: LOG_LEVELS.includes(String(level))
          ? (level as LogFrame['level'])
          : LogFrameLevel.info,
        message: typeof message === 'string' ? message : '',
      };
    }
    case 'model_download': {
      const { status, message } = frame;
      if (!DOWNLOAD_STATUSES.includes(String(status))) return undefined;
      return {
        kind,
        status: status as ModelDownloadFrame['status'],
        message: typeof message === 'string' ? message : '',
      };
    }
    case 'preview_dataset':
    case 'failed_records':
      return { kind, records: asRecordList(frame.records) };
    case 'trace_dataset': {
      const column = frame.original_text_column;
      return {
        kind,
        records: asRecordList(frame.records),
        original_text_column: typeof column === 'string' ? column : undefined,
      };
    }
    case 'heartbeat':
    case 'done':
      return { kind };
    case 'error':
      return {
        kind,
        message: typeof frame.message === 'string' ? frame.message : 'The preview run failed.',
      };
    default:
      return undefined;
  }
};

/** FastAPI returns `detail` as either a plain string or a list of pydantic errors. */
const messageFromErrorBody = (body: string): string | undefined => {
  let decoded: unknown;
  try {
    decoded = JSON.parse(body);
  } catch {
    return body.trim() || undefined;
  }
  const detail = asRecord(decoded)?.detail;
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return undefined;
  const messages = detail.flatMap((item) => {
    const msg = asRecord(item)?.msg;
    return typeof msg === 'string' ? [msg] : [];
  });
  return messages.length ? messages.join(' ') : undefined;
};

const previewPath = (workspace: string): string =>
  `/apis/anonymizer/v2/workspaces/${encodeURIComponent(workspace)}/preview`;

export const streamAnonymizerPreview = async (
  workspace: string,
  request: PreviewRequest,
  accessToken: string | undefined,
  signal: AbortSignal,
  onFrame: (frame: PreviewFrame) => void
): Promise<void> => {
  const response = await platformFetch(`${PLATFORM_BASE_URL}${previewPath(workspace)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(messageFromErrorBody(body) ?? `Preview failed: ${response.status}`);
  }
  if (!response.body) throw new Error('The preview response was empty.');

  await readLineDelimitedStream(response.body, (line) => {
    const frame = parsePreviewFrame(line);
    if (frame) onFrame(frame);
  });
};
