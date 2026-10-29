// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import type { HelixJobLog, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

/**
 * Formats platform job logs into a raw timestamped text block, e.g. for download.
 */
export const formatLogs = (logEntries: HelixJobLog[]): string => {
  return logEntries.map((log) => `[${log.timestamp}]   ${log.message}`).join('\n');
};

// ANSI colour codes (CSI sequences) from terminal-style loggers.
// eslint-disable-next-line no-control-regex
const ANSI_ESCAPE = /\x1b\[[0-?]*[ -/]*[@-~]/g;

/**
 * Formats one log entry as the log viewer shows it: ANSI codes stripped and carriage returns
 * normalized, since CodeMirror also breaks lines on a lone \r.
 */
export const formatLogEntryForDisplay = (log: HelixJobLog): string =>
  `[${log.timestamp}]   ${log.message}`.replace(ANSI_ESCAPE, '').replace(/\r\n?/g, '\n');

/** Formats log entries as the log viewer shows them, so copied text matches the screen. */
export const formatLogsForDisplay = (logs: readonly HelixJobLog[]): string =>
  logs.map(formatLogEntryForDisplay).join('\n');

/** Whether more lines may still arrive (`live`) or the log has ended (`ended`). */
export type LogStatus = 'live' | 'ended';

/** Live while the job can still write lines, ended once it's terminal; undefined when the status is unknown. */
export const logStatusForJob = (status: HelixJobStatus | undefined): LogStatus | undefined =>
  status ? (CJobTerminalStatuses.includes(status) ? 'ended' : 'live') : undefined;

/**
 * Incremental state of a multi-page log fetch. `total` is the server-side line count
 * for the whole job, known once the first page resolves.
 */
export interface LogLoadProgress {
  /** Lines fetched so far across every page walked. Not reduced by retention trimming. */
  loaded: number;
  total: number;
}
