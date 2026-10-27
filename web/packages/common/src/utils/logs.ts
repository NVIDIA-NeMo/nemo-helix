// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

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

/** Compares log entries by content, since every fetch returns fresh objects. */
export const isSameLogEntry = (a: HelixJobLog, b: HelixJobLog): boolean =>
  a.timestamp === b.timestamp &&
  a.message === b.message &&
  a.job_step === b.job_step &&
  a.job_task === b.job_task;

/** Reaches the parts of a log that weren't loaded up front. */
export interface LogPaging {
  /** The server holds lines older than the ones loaded. */
  hasOlderLogs: boolean;
  isFetchingOlderLogs: boolean;
  /** Prepends the next page of older lines to the loaded logs. */
  fetchOlderLogs: () => Promise<void>;
  /** Fetches every line of the log, oldest first, e.g. for download. */
  fetchAllLogs: () => Promise<HelixJobLog[]>;
}
