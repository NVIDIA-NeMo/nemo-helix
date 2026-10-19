// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

/**
 * Formats platform job logs into a timestamped text block for display in a code snippet.
 */
export const formatLogs = (logEntries: HelixJobLog[]): string => {
  return logEntries.map((log) => `[${log.timestamp}]   ${log.message}`).join('\n');
};

/**
 * Incremental state of a multi-page log fetch. `total` is the server-side line count
 * for the whole job, known once the first page resolves.
 */
export interface LogLoadProgress {
  /** Lines fetched so far across every page walked. Not reduced by retention trimming. */
  loaded: number;
  total: number;
}
