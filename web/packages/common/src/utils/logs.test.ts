// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

import { formatLogs } from './logs';

const createLog = (message: string, timestamp: string): HelixJobLog => ({
  timestamp,
  job: 'test-job',
  job_step: 'training',
  job_task: 'main',
  message,
});

describe('formatLogs', () => {
  it('formats a single log entry with timestamp and message', () => {
    const logs = [createLog('Training started', '2026-02-20T19:08:56Z')];
    const result = formatLogs(logs);
    expect(result).toBe('[2026-02-20T19:08:56Z]   Training started');
  });

  it('formats multiple log entries separated by newlines', () => {
    const logs = [
      createLog('Step 1 complete', '2026-02-20T19:08:56Z'),
      createLog('Step 2 complete', '2026-02-20T19:09:00Z'),
      createLog('Training finished', '2026-02-20T19:10:00Z'),
    ];
    const result = formatLogs(logs);
    const lines = result.split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe('[2026-02-20T19:08:56Z]   Step 1 complete');
    expect(lines[1]).toBe('[2026-02-20T19:09:00Z]   Step 2 complete');
    expect(lines[2]).toBe('[2026-02-20T19:10:00Z]   Training finished');
  });

  it('returns no trailing whitespace', () => {
    const logs = [createLog('done', '2026-02-20T19:10:00Z')];
    const result = formatLogs(logs);
    expect(result).toBe(result.trimEnd());
  });

  it('returns an empty string for empty array', () => {
    const result = formatLogs([]);
    expect(result).toBe('');
  });
});
