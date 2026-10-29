// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatLogs, formatLogsForDisplay, logStatusForJob } from '@nemo/common/src/utils/logs';
import { HelixJobLog, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

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

describe('formatLogsForDisplay', () => {
  it('strips ANSI colour codes', () => {
    const logs = [
      createLog('\x1b[2mdim\x1b[0m [\x1b[32minfo\x1b[0m] done', '2026-02-20T19:08:56Z'),
    ];
    expect(formatLogsForDisplay(logs)).toBe('[2026-02-20T19:08:56Z]   dim [info] done');
  });

  it('normalizes carriage returns to newlines', () => {
    const logs = [
      createLog('Traceback\r\n  boom', '2026-02-20T19:08:56Z'),
      createLog('progress 50%\r', '2026-02-20T19:09:00Z'),
    ];
    expect(formatLogsForDisplay(logs)).toBe(
      '[2026-02-20T19:08:56Z]   Traceback\n  boom\n[2026-02-20T19:09:00Z]   progress 50%\n'
    );
  });

  it('leaves the raw formatter untouched', () => {
    const logs = [createLog('\x1b[32mok\x1b[0m\r', '2026-02-20T19:08:56Z')];
    expect(formatLogs(logs)).toBe('[2026-02-20T19:08:56Z]   \x1b[32mok\x1b[0m\r');
  });
});

describe('logStatusForJob', () => {
  it.each([
    [HelixJobStatus.created, 'live'],
    [HelixJobStatus.active, 'live'],
    [HelixJobStatus.paused, 'live'],
    [HelixJobStatus.completed, 'ended'],
    [HelixJobStatus.error, 'ended'],
    [HelixJobStatus.cancelled, 'ended'],
    [undefined, undefined],
  ])('maps %s -> %s', (status, expected) => {
    expect(logStatusForJob(status)).toBe(expected);
  });
});
