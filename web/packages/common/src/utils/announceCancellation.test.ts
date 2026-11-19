// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { announceCancellation } from '@nemo/common/src/utils/announceCancellation';

const JOB = 'my-job';
const INTERVAL = 1000;
const TIMEOUT = 10_000;

const setup = (statuses: (string | Error)[]) => {
  const toast = { info: vi.fn(), success: vi.fn(), error: vi.fn() };
  const queue = [...statuses];
  const fetchStatus = vi.fn(async () => {
    const next = queue.length > 1 ? queue.shift()! : queue[0];
    if (next instanceof Error) throw next;
    return next;
  });
  const run = (status: string | undefined) =>
    announceCancellation({
      toast,
      jobName: JOB,
      status,
      fetchStatus,
      intervalMs: INTERVAL,
      timeoutMs: TIMEOUT,
    });
  return { toast, fetchStatus, run };
};

describe('announceCancellation', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('announces one success and never polls when the request already cancelled the job', async () => {
    const { toast, fetchStatus, run } = setup(['cancelled']);

    await run('cancelled');

    expect(toast.success).toHaveBeenCalledTimes(1);
    expect(toast.success).toHaveBeenCalledWith(expect.stringContaining(JOB));
    expect(toast.info).not.toHaveBeenCalled();
    expect(fetchStatus).not.toHaveBeenCalled();
  });

  it('says the request was accepted, then announces success once the job is cancelled', async () => {
    const { toast, fetchStatus, run } = setup(['cancelling', 'cancelling', 'cancelled']);

    const done = run('cancelling');
    await vi.advanceTimersByTimeAsync(0);

    expect(toast.info).toHaveBeenCalledTimes(1);
    expect(toast.info).toHaveBeenCalledWith(expect.stringMatching(/cancellation requested/i));
    expect(toast.success).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(INTERVAL * 3);
    await done;

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(toast.success).toHaveBeenCalledTimes(1);
    expect(toast.error).not.toHaveBeenCalled();
  });

  it('announces an error when the job ends in error while cancelling', async () => {
    const { toast, run } = setup(['cancelling', 'error']);

    const done = run('cancelling');
    await vi.advanceTimersByTimeAsync(INTERVAL * 3);
    await done;

    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(toast.error).toHaveBeenCalledWith(expect.stringContaining(JOB));
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('keeps polling through a failed status check', async () => {
    const { toast, fetchStatus, run } = setup([new Error('network'), 'cancelling', 'cancelled']);

    const done = run('cancelling');
    await vi.advanceTimersByTimeAsync(INTERVAL * 4);
    await done;

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(toast.success).toHaveBeenCalledTimes(1);
  });

  it('stops polling and says so when the job is still cancelling at the deadline', async () => {
    const { toast, fetchStatus, run } = setup(['cancelling']);

    const done = run('cancelling');
    await vi.advanceTimersByTimeAsync(TIMEOUT + INTERVAL);
    await done;
    const callsAtDeadline = fetchStatus.mock.calls.length;

    expect(toast.info).toHaveBeenCalledTimes(2);
    expect(toast.info).toHaveBeenLastCalledWith(expect.stringMatching(/still cancelling/i));
    expect(toast.success).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(INTERVAL * 5);
    expect(fetchStatus).toHaveBeenCalledTimes(callsAtDeadline);
  });

  it.each([
    ['every status check failed', [new Error('network')]],
    ['the last status seen was active', ['active']],
  ])('does not claim it is still cancelling when %s', async (_, statuses) => {
    const { toast, run } = setup(statuses);

    const done = run('cancelling');
    await vi.advanceTimersByTimeAsync(TIMEOUT + INTERVAL);
    await done;

    expect(toast.info).toHaveBeenCalledTimes(2);
    expect(toast.info).toHaveBeenLastCalledWith(expect.stringContaining(JOB));
    expect(toast.info).not.toHaveBeenLastCalledWith(expect.stringMatching(/still cancelling/i));
  });
});
