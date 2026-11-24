// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { endedJobFailure, jobFailure } from '@agent-hardener/api/jobFailure';

it('reads the message and remediation the backend wrote', () => {
  const job = { error_details: { message: ' victim answered HTTP 500 ', remediation: 'Check the victim logs.' } };
  expect(jobFailure(job)).toEqual({ message: 'victim answered HTTP 500', remediation: 'Check the victim logs.' });
});

it('leaves remediation out when the backend gave none', () => {
  expect(jobFailure({ error_details: { message: 'boom', remediation: '' } })).toEqual({
    message: 'boom',
    remediation: undefined,
  });
});

it('is undefined for a job with no usable message', () => {
  expect(jobFailure(undefined)).toBeUndefined();
  expect(jobFailure({ error_details: null })).toBeUndefined();
  expect(jobFailure({ error_details: { message: 42 } })).toBeUndefined();
});

describe('endedJobFailure', () => {
  it('is undefined while the job runs and once it completed', () => {
    expect(endedJobFailure({ status: 'active' })).toBeUndefined();
    expect(endedJobFailure({ status: 'completed', error_details: { message: 'stale' } })).toBeUndefined();
    expect(endedJobFailure(undefined)).toBeUndefined();
  });

  it("gives a failed job's own message and remediation", () => {
    const job = { status: 'error', error_details: { message: 'sandbox failed', remediation: 'Check Docker.' } };
    expect(endedJobFailure(job)).toEqual({ message: 'sandbox failed', remediation: 'Check Docker.' });
  });

  it('still says something when a failed job left no message', () => {
    // Studio used to show nothing at all here, so repeat clicks queued more failing jobs.
    expect(endedJobFailure({ status: 'error', error_details: null })?.message).toMatch(/without completing/);
  });

  it('calls a cancelled job cancelled, not failed with a stale message', () => {
    expect(endedJobFailure({ status: 'cancelled', error_details: { message: 'boom' } })).toEqual({
      message: 'The job was cancelled.',
    });
  });
});
