// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  buildJobNotification,
  findJobTransitions,
  getJobNotificationPath,
  type NotifiableJob,
} from '@studio/util/jobNotifications';

const job = (name: string, status: HelixJobStatus, source = 'customization'): NotifiableJob => ({
  name,
  status,
  source,
});

describe('getJobNotificationPath', () => {
  it.each([
    ['customization', `/workspaces/${DEFAULT_WORKSPACE}/fine-tune/my-job`],
    ['data-designer', `/workspaces/${DEFAULT_WORKSPACE}/data-designer/my-job`],
    ['safe-synthesizer', `/workspaces/${DEFAULT_WORKSPACE}/safe-synthesizer/job/my-job`],
    ['evaluator-metrics', `/workspaces/${DEFAULT_WORKSPACE}/jobs/my-job`],
    ['anonymizer', `/workspaces/${DEFAULT_WORKSPACE}/jobs/my-job`],
  ])('routes a %s job to %s', (source, expected) => {
    expect(
      getJobNotificationPath(job('my-job', HelixJobStatus.completed, source), DEFAULT_WORKSPACE)
    ).toBe(expected);
  });
});

describe('buildJobNotification', () => {
  it('announces a completed job', () => {
    expect(buildJobNotification(job('a', HelixJobStatus.completed))).toMatchObject({
      title: 'Job completed',
      tag: 'a',
    });
  });

  it('announces a failed job', () => {
    expect(buildJobNotification(job('a', HelixJobStatus.error))).toMatchObject({
      title: 'Job failed',
    });
  });

  it.each([HelixJobStatus.cancelled, HelixJobStatus.active, HelixJobStatus.queued])(
    'stays quiet for %s',
    (status) => {
      expect(buildJobNotification(job('a', status))).toBeNull();
    }
  );
});

describe('findJobTransitions', () => {
  it('does not report a job first seen already finished', () => {
    const { finished, watched } = findJobTransitions(
      [job('old', HelixJobStatus.completed)],
      new Set()
    );
    expect(finished).toEqual([]);
    expect(watched.size).toBe(0);
  });

  it('watches a running job, then reports it once it completes', () => {
    const first = findJobTransitions([job('a', HelixJobStatus.active)], new Set());
    expect(first.watched.has('a')).toBe(true);
    expect(first.finished).toEqual([]);

    const second = findJobTransitions([job('a', HelixJobStatus.completed)], first.watched);
    expect(second.finished.map((j) => j.name)).toEqual(['a']);
    expect(second.watched.has('a')).toBe(false);
  });

  it('reports a watched job that fails', () => {
    const { finished } = findJobTransitions([job('a', HelixJobStatus.error)], new Set(['a']));
    expect(finished.map((j) => j.status)).toEqual([HelixJobStatus.error]);
  });

  it('reports a finished job only once', () => {
    const first = findJobTransitions([job('a', HelixJobStatus.completed)], new Set(['a']));
    const again = findJobTransitions([job('a', HelixJobStatus.completed)], first.watched);
    expect(again.finished).toEqual([]);
  });

  it('does not mutate the previous set', () => {
    const previous = new Set(['a']);
    findJobTransitions([job('a', HelixJobStatus.completed)], previous);
    expect(previous.has('a')).toBe(true);
  });
});
