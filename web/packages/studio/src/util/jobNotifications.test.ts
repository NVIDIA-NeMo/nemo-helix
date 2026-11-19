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
      sound: 'success',
    });
  });

  it('announces a failed job', () => {
    expect(buildJobNotification(job('a', HelixJobStatus.error))).toMatchObject({
      title: 'Job failed',
      sound: 'failure',
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
  const SESSION_START = '2026-11-10T12:00:00Z';
  const BEFORE_SESSION = '2026-11-10T11:00:00Z';
  const DURING_SESSION = '2026-11-10T12:05:00Z';

  const options = (alreadyNotified: string[] = []) => ({
    startedSince: SESSION_START,
    alreadyNotified: new Set(alreadyNotified),
  });
  const jobAt = (name: string, status: HelixJobStatus, created_at: string): NotifiableJob => ({
    ...job(name, status),
    created_at,
  });

  it('does not report a job from before the session that is first seen finished', () => {
    const { finished, watched } = findJobTransitions(
      [jobAt('old', HelixJobStatus.completed, BEFORE_SESSION)],
      new Set(),
      options()
    );
    expect(finished).toEqual([]);
    expect(watched.size).toBe(0);
  });

  it('does not report a finished job whose creation time is unknown', () => {
    const { finished } = findJobTransitions(
      [job('no-date', HelixJobStatus.completed)],
      new Set(),
      options()
    );
    expect(finished).toEqual([]);
  });

  it('reports a job started this session even if it was never seen running', () => {
    const { finished } = findJobTransitions(
      [jobAt('fast-fail', HelixJobStatus.error, DURING_SESSION)],
      new Set(),
      options()
    );
    expect(finished.map((j) => j.name)).toEqual(['fast-fail']);
  });

  it('watches a running job, then reports it once it completes', () => {
    const first = findJobTransitions(
      [jobAt('a', HelixJobStatus.active, BEFORE_SESSION)],
      new Set(),
      options()
    );
    expect(first.watched.has('a')).toBe(true);
    expect(first.finished).toEqual([]);

    // Created before the session, but it was seen running, so it is still announced.
    const second = findJobTransitions(
      [jobAt('a', HelixJobStatus.completed, BEFORE_SESSION)],
      first.watched,
      options()
    );
    expect(second.finished.map((j) => j.name)).toEqual(['a']);
    expect(second.watched.has('a')).toBe(false);
  });

  it('reports a watched job that fails', () => {
    const { finished } = findJobTransitions(
      [job('a', HelixJobStatus.error)],
      new Set(['a']),
      options()
    );
    expect(finished.map((j) => j.status)).toEqual([HelixJobStatus.error]);
  });

  it('does not report a job that has already been announced', () => {
    const { finished } = findJobTransitions(
      [jobAt('a', HelixJobStatus.error, DURING_SESSION)],
      new Set(['a']),
      options(['a'])
    );
    expect(finished).toEqual([]);
  });

  it('does not mutate the previous set', () => {
    const previous = new Set(['a']);
    findJobTransitions([job('a', HelixJobStatus.completed)], previous, options());
    expect(previous.has('a')).toBe(true);
  });
});
