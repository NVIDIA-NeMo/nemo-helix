// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isQueuedTooLong,
  isTerminalPackageStatus,
  parsePackageResult,
  QUEUED_STALL_MS,
} from '@studio/api/agents/usePackageAgent';

describe('parsePackageResult', () => {
  it('reads the tag a deployment needs', () => {
    const result = parsePackageResult({
      image: 'nemo-agents/default/my-agent:1.0',
      agent: 'my-agent',
      published: 'nvcr.io/my-org/nemo-agents/default/my-agent:1.0',
    });

    expect(result).toEqual({
      image: 'nemo-agents/default/my-agent:1.0',
      agent: 'my-agent',
      published: 'nvcr.io/my-org/nemo-agents/default/my-agent:1.0',
    });
  });

  it('defaults published to empty when the job did not push', () => {
    expect(parsePackageResult({ image: 'my-agent:1.0', agent: 'my-agent', published: '' })).toEqual(
      {
        image: 'my-agent:1.0',
        agent: 'my-agent',
        published: '',
      }
    );
  });

  it.each([
    ['a missing image', { agent: 'my-agent' }],
    ['an empty image', { image: '', agent: 'my-agent' }],
    ['a whitespace-only image', { image: '   ', agent: 'my-agent' }],
    ['a non-string image', { image: 42 }],
    ['null', null],
    ['a bare number', 7],
    ['a string', 'nope'],
  ])('treats %s as no result rather than a blank tag', (_label, payload) => {
    expect(parsePackageResult(payload)).toBeUndefined();
  });

  it('trims a padded image so it matches what deployment validation accepts', () => {
    expect(parsePackageResult({ image: '  my-agent:1.0  ', agent: 'my-agent' })?.image).toBe(
      'my-agent:1.0'
    );
  });

  it('ignores non-string companions instead of leaking them into the tag', () => {
    expect(parsePackageResult({ image: 'my-agent:1.0', agent: 7, published: null })).toEqual({
      image: 'my-agent:1.0',
      agent: '',
      published: '',
    });
  });
});

describe('isTerminalPackageStatus', () => {
  it.each(['completed', 'error', 'cancelled'])('stops polling on %s', (status) => {
    expect(isTerminalPackageStatus(status)).toBe(true);
  });

  it.each(['created', 'pending', 'active'])('keeps polling on %s', (status) => {
    expect(isTerminalPackageStatus(status)).toBe(false);
  });

  it('keeps polling before the first status arrives', () => {
    expect(isTerminalPackageStatus(undefined)).toBe(false);
  });
});

describe('isQueuedTooLong', () => {
  const submittedAt = 1_000_000;

  it('warns once a created job has waited past the threshold', () => {
    expect(isQueuedTooLong('created', submittedAt, submittedAt + QUEUED_STALL_MS + 1)).toBe(true);
  });

  it('stays quiet while the job is still plausibly queued', () => {
    expect(isQueuedTooLong('created', submittedAt, submittedAt + QUEUED_STALL_MS)).toBe(false);
  });

  it.each(['active', 'completed', 'error'])('never warns for %s, which is not stuck', (status) => {
    expect(isQueuedTooLong(status, submittedAt, submittedAt + QUEUED_STALL_MS * 10)).toBe(false);
  });

  it('needs a submit time before it can judge', () => {
    expect(isQueuedTooLong('created', undefined, submittedAt + QUEUED_STALL_MS * 10)).toBe(false);
  });
});
