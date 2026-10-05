// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobResultResponse } from '@nemo/sdk/generated/platform/schema';
import { resolveArtifactItems } from '@studio/routes/JobDetailRoute/utils';

const makeResult = (overrides: Partial<HelixJobResultResponse> = {}): HelixJobResultResponse => ({
  name: 'result',
  job: 'job-1',
  workspace: 'default',
  artifact_url: '',
  artifact_storage_type: 'fileset',
  ...overrides,
});

describe('resolveArtifactItems', () => {
  const WORKSPACE = 'default';

  it('returns an empty list when there are no results', () => {
    expect(resolveArtifactItems([], WORKSPACE)).toEqual([]);
  });

  it('resolves a fileset reference into an ArtifactItem', () => {
    expect(
      resolveArtifactItems(
        [
          makeResult({
            name: 'metrics',
            artifact_url: 'default/training-fs#results/attempt-1/metrics.json',
          }),
        ],
        WORKSPACE
      )
    ).toEqual([
      {
        resultName: 'metrics',
        workspace: 'default',
        fileset: 'training-fs',
        objectPath: 'results/attempt-1/metrics.json',
      },
    ]);
  });

  it('returns one item per result, preserving order', () => {
    const items = resolveArtifactItems(
      [
        makeResult({ name: 'a', artifact_url: 'default/fs#results/a.json' }),
        makeResult({ name: 'b', artifact_url: 'default/fs#results/b' }),
      ],
      WORKSPACE
    );
    expect(items.map((i) => i.resultName)).toEqual(['a', 'b']);
  });

  it('uses workspaceFallback when the artifact_url omits a workspace', () => {
    expect(
      resolveArtifactItems(
        [makeResult({ artifact_url: 'training-fs#results/metrics.json' })],
        WORKSPACE
      )[0]
    ).toMatchObject({ workspace: 'default', fileset: 'training-fs' });
  });

  it.each([
    ['missing artifact_url', { artifact_url: '' }],
    ['non-fileset URL scheme', { artifact_url: 'file:///tmp/output.json' }],
    ['root-only fileset reference', { artifact_url: 'default/training-fs' }],
    [
      'non-fileset storage type',
      {
        artifact_url: 'default/training-fs#results/metrics.json',
        // Cast — current FileStorageType only declares 'fileset', but the
        // schema is open-ended and the guard should still be defensive.
        artifact_storage_type: 'opaque' as never,
      },
    ],
  ])('skips a result with %s', (_label, overrides) => {
    expect(resolveArtifactItems([makeResult(overrides)], WORKSPACE)).toEqual([]);
  });
});
