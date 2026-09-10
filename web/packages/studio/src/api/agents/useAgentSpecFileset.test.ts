// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { agentSpecSource } from '@studio/api/agents/useAgentSpecFileset';

const fileset = (storage: unknown): FilesetOutput =>
  ({ name: 'calc-ethos', workspace: 'ws', storage }) as FilesetOutput;

describe('agentSpecSource', () => {
  it('reads the repository, the tracked ref, and the pinned revision', () => {
    const source = agentSpecSource(
      fileset({
        type: 'github',
        owner: 'acme',
        repo: 'agents',
        revision: 'a'.repeat(40),
        original_revision: 'main',
        path: '',
      })
    );

    expect(source).toMatchObject({
      repository: 'acme/agents',
      trackedRevision: 'main',
      revision: 'a'.repeat(40),
    });
  });

  it('appends the sub-directory the fileset is scoped to', () => {
    const source = agentSpecSource(
      fileset({
        type: 'github',
        owner: 'acme',
        repo: 'agents',
        revision: 'abc',
        path: 'agents/calc',
      })
    );

    expect(source?.repository).toBe('acme/agents/agents/calc');
    expect(source?.webUrl).toBe('https://github.com/acme/agents/tree/abc/agents/calc');
  });

  it('reports no tracked ref for a fileset pinned to a commit', () => {
    const source = agentSpecSource(
      fileset({ type: 'github', owner: 'acme', repo: 'agents', revision: 'abc123' })
    );

    expect(source?.trackedRevision).toBeUndefined();
  });

  it('ignores a fileset that is not repository-backed', () => {
    expect(agentSpecSource(fileset({ type: 'local', path: '/data/calc' }))).toBeUndefined();
    expect(agentSpecSource(undefined)).toBeUndefined();
  });
});
