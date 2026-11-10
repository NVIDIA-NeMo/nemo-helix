// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { agentIdentityForTarget, agentNameForTarget } from '@studio/api/evaluation/agentTarget';

const OTHER_WORKSPACE = 'other-ws';

describe('registered agent target identity', () => {
  it.each(['agent', 'fabric', 'harbor', 'gym'])('resolves %s targets consistently', (kind) => {
    for (const name of ['my-agent', `${DEFAULT_WORKSPACE}/my-agent`]) {
      const target =
        kind === 'agent' ? { kind, agent: { name } } : { kind, source: { agent: name } };
      expect(agentIdentityForTarget(target, DEFAULT_WORKSPACE)).toBe(
        `${DEFAULT_WORKSPACE}/my-agent`
      );
      expect(agentNameForTarget(target, DEFAULT_WORKSPACE)).toBe('my-agent');
    }
  });

  it.each(['agent', 'fabric', 'harbor', 'gym'])(
    'preserves foreign workspace identity for %s',
    (kind) => {
      const name = `${OTHER_WORKSPACE}/my-agent`;
      const target =
        kind === 'agent' ? { kind, agent: { name } } : { kind, source: { agent: name } };
      expect(agentIdentityForTarget(target, DEFAULT_WORKSPACE)).toBe(name);
      expect(agentNameForTarget(target, DEFAULT_WORKSPACE)).not.toBe('my-agent');
    }
  );

  it.each([
    undefined,
    null,
    {},
    { kind: 'model', name: 'my-agent', source: { agent: 'my-agent' } },
    { kind: 'endpoint', name: 'my-agent' },
    { kind: 'agent', name: 'my-agent' },
    { kind: 'agent', agent: { name: '' } },
    { kind: 'fabric', name: 'my-agent', source: { config: {} } },
    { kind: 'harbor', source: { agent: { name: 'my-agent' } } },
    { kind: 'gym', source: { agent: '  ' } },
    { format: 'generic', name: 'my-agent', url: 'https://api.example.com/v1/chat/completions' },
  ])('does not attribute unrelated or malformed targets: %j', (target) => {
    expect(agentIdentityForTarget(target, DEFAULT_WORKSPACE)).toBeNull();
    expect(agentNameForTarget(target, DEFAULT_WORKSPACE)).toBeNull();
  });

  it("reads a generic endpoint's agent from the platform route it calls, not its label", () => {
    const target = {
      format: 'generic',
      name: 'display-label',
      url: `http://localhost:8080/apis/agents/v2/workspaces/${DEFAULT_WORKSPACE}/agents/my-agent/-/v1/chat/completions`,
    };
    expect(agentNameForTarget(target, DEFAULT_WORKSPACE)).toBe('my-agent');
    expect(
      agentNameForTarget(
        { ...target, url: target.url.replace(DEFAULT_WORKSPACE, 'other') },
        DEFAULT_WORKSPACE
      )
    ).toBe('other/my-agent');
  });
});
