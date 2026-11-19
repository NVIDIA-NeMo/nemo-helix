// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AgentNetworkPolicy,
  AgentSandbox,
  NetworkRequest,
} from '@studio/routes/agents/AgentDetailRoute/network/types';
import {
  agentYamlFragment,
  binaryName,
  pendingCount,
  requestTarget,
  ruleBinaries,
  ruleEndpoints,
  sortRequests,
  sortSandboxes,
} from '@studio/routes/agents/AgentDetailRoute/network/utils';
import { parse } from 'yaml';

const request = (overrides: Partial<NetworkRequest> = {}): NetworkRequest => ({
  id: 'c1',
  status: 'pending',
  rule_name: 'allow_example_com_443',
  endpoints: ['example.com:443'],
  binary: '/usr/local/bin/python3.13',
  rationale: '',
  security_notes: '',
  rejection_reason: '',
  hit_count: 1,
  first_seen: null,
  last_seen: null,
  decided: null,
  rule: {
    name: 'allow_example_com_443',
    endpoints: [{ host: 'example.com', port: 443 }],
    binaries: [{ path: '/usr/local/bin/python3.13' }],
  },
  remembered: false,
  ...overrides,
});

const sandbox = (name: string, overrides: Partial<AgentSandbox> = {}): AgentSandbox => ({
  sandbox: `nhx-${name}`,
  kind: 'job',
  name,
  phase: 'ready',
  requests: [],
  error: '',
  ...overrides,
});

describe('sortRequests', () => {
  it('puts requests that need a decision first, then the most recently tried', () => {
    const sorted = sortRequests([
      request({ id: 'rejected', status: 'rejected' }),
      request({ id: 'approved', status: 'approved' }),
      request({ id: 'older', last_seen: '2026-11-19T20:00:00Z' }),
      request({ id: 'newer', last_seen: '2026-11-19T20:05:00Z' }),
    ]);

    expect(sorted.map(({ id }) => id)).toEqual(['newer', 'older', 'approved', 'rejected']);
  });

  it('keeps a status it does not know after the known ones', () => {
    const sorted = sortRequests([request({ id: 'odd', status: 'superseded' }), request()]);

    expect(sorted.map(({ id }) => id)).toEqual(['c1', 'odd']);
  });
});

describe('sortSandboxes', () => {
  it('puts a sandbox with a pending request ahead of running and finished ones', () => {
    const sorted = sortSandboxes([
      sandbox('finished', { phase: 'completed', requests: [request({ status: 'approved' })] }),
      sandbox('running'),
      sandbox('blocked', { phase: 'error', requests: [request()] }),
    ]);

    expect(sorted.map(({ name }) => name)).toEqual(['blocked', 'running', 'finished']);
  });
});

describe('pendingCount', () => {
  it('counts pending requests across every sandbox', () => {
    const policy: AgentNetworkPolicy = {
      rules: {},
      agent_yaml: '',
      gateway_error: '',
      sandboxes: [
        sandbox('a', { requests: [request(), request({ id: 'c2', status: 'approved' })] }),
        sandbox('b', { requests: [request({ id: 'c3' })] }),
      ],
    };

    expect(pendingCount(policy)).toBe(2);
    expect(pendingCount(undefined)).toBe(0);
  });
});

describe('request and rule text', () => {
  it('names a request by its destinations, or its rule when it has none', () => {
    expect(requestTarget(request())).toBe('example.com:443');
    expect(requestTarget(request({ endpoints: [] }))).toBe('allow_example_com_443');
  });

  it('lists a rule’s endpoints and binaries', () => {
    const { rule } = request();

    expect(ruleEndpoints(rule)).toEqual(['example.com:443']);
    expect(ruleBinaries(rule)).toEqual(['/usr/local/bin/python3.13']);
    expect(ruleEndpoints({})).toEqual([]);
  });

  it('names an executable by its file name', () => {
    expect(binaryName('/usr/local/bin/python3.13')).toBe('python3.13');
    expect(binaryName('curl')).toBe('curl');
  });
});

describe('agentYamlFragment', () => {
  it('writes the rule where agent.yaml registration reads it', () => {
    const { rule } = request();

    expect(parse(agentYamlFragment('allow_example_com_443', rule))).toEqual({
      helix_registration: { network_policies: { allow_example_com_443: rule } },
    });
  });
});
