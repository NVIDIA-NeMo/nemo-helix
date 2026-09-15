// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  applyAgentName,
  parseAtifDocument,
  parseAtifDocuments,
  reattributedFrom,
} from '@studio/components/ImportTracesModal/parseAtifTraces';

const trajectory = (overrides: Record<string, unknown> = {}) => ({
  schema_version: 'ATIF-v1.5',
  agent: { name: 'email-security-triage', version: '0.7.0' },
  steps: [
    { step_id: 1, timestamp: '2026-08-26T21:50:23Z', source: 'user', message: 'is this legit?' },
    { step_id: 2, timestamp: '2026-08-26T21:50:29Z', source: 'agent', message: 'phishing' },
  ],
  ...overrides,
});

describe('parseAtifDocument', () => {
  it('accepts a single trajectory', () => {
    const { traces, failures } = parseAtifDocument('trace-01.json', JSON.stringify(trajectory()));

    expect(failures).toEqual([]);
    expect(traces).toHaveLength(1);
    expect(traces[0].label).toBe('trace-01.json');
    expect(traces[0].trajectory.agent.name).toBe('email-security-triage');
  });

  it('accepts an array and indexes each entry', () => {
    const { traces, failures } = parseAtifDocument(
      'batch.json',
      JSON.stringify([trajectory(), trajectory()])
    );

    expect(failures).toEqual([]);
    expect(traces.map(({ label }) => label)).toEqual(['batch.json [1]', 'batch.json [2]']);
  });

  it('treats whitespace-only input as no input rather than a failure', () => {
    expect(parseAtifDocument('pasted', '   \n ')).toEqual({ traces: [], failures: [] });
  });

  it('reports malformed JSON against its label', () => {
    const { traces, failures } = parseAtifDocument('trace-01.json', '{ nope');

    expect(traces).toEqual([]);
    expect(failures).toHaveLength(1);
    expect(failures[0].label).toBe('trace-01.json');
  });

  it.each([
    [{ schema_version: undefined }, /schema_version/],
    [{ schema_version: 'ATIF-v9.9' }, /Unsupported schema_version/],
    [{ agent: undefined }, /"agent"/],
    [{ agent: { version: '1.0.0' } }, /agent\.name/],
    [{ steps: {} }, /must be an array/],
  ])('rejects %j', (overrides, expected) => {
    const { traces, failures } = parseAtifDocument(
      'trace.json',
      JSON.stringify(trajectory(overrides))
    );

    expect(traces).toEqual([]);
    expect(failures[0].message).toMatch(expected);
  });

  it('rejects steps that are not 1-based and sequential', () => {
    const { failures } = parseAtifDocument(
      'trace.json',
      JSON.stringify(trajectory({ steps: [{ step_id: 1 }, { step_id: 3 }] }))
    );

    expect(failures[0].message).toMatch(/Step 2 must be an object with "step_id": 2/);
  });

  it('keeps valid entries alongside invalid ones', () => {
    const { traces, failures } = parseAtifDocument(
      'batch.json',
      JSON.stringify([trajectory(), { schema_version: 'ATIF-v1.5' }])
    );

    expect(traces.map(({ label }) => label)).toEqual(['batch.json [1]']);
    expect(failures.map(({ label }) => label)).toEqual(['batch.json [2]']);
  });
});

describe('parseAtifDocuments', () => {
  it('flattens several documents, preserving input order', () => {
    const { traces, failures } = parseAtifDocuments([
      { label: 'a.json', text: JSON.stringify(trajectory()) },
      { label: 'b.json', text: 'not json' },
      { label: 'c.json', text: JSON.stringify(trajectory()) },
      { label: 'Pasted JSON', text: '' },
    ]);

    expect(traces.map(({ label }) => label)).toEqual(['a.json', 'c.json']);
    expect(failures.map(({ label }) => label)).toEqual(['b.json']);
  });
});

describe('applyAgentName', () => {
  it('overrides the agent name and keeps the rest of the agent block', () => {
    const { traces } = parseAtifDocument('trace.json', JSON.stringify(trajectory()));

    const reattributed = applyAgentName(traces[0].trajectory, 'recipe-agent');

    expect(reattributed.agent).toEqual({ name: 'recipe-agent', version: '0.7.0' });
    expect(reattributed.steps).toBe(traces[0].trajectory.steps);
  });

  it('does not mutate the source trajectory', () => {
    const { traces } = parseAtifDocument('trace.json', JSON.stringify(trajectory()));

    applyAgentName(traces[0].trajectory, 'recipe-agent');

    expect(traces[0].trajectory.agent.name).toBe('email-security-triage');
  });
});

describe('reattributedFrom', () => {
  it('names the original agent when it differs', () => {
    const { traces } = parseAtifDocument('trace.json', JSON.stringify(trajectory()));

    expect(reattributedFrom(traces[0].trajectory, 'recipe-agent')).toBe('email-security-triage');
  });

  it('is undefined when the trajectory already names the target agent', () => {
    const { traces } = parseAtifDocument('trace.json', JSON.stringify(trajectory()));

    expect(reattributedFrom(traces[0].trajectory, 'email-security-triage')).toBeUndefined();
  });
});
