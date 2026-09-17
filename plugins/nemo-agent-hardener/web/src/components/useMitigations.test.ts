// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  deriveRecommendations,
  type Mitigations,
} from '@agent-hardener/components/useMitigations';

// Mirrors the guardrails defender's yaml_writer: a `custom_guardrail_<n>` middleware added under `middleware:`.
const withGuardrail = (name: string, tool: string, instructions: string): string =>
  [
    'middleware:',
    `  ${name}:`,
    '    _type: pre_tool_verifier',
    `    target_function_or_group: ${tool}`,
    `    system_instructions: ${instructions}`,
  ].join('\n');

describe('deriveRecommendations', () => {
  it('returns nothing without mitigations', () => {
    expect(deriveRecommendations(undefined)).toEqual([]);
  });

  it('surfaces each guardrail added to the workflow, tagged with its target tool', () => {
    const mitigations: Mitigations = {
      workflow: {
        before: 'middleware: {}\n',
        after: withGuardrail(
          'custom_guardrail_1',
          'bash_executor',
          'Refuse destructive shell commands. Then stop.'
        ),
      },
    };
    const recs = deriveRecommendations(mitigations);
    expect(recs).toHaveLength(1);
    expect(recs[0]?.title).toBe('Added a guardrail on bash_executor');
    expect(recs[0]?.detail).toBe('Refuse destructive shell commands.'); // first sentence only
  });

  it('ignores guardrails already present in the baseline workflow', () => {
    const guardrail = withGuardrail('custom_guardrail_1', 'bash_executor', 'Refuse.');
    expect(deriveRecommendations({ workflow: { before: guardrail, after: guardrail } })).toEqual(
      []
    );
  });

  it('summarizes the policy diff with factual per-section deltas (no "hardened" claim)', () => {
    const before = 'filesystem_policy:\n  read_only: [/usr, /etc]\n  read_write: [/sandbox]\n';
    const after = 'filesystem_policy:\n  read_only: []\n  read_write: [/sandbox]\n';
    const recs = deriveRecommendations({ policy: { before, after } });
    expect(recs).toHaveLength(1);
    expect(recs[0]?.title).toBe('OpenShell policy changes');
    expect(recs[0]?.detail).toContain('Filesystem read-only paths: 2 → 0');
    expect(recs[0]?.detail).not.toMatch(/hardened|tightened/i);
  });

  it('tolerates malformed YAML without throwing', () => {
    expect(deriveRecommendations({ workflow: { before: '::: not yaml', after: ':::' } })).toEqual(
      []
    );
  });
});
