// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  deriveRecommendations,
  type Mitigations,
} from '@agent-hardener/components/useMitigations';

// Mirrors the guardrails defender's component_writer: a `custom_guardrail_<n>` entry appended to the
// victim's NeMo Relay plugin config.
const withGuardrail = (name: string, tool: string, instructions: string): string =>
  [
    'version = 1',
    '[[components]]',
    'kind = "agent_hardener.pre_tool_verifier"',
    'enabled = true',
    '[[components.config.guardrails]]',
    `name = "${name}"`,
    `target_tool = "${tool}"`,
    `system_instructions = "${instructions}"`,
  ].join('\n');

describe('deriveRecommendations', () => {
  it('returns nothing without mitigations', () => {
    expect(deriveRecommendations(undefined)).toEqual([]);
  });

  it('surfaces each guardrail added by the run, tagged with its target tool', () => {
    const mitigations: Mitigations = {
      guardrails: {
        before: 'version = 1\n',
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

  it('ignores guardrails already present in the baseline', () => {
    const guardrail = withGuardrail('custom_guardrail_1', 'bash_executor', 'Refuse.');
    expect(deriveRecommendations({ guardrails: { before: guardrail, after: guardrail } })).toEqual(
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

  it('tolerates a malformed guardrail file without throwing', () => {
    expect(deriveRecommendations({ guardrails: { before: '::: not toml', after: ':::' } })).toEqual(
      []
    );
  });
});
