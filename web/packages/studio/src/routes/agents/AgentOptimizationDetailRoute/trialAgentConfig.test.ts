// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import {
  applyTrialToAgentConfig,
  coerceParamValue,
  parseSearchSpace,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/trialAgentConfig';

const trial = (params: Record<string, string>): Trial => ({
  number: 3,
  state: 'COMPLETE',
  durationSeconds: 10,
  paretoOptimal: true,
  metrics: [],
  params: Object.entries(params).map(([name, value]) => ({ name, value })),
});

const specAgent = (harnessModel?: Record<string, unknown>) => ({
  config_format: 'nemo-agents-spec-v1',
  name: 'hermes',
  default_harness: 'main',
  harnesses: { main: { kind: 'hermes', ...(harnessModel ? { model: harnessModel } : {}) } },
  models: { default: { provider: 'nvidia', model: 'llama', temperature: 0.7 } },
});

describe('parseSearchSpace', () => {
  it('keys entries by param name and keeps categorical values', () => {
    expect(
      parseSearchSpace({
        optimizer: {
          search_space: {
            temperature: { type: 'fabric', path: 'models.default.temperature', values: [0, 0.2] },
            top_p: { type: 'fabric', path: 'models.default.top_p', low: 0.1, high: 1 },
            broken: { type: 'fabric' },
          },
        },
      })
    ).toEqual({
      temperature: { path: 'models.default.temperature', values: [0, 0.2] },
      top_p: { path: 'models.default.top_p', values: undefined },
    });
  });

  it('returns an empty space when the optimizer section is missing', () => {
    expect(parseSearchSpace({})).toEqual({});
  });
});

describe('coerceParamValue', () => {
  it('restores the declared categorical choice', () => {
    expect(coerceParamValue('fast', { path: 'p', values: ['fast', 'slow'] })).toBe('fast');
    expect(coerceParamValue('0.2', { path: 'p', values: [0, 0.2] })).toBe(0.2);
  });

  it('parses numeric range values as numbers', () => {
    expect(coerceParamValue('512', { path: 'p' })).toBe(512);
    expect(coerceParamValue('0.35', { path: 'p' })).toBe(0.35);
  });
});

describe('applyTrialToAgentConfig', () => {
  const searchSpace = {
    temperature: { path: 'models.default.temperature' },
    max_tokens: { path: 'models.default.max_tokens' },
  };

  it('writes trial values onto models.default without mutating the source', () => {
    const source = specAgent();
    const next = applyTrialToAgentConfig(
      source,
      trial({ temperature: '0.2', max_tokens: '1024' }),
      searchSpace
    );

    expect(next.models).toEqual({
      default: { provider: 'nvidia', model: 'llama', temperature: 0.2, max_tokens: 1024 },
    });
    expect(source.models.default.temperature).toBe(0.7);
  });

  it('writes onto the default harness model when the harness pins one', () => {
    const next = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ temperature: '0.1' }),
      searchSpace
    );

    expect(next.harnesses).toEqual({
      main: { kind: 'hermes', model: { provider: 'nvidia', model: 'nemotron', temperature: 0.1 } },
    });
    expect((next.models as { default: { temperature: number } }).default.temperature).toBe(0.7);
  });

  it('rejects non-model paths on a platform agent', () => {
    expect(() =>
      applyTrialToAgentConfig(specAgent(), trial({ depth: '3' }), {
        depth: { path: 'adapter.settings.depth' },
      })
    ).toThrow(/only model settings are supported/);
  });

  it('rejects params that are not in the search space', () => {
    expect(() =>
      applyTrialToAgentConfig(specAgent(), trial({ unknown: '1' }), searchSpace)
    ).toThrow(/not in the study's search space/);
  });
});
