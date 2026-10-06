// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import {
  applyTrialToAgentConfig,
  coerceParamValue,
  parseStudyConfig,
  type SearchSpace,
  type StudyConfig,
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
  models: {
    default: { provider: 'nvidia', model: 'llama', temperature: 0.7 },
    fast: { provider: 'nvidia', model: 'llama-mini', temperature: 0.5 },
  },
});

const study = (
  searchSpace: SearchSpace,
  overlayModels: StudyConfig['overlayModels'] = {}
): StudyConfig => ({ searchSpace, overlayModels });

describe('parseStudyConfig', () => {
  it('keys entries by param name and keeps categorical values', () => {
    expect(
      parseStudyConfig({
        optimizer: {
          search_space: {
            temperature: { type: 'fabric', path: 'models.default.temperature', values: [0, 0.2] },
            top_p: { type: 'fabric', path: 'models.default.top_p', low: 0.1, high: 1 },
            broken: { type: 'fabric' },
          },
        },
      }).searchSpace
    ).toEqual({
      temperature: { path: 'models.default.temperature', values: [0, 0.2] },
      top_p: { path: 'models.default.top_p', values: undefined },
    });
  });

  it('collects the models the optimize config merges over the agent', () => {
    expect(
      parseStudyConfig({ optimizer: {}, models: { judge: { model: 'x' }, broken: 'nope' } })
        .overlayModels
    ).toEqual({ judge: { model: 'x' } });
  });

  it('returns an empty study config when the optimizer section is missing', () => {
    expect(parseStudyConfig({})).toEqual({ searchSpace: {}, overlayModels: {} });
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
  const searchSpace = study({
    temperature: { path: 'models.default.temperature' },
    max_tokens: { path: 'models.default.max_tokens' },
  });

  it('writes trial values onto models.default without mutating the source', () => {
    const source = specAgent();
    const { config: next, skipped } = applyTrialToAgentConfig(
      source,
      trial({ temperature: '0.2', max_tokens: '1024' }),
      searchSpace
    );

    expect(next.models).toEqual({
      default: { provider: 'nvidia', model: 'llama', temperature: 0.2, max_tokens: 1024 },
      fast: { provider: 'nvidia', model: 'llama-mini', temperature: 0.5 },
    });
    expect(source.models.default.temperature).toBe(0.7);
    expect(skipped).toEqual([]);
  });

  it('writes onto the default harness model when the harness pins one', () => {
    const { config: next } = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ temperature: '0.1' }),
      searchSpace
    );

    expect(next.harnesses).toEqual({
      main: { kind: 'hermes', model: { provider: 'nvidia', model: 'nemotron', temperature: 0.1 } },
    });
    expect((next.models as { default: { temperature: number } }).default.temperature).toBe(0.7);
  });

  it('writes a secondary model through under its own key', () => {
    const { config: next } = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ fast_temperature: '0.1' }),
      study({ fast_temperature: { path: 'models.fast.temperature' } })
    );

    expect((next.models as { fast: { temperature: number } }).fast.temperature).toBe(0.1);
  });

  it('applies values on a model the optimize config restates', () => {
    const { config: next, modelMismatches } = applyTrialToAgentConfig(
      specAgent(),
      trial({ temperature: '0.2' }),
      { ...searchSpace, overlayModels: { default: { provider: 'nvidia', model: 'llama' } } }
    );

    expect((next.models as { default: { temperature: number } }).default.temperature).toBe(0.2);
    expect(modelMismatches).toEqual([]);
  });

  it('reports when the study ran a model under a different id than the agent', () => {
    const { config: next, modelMismatches } = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ temperature: '0.2', max_tokens: '512' }),
      { ...searchSpace, overlayModels: { default: { model: 'nvidia/nemotron' } } }
    );

    expect(next.harnesses).toEqual({
      main: {
        kind: 'hermes',
        model: { provider: 'nvidia', model: 'nemotron', temperature: 0.2, max_tokens: 512 },
      },
    });
    expect(modelMismatches).toEqual([
      { modelKey: 'default', studyModel: 'nvidia/nemotron', agentModel: 'nemotron' },
    ]);
  });

  it('skips params on a model that only the optimize config defines', () => {
    const { config: next, skipped } = applyTrialToAgentConfig(
      specAgent(),
      trial({ temperature: '0.2', judge_temperature: '0' }),
      study(
        {
          temperature: { path: 'models.default.temperature' },
          judge_temperature: { path: 'models.judge.temperature' },
        },
        { judge: { model: 'judge-model' } }
      )
    );

    expect(skipped).toEqual(['judge_temperature']);
    expect(next.models).not.toHaveProperty('judge');
    expect((next.models as { default: { temperature: number } }).default.temperature).toBe(0.2);
  });

  it('rejects a trial whose params all tune study-only models', () => {
    expect(() =>
      applyTrialToAgentConfig(
        specAgent(),
        trial({ judge_temperature: '0' }),
        study(
          { judge_temperature: { path: 'models.judge.temperature' } },
          { judge: { model: 'judge-model' } }
        )
      )
    ).toThrow(/None of trial 3's parameters apply/);
  });

  it('rejects a model the agent does not define', () => {
    expect(() =>
      applyTrialToAgentConfig(
        specAgent(),
        trial({ slow_temperature: '0' }),
        study({ slow_temperature: { path: 'models.slow.temperature' } })
      )
    ).toThrow(/does not define/);
  });

  it('rejects non-model paths on a platform agent', () => {
    expect(() =>
      applyTrialToAgentConfig(
        specAgent(),
        trial({ depth: '3' }),
        study({ depth: { path: 'adapter.settings.depth' } })
      )
    ).toThrow(/only model settings are supported/);
  });

  it('rejects params that are not in the search space', () => {
    expect(() =>
      applyTrialToAgentConfig(specAgent(), trial({ unknown: '1' }), searchSpace)
    ).toThrow(/not in the study's search space/);
  });
});
