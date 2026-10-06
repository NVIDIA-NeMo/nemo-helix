// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import {
  applyTrialToAgentConfig,
  coerceParamValue,
  parseStudyConfig,
  type SearchSpaceEntry,
  type StudyConfig,
  studyConfigLocation,
  withAgentName,
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
  searchSpace: Record<string, SearchSpaceEntry>,
  overlayModels: Record<string, Record<string, unknown>> = {}
): StudyConfig => ({
  searchSpace: new Map(Object.entries(searchSpace)),
  overlayModels: new Map(Object.entries(overlayModels)),
});

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
    ).toEqual(
      new Map([
        ['temperature', { path: 'models.default.temperature', values: [0, 0.2] }],
        ['top_p', { path: 'models.default.top_p', values: undefined }],
      ])
    );
  });

  it('collects the models the optimize config merges over the agent', () => {
    expect(
      parseStudyConfig({ optimizer: {}, models: { judge: { model: 'x' }, broken: 'nope' } })
        .overlayModels
    ).toEqual(new Map([['judge', { model: 'x' }]]));
  });

  it('returns an empty study config when the optimizer section is missing', () => {
    expect(parseStudyConfig({})).toEqual({ searchSpace: new Map(), overlayModels: new Map() });
  });
});

describe('studyConfigLocation', () => {
  it('resolves a bare fileset name against the study workspace', () => {
    expect(
      studyConfigLocation(
        { strategy: 'legacy', optimize_config: 'optimize.yaml', optimize_config_fileset: 'bundle' },
        'ws'
      )
    ).toEqual({ workspace: 'ws', fileset: 'bundle', path: 'optimize.yaml' });
  });

  it('keeps the workspace of a qualified fileset ref', () => {
    expect(
      studyConfigLocation(
        {
          strategy: 'legacy',
          optimize_config: 'optimize.yaml',
          optimize_config_fileset: 'other/bundle',
        },
        'ws'
      )
    ).toEqual({ workspace: 'other', fileset: 'bundle', path: 'optimize.yaml' });
  });

  it('is undefined without a fileset or config path', () => {
    expect(studyConfigLocation({ strategy: 'legacy', optimize_config: 'o.yaml' }, 'ws')).toBe(
      undefined
    );
    expect(studyConfigLocation({ strategy: 'legacy', optimize_config_fileset: 'b' }, 'ws')).toBe(
      undefined
    );
  });
});

describe('coerceParamValue', () => {
  it('restores the declared categorical choice', () => {
    expect(coerceParamValue('fast', { path: 'p', values: ['fast', 'slow'] })).toBe('fast');
    expect(coerceParamValue('0.2', { path: 'p', values: [0, 0.2] })).toBe(0.2);
  });

  it('matches the way Python writes booleans and floats', () => {
    expect(coerceParamValue('False', { path: 'p', values: [true, false] })).toBe(false);
    expect(coerceParamValue('True', { path: 'p', values: [true, false] })).toBe(true);
    expect(coerceParamValue('1.0', { path: 'p', values: [0.5, 1] })).toBe(1);
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
    const { config: next, addedModels } = applyTrialToAgentConfig(
      source,
      trial({ temperature: '0.2', max_tokens: '1024' }),
      searchSpace
    );

    expect(next.models).toEqual({
      default: { provider: 'nvidia', model: 'llama', temperature: 0.2, max_tokens: 1024 },
      fast: { provider: 'nvidia', model: 'llama-mini', temperature: 0.5 },
    });
    expect(source.models.default.temperature).toBe(0.7);
    expect(addedModels).toEqual([]);
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

  it('writes onto a harness whose name contains a dot', () => {
    const source = {
      ...specAgent(),
      default_harness: 'main.v2',
      harnesses: { 'main.v2': { kind: 'hermes', model: { provider: 'nvidia', model: 'x' } } },
    };
    const { config: next } = applyTrialToAgentConfig(
      source,
      trial({ temperature: '0.1' }),
      searchSpace
    );

    expect(next.harnesses).toEqual({
      'main.v2': { kind: 'hermes', model: { provider: 'nvidia', model: 'x', temperature: 0.1 } },
    });
  });

  it('writes a secondary model through under its own key', () => {
    const { config: next } = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ fast_temperature: '0.1' }),
      study({ fast_temperature: { path: 'models.fast.temperature' } })
    );

    expect((next.models as { fast: { temperature: number } }).fast.temperature).toBe(0.1);
  });

  it('deploys a categorical boolean as the boolean the trial ran', () => {
    const { config: next } = applyTrialToAgentConfig(
      specAgent(),
      trial({ stream: 'False' }),
      study({ stream: { path: 'models.default.settings.stream', values: [true, false] } })
    );

    expect(next.models).toMatchObject({ default: { settings: { stream: false } } });
  });

  it('reports nothing when the optimize config restates the agent model as is', () => {
    const { modelDifferences } = applyTrialToAgentConfig(
      specAgent(),
      trial({ temperature: '0.2' }),
      study(
        { temperature: { path: 'models.default.temperature' } },
        { default: { provider: 'nvidia', model: 'llama', temperature: 0.7 } }
      )
    );

    expect(modelDifferences).toEqual([]);
  });

  it('reports every setting the study ran differently, keeping the agent settings', () => {
    const { config: next, modelDifferences } = applyTrialToAgentConfig(
      specAgent({ provider: 'nvidia', model: 'nemotron' }),
      trial({ temperature: '0.2', max_tokens: '512' }),
      study(
        {
          temperature: { path: 'models.default.temperature' },
          max_tokens: { path: 'models.default.max_tokens' },
        },
        { default: { provider: 'openai', model: 'nvidia/nemotron', base_url: 'http://eval' } }
      )
    );

    expect(next.harnesses).toEqual({
      main: {
        kind: 'hermes',
        model: { provider: 'nvidia', model: 'nemotron', temperature: 0.2, max_tokens: 512 },
      },
    });
    expect(modelDifferences).toEqual([
      {
        modelKey: 'default',
        fields: ['provider', 'model', 'base_url'],
        studyModel: 'nvidia/nemotron',
        agentModel: 'nemotron',
      },
    ]);
  });

  it('compares against the model id the trial deploys', () => {
    const { modelDifferences } = applyTrialToAgentConfig(
      specAgent(),
      trial({ model: 'llama-big' }),
      study(
        { model: { path: 'models.default.model', values: ['llama', 'llama-big'] } },
        { default: { provider: 'nvidia', model: 'llama', temperature: 0.7 } }
      )
    );

    expect(modelDifferences).toEqual([]);
  });

  it('adds models only the optimize config defines, with the trial values on them', () => {
    const { config: next, addedModels } = applyTrialToAgentConfig(
      specAgent(),
      trial({ temperature: '0.2', judge_temperature: '0' }),
      study(
        {
          temperature: { path: 'models.default.temperature' },
          judge_temperature: { path: 'models.judge.temperature', values: [0, 0.5] },
        },
        { judge: { provider: 'nvidia', model: 'judge-model' } }
      )
    );

    expect(addedModels).toEqual(['judge']);
    expect(next.models).toMatchObject({
      default: { temperature: 0.2 },
      judge: { provider: 'nvidia', model: 'judge-model', temperature: 0 },
    });
  });

  it('rejects a model neither the agent nor the optimize config defines', () => {
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

  it.each(['models.__proto__.polluted', 'models.default.constructor.polluted'])(
    'rejects the prototype path %s without writing to Object.prototype',
    (path) => {
      expect(() =>
        applyTrialToAgentConfig(
          { name: 'fabric-agent', models: { default: { model: 'x' } } },
          trial({ evil: '1' }),
          study({ evil: { path } })
        )
      ).toThrow(/Cannot apply/);
      expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    }
  );

  it('does not resolve param names through the prototype', () => {
    expect(() =>
      applyTrialToAgentConfig(specAgent(), trial({ constructor: '1' }), searchSpace)
    ).toThrow(/not in the study's search space/);
  });
});

describe('withAgentName', () => {
  it('renames the config and the name its traces are tagged with', () => {
    expect(
      withAgentName({ name: 'hermes', telemetry: { enabled: true, agent_name: 'hermes' } }, 'h2')
    ).toEqual({ name: 'h2', telemetry: { enabled: true, agent_name: 'h2' } });
  });

  it('leaves out fields the source config does not set', () => {
    expect(withAgentName({ name: 'hermes', telemetry: { enabled: true } }, 'h2')).toEqual({
      name: 'h2',
      telemetry: { enabled: true },
    });
  });
});
