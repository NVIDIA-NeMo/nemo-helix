// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildOptimizeConfig,
  formatRange,
  intentById,
  OPTIMIZATION_INTENTS,
  STUDY_METRIC,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfig';
import { parse as parseYaml } from 'yaml';

const parameters = intentById('accuracy').parameters;

describe('buildOptimizeConfig', () => {
  it('writes the search space under optimizer, keyed by the logical parameter name', () => {
    const config = parseYaml(buildOptimizeConfig({ parameters, trials: 8 }));

    expect(config.optimizer.numeric).toMatchObject({ enabled: true, n_trials: 8 });
    // Not under `numeric` — the backend reads `optimizer.search_space` and fails without it.
    expect(config.optimizer.numeric).not.toHaveProperty('search_space');
    expect(config.optimizer.search_space).toEqual({
      temperature: { type: 'fabric', path: 'models.default.temperature', low: 0, high: 0.6 },
    });
  });

  it('declares an objective, without which the study cannot be created', () => {
    const config = parseYaml(buildOptimizeConfig({ parameters, trials: 8 }));

    expect(config.optimizer.eval_metrics).toEqual({
      [STUDY_METRIC]: { evaluator_name: STUDY_METRIC, direction: 'maximize', weight: 1 },
    });
  });

  it('keeps float bounds float in the YAML so they are not sampled as integers', () => {
    const yaml = buildOptimizeConfig({
      parameters: [{ path: 'models.default.temperature', label: 'temperature', type: 'float', low: 0, high: 1 }],
      trials: 8,
    });

    expect(yaml).toContain('low: 0.0');
    expect(yaml).toContain('high: 1.0');
  });

  it('leaves int bounds bare so they are sampled as integers', () => {
    const yaml = buildOptimizeConfig({
      parameters: [{ path: 'models.default.n', label: 'n', type: 'int', low: 128, high: 768 }],
      trials: 8,
    });

    expect(yaml).toContain('low: 128\n');
    expect(yaml).toContain('high: 768\n');
  });

  it('builds an eval block the trial path can execute', () => {
    const config = parseYaml(
      buildOptimizeConfig({
        parameters,
        trials: 4,
        datasetPath: 'dataset.json',
        judgeModel: 'judge-model',
        judgeModelUrl: 'https://nmp.example.com/apis/inference-gateway/v2/x/model/judge-model/-/v1',
        experimentId: 'exp-1',
      })
    );

    expect(config.eval.general.dataset).toEqual({ file_path: 'dataset.json' });
    // A mapping of typed evaluators, not a list of names: the trial path rejects anything else.
    expect(config.eval.evaluators.quality).toMatchObject({
      _type: 'tunable_rag_evaluator',
      llm_name: 'optimize_judge',
    });
    // The judge role has to resolve under the payload's models, which the overlay contributes.
    expect(config.models.optimize_judge).toMatchObject({
      provider: 'openai',
      model: 'judge-model',
    });
    expect(config.metadata).toEqual({ experiment_id: 'exp-1' });
  });

  it('omits the eval block until rows are staged, rather than pointing at nothing', () => {
    const config = parseYaml(buildOptimizeConfig({ parameters, trials: 4 }));

    expect(config).not.toHaveProperty('eval');
    expect(config).not.toHaveProperty('models');
    expect(config).not.toHaveProperty('metadata');
  });
});

describe('formatRange', () => {
  it('keeps a decimal on floats so the range does not read as an integer one', () => {
    expect(formatRange({ path: 'p', label: 'temperature', type: 'float', low: 0, high: 1 })).toBe(
      '0.0–1.0'
    );
    expect(formatRange({ path: 'p', label: 'max_tokens', type: 'int', low: 128, high: 768 })).toBe(
      '128–768'
    );
  });
});

describe('OPTIMIZATION_INTENTS', () => {
  it('gives every intent a non-empty search space', () => {
    for (const intent of OPTIMIZATION_INTENTS) {
      expect(intent.parameters.length).toBeGreaterThan(0);
      for (const parameter of intent.parameters) {
        expect(parameter.high).toBeGreaterThan(parameter.low);
        expect(parameter.path.endsWith(parameter.label)).toBe(true);
      }
    }
  });

  it('only sweeps parameters a Fabric agent actually reads', () => {
    // `ModelConfig` forbids unknown keys and the deepagents adapter forwards temperature alone, so
    // anything else here fails the run or is silently ignored.
    for (const intent of OPTIMIZATION_INTENTS) {
      for (const parameter of intent.parameters) {
        expect(parameter.path).toBe('models.default.temperature');
      }
    }
  });
});
