// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { optimizeBundleProblems } from '@studio/api/agents/optimizeBundle';
import type { OptimizationFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import {
  buildOptimizeConfig,
  buildStudyBundle,
  gatewayAgentModel,
  OPTIMIZE_CONFIG_PATH,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizeConfig';
import { intentById } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { parse } from 'yaml';

const values = (overrides: Partial<OptimizationFormOutput> = {}): OptimizationFormOutput => ({
  name: 'react-agent-cost-1005-120000',
  intent: 'cost',
  budget: 'quick',
  experimentId: 'exp-1',
  searchSpace: intentById('cost').parameters,
  ...overrides,
});

describe('buildOptimizeConfig', () => {
  const text = buildOptimizeConfig({ values: values() });
  const config = parse(text);

  it('sweeps each parameter at its config path, with the budget as the trial count', () => {
    expect(config.optimizer.numeric).toEqual({ enabled: true, n_trials: 4 });
    expect(config.optimizer.reps_per_param_set).toBe(1);
    expect(config.optimizer.search_space).toEqual({
      temperature: { type: 'fabric', path: 'models.default.temperature', low: 0, high: 1 },
      max_tokens: { type: 'fabric', path: 'models.default.max_tokens', low: 256, high: 1024 },
      gateway_credential: {
        type: 'fabric',
        path: 'environment.env.NEMO_AGENTS_IGW_API_KEY',
        values: ['not-used'],
      },
    });
  });

  it('keeps float bounds floats, so a 0–1 sweep is not read back as an int range', () => {
    expect(text).toMatch(/low: 0\.0\n\s+high: 1\.0/);
    expect(text).toMatch(/low: 256\n\s+high: 1024/);
  });

  it('maximizes the evaluator score and leaves dataset, judge and evaluator to the job', () => {
    expect(config.optimizer.eval_metrics).toEqual({
      average_score: {
        evaluator_name: 'average_score',
        direction: 'maximize',
        weight: 1,
      },
    });
    expect(config.eval).toBeUndefined();
    expect(config.models).toBeUndefined();
  });

  it('carries the experiment on the optimizer, which survives the merge onto the agent', () => {
    expect(config.optimizer.experiment_id).toBe('exp-1');
    expect(config.metadata).toBeUndefined();
  });

  it('passes the same preflight the bundle upload runs', () => {
    expect(
      optimizeBundleProblems({
        config,
        bundlePaths: new Set([OPTIMIZE_CONFIG_PATH]),
        agent: 'react-agent',
      })
    ).toEqual([]);
  });
});

describe('gatewayAgentModel', () => {
  const providerAgent = {
    models: {
      default: {
        provider: 'nvidia',
        model: 'agent-model',
        api_key_env: 'NVIDIA_API_KEY',
        temperature: 0,
        settings: {},
      },
    },
  };

  it('routes a provider-keyed model through the gateway, keeping its other fields', () => {
    expect(gatewayAgentModel('ws', providerAgent)).toEqual({
      provider: 'nvidia',
      model: 'agent-model',
      temperature: 0,
      settings: {},
      base_url: '${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
  });

  it('leaves a model already on the gateway alone', () => {
    const agent = {
      models: {
        default: {
          model: 'm',
          base_url: 'http://h/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
        },
      },
    };
    expect(gatewayAgentModel('ws', agent)).toBeUndefined();
  });

  it('overrides the default harness model, which outranks models.default', () => {
    const agent = {
      ...providerAgent,
      default_harness: 'main',
      harnesses: {
        main: { kind: 'deepagents', model: { provider: 'nvidia', model: 'harness-model' } },
      },
    };
    expect(gatewayAgentModel('ws', agent)).toMatchObject({
      model: 'harness-model',
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
  });

  it('reads a legacy settings.base_url as the model URL', () => {
    const routed = {
      models: {
        default: { model: 'm', settings: { base_url: 'http://h/apis/inference-gateway/v2/x' } },
      },
    };
    expect(gatewayAgentModel('ws', routed)).toBeUndefined();

    const direct = {
      models: { default: { model: 'm', settings: { base_url: 'https://api', top_k: 5 } } },
    };
    expect(gatewayAgentModel('ws', direct)).toEqual({
      model: 'm',
      settings: { top_k: 5 },
      base_url: '${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
  });

  it('has nothing to override when the agent declares no default model', () => {
    expect(gatewayAgentModel('ws', undefined)).toBeUndefined();
    expect(gatewayAgentModel('ws', { models: {} })).toBeUndefined();
  });

  it('puts the re-routed model in the overlay as models.default', () => {
    const config = parse(
      buildOptimizeConfig({ values: values(), agentModel: gatewayAgentModel('ws', providerAgent) })
    );
    expect(config.models.default).toMatchObject({
      model: 'agent-model',
      base_url: '${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
  });

  it('adds no models without an agent model', () => {
    expect(parse(buildOptimizeConfig({ values: values() })).models).toBe(undefined);
  });
});

describe('buildStudyBundle', () => {
  it('stages only the optimize config; the job generates the rows it scores', () => {
    expect(buildStudyBundle({ values: values() }).map((entry) => entry.path)).toEqual([
      OPTIMIZE_CONFIG_PATH,
    ]);
  });
});
