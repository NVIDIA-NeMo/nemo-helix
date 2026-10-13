// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { optimizeBundleProblems } from '@studio/api/agents/optimizeBundle';
import type { OptimizationFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { CANDIDATE_MARKER } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/mimicEvaluation';
import {
  buildOptimizeConfig,
  buildStudyBundle,
  gatewayAgentModel,
  OPTIMIZE_CONFIG_PATH,
  STUDY_DATASET_PATH,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizeConfig';
import type { StudyEvaluation } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';
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

const evaluationData: StudyEvaluation = {
  spec: {
    dataset: 'ws/data#rows.jsonl',
    prompt_template: '{{ item.question }}',
    metrics: [
      {
        bundle_kind: 'metric-bundle',
        bundle_format_version: 'v1',
        metric_type: 'llm-judge',
        outputs: [{ name: 'accuracy', value_json_schema: { type: 'number' } }],
        payload: {
          kind: 'inline',
          metric: {
            type: 'llm-judge',
            model: 'ws/original-judge',
            inference: { max_tokens: 1024 },
            scores: [{ name: 'accuracy', minimum: 0, maximum: 1 }],
            prompt_template: 'Compare {{ item.expected_answer }} against {{ sample.output_text }}.',
          },
        },
      },
    ],
  },
  records: [{ question: 'q', expected_answer: 'a' }],
};

describe('buildOptimizeConfig', () => {
  const text = buildOptimizeConfig({ workspace: 'ws', values: values(), evaluationData });
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

  it('uses the original judge and custom scoring supported by the unchanged backend', () => {
    expect(config.models.judge).toEqual({
      provider: 'nvidia',
      model: 'ws/original-judge',
      base_url: '${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
    });
    expect(config.eval.original_evaluation).toBeUndefined();
    expect(config.eval.evaluators.accuracy).toMatchObject({
      _type: 'tunable_rag_evaluator',
      llm_name: 'judge',
      default_scoring: false,
      inference: { max_tokens: 1024 },
    });
    expect(config.eval.evaluators.accuracy.judge_llm_prompt).toContain('"score"');
    expect(config.optimizer.eval_metrics).toEqual({
      average_score: {
        evaluator_name: 'average_score',
        direction: 'maximize',
        weight: 1,
      },
    });
  });

  it('carries the experiment on the optimizer, which survives the merge onto the agent', () => {
    expect(config.optimizer.experiment_id).toBe('exp-1');
    expect(config.metadata).toBeUndefined();
  });

  it('passes the same preflight the bundle upload runs', () => {
    expect(
      optimizeBundleProblems({
        config,
        bundlePaths: new Set([OPTIMIZE_CONFIG_PATH, STUDY_DATASET_PATH]),
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

  it('puts the re-routed model in the overlay as models.default, beside the judge', () => {
    const config = parse(
      buildOptimizeConfig({
        workspace: 'ws',
        values: values(),
        evaluationData,
        agentModel: gatewayAgentModel('ws', providerAgent),
      })
    );
    expect(config.models.default).toMatchObject({
      model: 'agent-model',
      base_url: '${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/ws/openai/-/v1',
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
    expect(config.models.judge.model).toBe('ws/original-judge');
  });

  it('adds no models.default without an agent model', () => {
    expect(
      parse(buildOptimizeConfig({ workspace: 'ws', values: values(), evaluationData })).models
        .default
    ).toBe(undefined);
  });
});

describe('buildStudyBundle', () => {
  it('stages the config and the rows it points at', async () => {
    const entries = buildStudyBundle({ workspace: 'ws', values: values(), evaluationData });

    expect(entries.map((entry) => entry.path)).toEqual([OPTIMIZE_CONFIG_PATH, STUDY_DATASET_PATH]);
    expect(JSON.parse(await entries[1].file.text())).toEqual([
      {
        id: '0',
        question: 'q',
        answer: expect.stringContaining(`Compare a against ${CANDIDATE_MARKER}.`),
      },
    ]);
  });
});
