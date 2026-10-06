// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetEntry } from '@studio/api/files/uploadFilesetEntries';
import type { OptimizationFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import type { StudyRow } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';
import {
  budgetById,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { asRecord } from '@studio/util/guards';
import { Document, Scalar } from 'yaml';

export const OPTIMIZE_CONFIG_PATH = 'optimize.yaml';
export const STUDY_DATASET_PATH = 'dataset.json';

/** The gateway holds the provider key; Fabric still requires a credential env value at startup. */
const IGW_API_KEY_ENV = 'NEMO_AGENTS_IGW_API_KEY';
const GATEWAY_PATH_MARKER = '/apis/inference-gateway/';

/** YAML bounds determine the range type; a decimal keeps a 0–1 sweep continuous. */
const float = (value: number): Scalar<number> => {
  const scalar = new Scalar(value);
  scalar.minFractionDigits = 1;
  return scalar;
};

const bound = (value: number, type: SearchParameter['type']): Scalar<number> | number =>
  type === 'float' ? float(value) : Math.round(value);

const gatewayBaseUrl = (workspace: string): string =>
  `\${NHX_BASE_URL}${GATEWAY_PATH_MARKER}v2/workspaces/${workspace}/openai/-/v1`;

/** The gateway never routes across workspaces, so a `workspace/name` ref is served by its own. */
const modelWorkspace = (modelRef: string, fallback: string): string =>
  modelRef.includes('/') ? modelRef.slice(0, modelRef.indexOf('/')) : fallback;

/** The model the agent runs, as the Fabric translation picks it: the default harness's own model
 *  outranks `models.default`, and is what the trial payload carries as `models.default`. */
const selectedAgentModel = (
  agentConfig: Record<string, unknown> | undefined
): Record<string, unknown> | undefined => {
  const harnesses = asRecord(agentConfig?.harnesses);
  const defaultHarness = agentConfig?.default_harness;
  const harness =
    typeof defaultHarness === 'string' ? asRecord(harnesses?.[defaultHarness]) : undefined;
  return asRecord(harness?.model) ?? asRecord(asRecord(agentConfig?.models)?.default);
};

/** Route provider models through gateway credentials. The overlay replaces the model wholesale,
 *  so retain its sampling and settings. Already-routed or absent models need no override. A
 *  legacy `settings.base_url` counts as the model's URL, as it does for the translation. */
export const gatewayAgentModel = (
  workspace: string,
  agentConfig: Record<string, unknown> | undefined
): Record<string, unknown> | undefined => {
  const model = selectedAgentModel(agentConfig);
  if (!model) return undefined;
  const { base_url: legacyBaseUrl, ...settings } = asRecord(model.settings) ?? {};
  const baseUrl = typeof model.base_url === 'string' ? model.base_url : legacyBaseUrl;
  if (typeof baseUrl === 'string' && baseUrl.includes(GATEWAY_PATH_MARKER)) return undefined;
  return {
    ...model,
    ...(model.settings === undefined ? {} : { settings }),
    base_url: gatewayBaseUrl(workspace),
    api_key_env: IGW_API_KEY_ENV,
  };
};

export interface BuildOptimizeConfigParams {
  workspace: string;
  values: OptimizationFormOutput;
  agentModel?: Record<string, unknown>;
}

/** The job merges only optimizer, eval and models onto the agent; metadata would be dropped.
 *  One repetition per row keeps the run count equal to trials × rows. */
export const buildOptimizeConfig = ({
  workspace,
  values,
  agentModel,
}: BuildOptimizeConfigParams): string => {
  const config = {
    models: {
      ...(agentModel ? { default: agentModel } : {}),
      judge: {
        provider: 'nvidia',
        model: values.judgeModel,
        base_url: gatewayBaseUrl(modelWorkspace(values.judgeModel, workspace)),
      },
    },
    optimizer: {
      experiment_id: values.experimentId,
      numeric: {
        enabled: true,
        n_trials: budgetById(values.budget).trials,
      },
      reps_per_param_set: 1,
      eval_metrics: {
        average_score: {
          evaluator_name: 'average_score',
          direction: 'maximize',
          weight: float(1),
        },
      },
      search_space: {
        ...Object.fromEntries(
          values.searchSpace.map(({ label, path, low, high, type }) => [
            label,
            { type: 'fabric', path, low: bound(low, type), high: bound(high, type) },
          ])
        ),
        // Overlays ignore environment.env; one fixed choice supplies Fabric's placeholder
        // credential without changing the trial budget or grid size.
        gateway_credential: {
          type: 'fabric',
          path: `environment.env.${IGW_API_KEY_ENV}`,
          values: ['not-used'],
        },
      },
    },
    eval: {
      general: {
        dataset: { file_path: STUDY_DATASET_PATH },
      },
      evaluators: {
        accuracy: {
          _type: 'tunable_rag_evaluator',
          llm_name: 'judge',
          inference: { temperature: float(0) },
          default_scoring: true,
          judge_llm_prompt:
            'Score whether the generated answer correctly addresses the question compared to the ' +
            'expected answer. Respond with exactly this JSON object and no other keys: ' +
            '{"coverage_score": <number from 0.0 to 1.0>, "correctness_score": <number from 0.0 ' +
            'to 1.0>, "relevance_score": <number from 0.0 to 1.0>, "reasoning": "<one or two ' +
            'sentences>"}',
        },
      },
    },
  };

  return new Document(config).toString();
};

/** The two files a study's bundle fileset holds: the optimize config and the rows it scores. */
export const buildStudyBundle = (
  params: BuildOptimizeConfigParams & { rows: StudyRow[] }
): FilesetEntry[] => [
  {
    path: OPTIMIZE_CONFIG_PATH,
    file: new File([buildOptimizeConfig(params)], OPTIMIZE_CONFIG_PATH, {
      type: 'application/yaml',
    }),
  },
  {
    path: STUDY_DATASET_PATH,
    file: new File([JSON.stringify(params.rows, null, 2)], STUDY_DATASET_PATH, {
      type: 'application/json',
    }),
  },
];
