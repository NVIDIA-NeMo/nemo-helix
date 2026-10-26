// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetEntry } from '@studio/api/files/types';
import type { OptimizationFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import {
  mimicEvaluation,
  MIMICKED_JUDGE_PROMPT,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/mimicEvaluation';
import type { StudyEvaluation } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';
import {
  budgetById,
  PROMPT_SEARCH,
  type SearchParameter,
  SYSTEM_PROMPT_PATH,
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

/** The agent's system prompt at {@link SYSTEM_PROMPT_PATH}, if it has one there. */
export const agentSystemPrompt = (
  agentConfig: Record<string, unknown> | undefined
): string | undefined => {
  const content = asRecord(asRecord(agentConfig?.instructions)?.system)?.content;
  return typeof content === 'string' ? content : undefined;
};

export interface BuildOptimizeConfigParams {
  workspace: string;
  values: OptimizationFormOutput;
  agentModel?: Record<string, unknown>;
  evaluationData: StudyEvaluation;
}

/** A registered model, called through the gateway of the workspace its URN names. */
const registeredModel = (workspace: string, model: string): Record<string, unknown> => {
  const modelWorkspace = model.includes('/') ? model.slice(0, model.indexOf('/')) : workspace;
  return { provider: 'nvidia', model, base_url: gatewayBaseUrl(modelWorkspace) };
};

const originalJudgeModel = (workspace: string, model: unknown): Record<string, unknown> => {
  if (typeof model === 'string') return registeredModel(workspace, model);
  const inline = asRecord(model);
  if (
    typeof inline?.url !== 'string' ||
    typeof inline.name !== 'string' ||
    inline.api_key_secret ||
    inline.default_headers
  ) {
    throw new Error(
      'Frontend-generated optimization requires a registered judge model or an inline model without custom authentication.'
    );
  }
  return {
    provider: 'openai',
    model: inline.served_model_name ?? inline.name,
    base_url: inline.url,
  };
};

/** The `models.*` role the prompt phase's rewriter is declared under. */
export const PROMPT_OPTIMIZER_MODEL = 'prompt_optimizer';

/** The search-space key for the system prompt, in either mode; the GA names its artifacts after
 *  it, and the sweep its trial parameter. */
export const SYSTEM_PROMPT_DIMENSION = 'system_prompt';

/** The prompt phase runs after the numeric sweep, starting from its best config. Its prompt seed
 *  is read from the agent at {@link SYSTEM_PROMPT_PATH}, so no copy of it travels in the bundle. */
const PROMPT_PHASE = {
  enabled: true,
  backend: 'ga',
  model: PROMPT_OPTIMIZER_MODEL,
  population_size: PROMPT_SEARCH.populationSize,
  generations: PROMPT_SEARCH.generations,
  oracle_feedback_mode: PROMPT_SEARCH.feedback,
};

/** The job merges only optimizer, eval and models onto the agent; metadata would be dropped.
 *  One repetition per row keeps the run count equal to trials × rows. */
export const buildOptimizeConfig = ({
  workspace,
  values,
  agentModel,
  evaluationData,
}: BuildOptimizeConfigParams): string => {
  const mimicked = mimicEvaluation(evaluationData);
  const { enabled, mode, purpose, model, versions } = values.prompt;
  const rewrite = enabled && mode === 'rewrite';
  // The GA rewrites a prompt dimension; the user's versions are a plain choice the numeric phase
  // sweeps like any other, so they share its trials.
  const promptDimension = {
    type: 'fabric',
    path: SYSTEM_PROMPT_PATH,
    ...(rewrite ? { is_prompt: true, purpose } : { values: versions }),
  };
  const config = {
    models: {
      ...(agentModel ? { default: agentModel } : {}),
      judge: originalJudgeModel(workspace, mimicked.model),
      ...(rewrite ? { [PROMPT_OPTIMIZER_MODEL]: registeredModel(workspace, model) } : {}),
    },
    optimizer: {
      experiment_id: values.experimentId,
      numeric: {
        enabled: true,
        n_trials: budgetById(values.budget).trials,
      },
      ...(rewrite ? { prompt: PROMPT_PHASE } : {}),
      reps_per_param_set: 1,
      eval_metrics: {
        average_score: { evaluator_name: 'average_score', direction: 'maximize', weight: float(1) },
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
        ...(enabled ? { [SYSTEM_PROMPT_DIMENSION]: promptDimension } : {}),
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
          default_scoring: false,
          inference: mimicked.inference,
          judge_llm_prompt: MIMICKED_JUDGE_PROMPT,
        },
      },
    },
  };

  return new Document(config).toString();
};

/** The two files a study's bundle fileset holds: the optimize config and the rows it scores. */
export const buildStudyBundle = (params: BuildOptimizeConfigParams): FilesetEntry[] => [
  {
    path: OPTIMIZE_CONFIG_PATH,
    file: new File([buildOptimizeConfig(params)], OPTIMIZE_CONFIG_PATH, {
      type: 'application/yaml',
    }),
  },
  {
    path: STUDY_DATASET_PATH,
    file: new File(
      [JSON.stringify(mimicEvaluation(params.evaluationData).rows, null, 2)],
      STUDY_DATASET_PATH,
      {
        type: 'application/json',
      }
    ),
  },
];
