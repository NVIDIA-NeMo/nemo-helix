// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Scalar, stringify as stringifyYaml } from 'yaml';

export const LLM_CONFIG_PREFIX = 'models.default';
export const STUDY_METRIC = 'average_score';

export interface SearchParameter {
  /** Dotted path into the agent config, e.g. `models.default.temperature`. */
  path: string;
  /** Trailing segment of `path`, which is what the UI labels the parameter with. */
  label: string;
  type: 'int' | 'float';
  low: number;
  high: number;
}

const parameter = (
  label: string,
  type: SearchParameter['type'],
  low: number,
  high: number
): SearchParameter => ({ path: `${LLM_CONFIG_PREFIX}.${label}`, label, type, low, high });

export type IntentId = 'accuracy' | 'brevity' | 'creativity' | 'cost';

export interface OptimizationIntent {
  id: IntentId;
  title: string;
  description: string;
  objective: string;
  parameters: SearchParameter[];
}

/** What the user is tuning for, and the search space that follows from it. */
export const OPTIMIZATION_INTENTS: OptimizationIntent[] = [
  {
    id: 'accuracy',
    title: 'Accuracy',
    description:
      'Pick when a wrong answer costs more than a long one — triage, extraction, anything with a right answer.',
    objective: "Maximize the evaluation's average score, searching the near-deterministic range.",
    parameters: [parameter('temperature', 'float', 0, 0.6)],
  },
  {
    id: 'brevity',
    title: 'Brevity',
    description:
      'Pick when answers are already correct but rambling, and length is driving your token bill or losing readers.',
    objective: "Maximize the evaluation's average score across the low-to-middle range.",
    parameters: [parameter('temperature', 'float', 0, 0.8)],
  },
  {
    id: 'creativity',
    title: 'Creativity',
    description:
      'Pick when outputs feel repetitive or templated and you want more variety across similar prompts.',
    objective: "Maximize the evaluation's average score across the high range, where outputs vary.",
    parameters: [parameter('temperature', 'float', 0.3, 1.5)],
  },
  {
    id: 'cost',
    title: 'Cost & speed',
    description:
      'Pick when quality already clears the bar and you want the cheapest config that still holds the line.',
    objective: "Maximize the evaluation's average score across the full range.",
    parameters: [parameter('temperature', 'float', 0, 1)],
  },
];

export const intentById = (id: IntentId): OptimizationIntent =>
  OPTIMIZATION_INTENTS.find((intent) => intent.id === id) ?? OPTIMIZATION_INTENTS[0];

export type BudgetId = 'quick' | 'standard' | 'thorough';

export interface OptimizationBudget {
  id: BudgetId;
  title: string;
  trials: number;
  estimatedMinutes: number;
}

export const OPTIMIZATION_BUDGETS: OptimizationBudget[] = [
  { id: 'quick', title: 'Quick', trials: 4, estimatedMinutes: 6 },
  { id: 'standard', title: 'Standard', trials: 8, estimatedMinutes: 12 },
  { id: 'thorough', title: 'Thorough', trials: 16, estimatedMinutes: 25 },
];

export const budgetById = (id: BudgetId): OptimizationBudget =>
  OPTIMIZATION_BUDGETS.find((budget) => budget.id === id) ?? OPTIMIZATION_BUDGETS[1];

export const formatRange = (parameter: SearchParameter): string =>
  parameter.type === 'float'
    ? `${parameter.low.toFixed(1)}–${parameter.high.toFixed(1)}`
    : `${parameter.low}–${parameter.high}`;

/** A bound that survives YAML round-tripping as the type it was authored as. */
const bound = (value: number, type: SearchParameter['type']): number | Scalar => {
  if (type !== 'float') return value;
  const scalar = new Scalar(value);
  scalar.minFractionDigits = 1;
  return scalar;
};

export interface OptimizeConfigInput {
  parameters: SearchParameter[];
  trials: number;
  datasetPath?: string;
  judgeModel?: string;
  judgeModelUrl?: string;
  experimentId?: string;
}

/** Fabric model role the judge is registered under. */
const JUDGE_ROLE = 'optimize_judge';

/** Evaluator name. */
const EVALUATOR_NAME = 'quality';

const JUDGE_PROMPT = `Score whether the generated answer satisfies the expected answer for the
instruction it was given. Judge only what the answer says, never how it is phrased. Return JSON
only.`;

const evalBlock = (datasetPath: string) => ({
  general: { dataset: { file_path: datasetPath }, max_concurrency: 4 },
  fabric: { capture_trajectory: false },
  evaluators: {
    [EVALUATOR_NAME]: {
      _type: 'tunable_rag_evaluator',
      llm_name: JUDGE_ROLE,
      default_scoring: true,
      default_score_weights: { coverage: 0.5, correctness: 0.3, relevance: 0.2 },
      judge_llm_prompt: JUDGE_PROMPT,
    },
  },
});

/** The Fabric-native optimization YAML the job runs. */
export const buildOptimizeConfig = ({
  parameters,
  trials,
  datasetPath,
  judgeModel,
  judgeModelUrl,
  experimentId,
}: OptimizeConfigInput): string => {
  const searchSpace = Object.fromEntries(
    parameters.map((parameter) => [
      parameter.label,
      {
        type: 'fabric',
        path: parameter.path,
        low: bound(parameter.low, parameter.type),
        high: bound(parameter.high, parameter.type),
      },
    ])
  );

  return stringifyYaml({
    optimizer: {
      numeric: { enabled: true, n_trials: trials },
      reps_per_param_set: 1,
      eval_metrics: {
        [STUDY_METRIC]: { evaluator_name: STUDY_METRIC, direction: 'maximize', weight: 1.0 },
      },
      search_space: searchSpace,
    },
    ...(judgeModel
      ? {
          models: {
            [JUDGE_ROLE]: {
              provider: 'openai',
              model: judgeModel,
              ...(judgeModelUrl ? { base_url: judgeModelUrl } : {}),
            },
          },
        }
      : {}),
    ...(datasetPath ? { eval: evalBlock(datasetPath) } : {}),
    ...(experimentId ? { metadata: { experiment_id: experimentId } } : {}),
  });
};
