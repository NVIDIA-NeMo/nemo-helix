// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Scalar, stringify as stringifyYaml } from 'yaml';

/** Config prefix every swept parameter hangs off. The study overlays a Fabric agent config
 *  (`nemo-agents-spec-v1`), whose sampling knobs live under `models.<role>` — and `default` is the
 *  role the build path scaffolds, so it is the only one we can assume without reading the agent's
 *  config. The Advanced section is where a different role gets fixed. */
export const LLM_CONFIG_PREFIX = 'models.default';

/** Objective the study optimizes. The only metric the optimize trial path emits is the judge's
 *  `average_score` — `tunable_rag_evaluator` publishes it under that name whatever the evaluator is
 *  called — so it is what `optimizer.eval_metrics` has to name for the reducer to find a value. */
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
  /** Plain-language restatement of the objective, echoed in the run summary. */
  objective: string;
  parameters: SearchParameter[];
}

/**
 * What the user is tuning for, and the search space that follows from it.
 *
 * Picking an intent is the only way the form sets a search space, so the user never has to know
 * which config key moves their objective. The Advanced section edits the result; it does not
 * replace this choice.
 *
 * Every intent sweeps temperature and nothing else. It is the only sampling knob a Fabric agent
 * exposes — `ModelConfig` forbids unknown keys, and the deepagents adapter forwards temperature
 * alone — so an intent differs from its neighbours in the range it searches, not in the parameters
 * it touches. Add a parameter here only once the adapter actually reads it.
 */
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
  /** Rough wall-clock, stated as a range the user can plan around rather than a promise. */
  estimatedMinutes: number;
}

export const OPTIMIZATION_BUDGETS: OptimizationBudget[] = [
  { id: 'quick', title: 'Quick', trials: 4, estimatedMinutes: 6 },
  { id: 'standard', title: 'Standard', trials: 8, estimatedMinutes: 12 },
  { id: 'thorough', title: 'Thorough', trials: 16, estimatedMinutes: 25 },
];

export const budgetById = (id: BudgetId): OptimizationBudget =>
  OPTIMIZATION_BUDGETS.find((budget) => budget.id === id) ?? OPTIMIZATION_BUDGETS[1];

/** `0.0–1.0` for a float, `128–768` for an int — floats keep a decimal so the range does not read
 *  as an integer one the sampler would round into. */
export const formatRange = (parameter: SearchParameter): string =>
  parameter.type === 'float'
    ? `${parameter.low.toFixed(1)}–${parameter.high.toFixed(1)}`
    : `${parameter.low}–${parameter.high}`;

/** A bound that survives YAML round-tripping as the type it was authored as.
 *
 *  The backend picks `suggest_int` vs `suggest_float` from the parsed Python type of `low`/`high`,
 *  and a plain `0` serializes as `0`, which parses back as an int. Without the forced decimal a
 *  0–1 temperature sweep would be sampled as the two integers 0 and 1. */
const bound = (value: number, type: SearchParameter['type']): number | Scalar => {
  if (type !== 'float') return value;
  const scalar = new Scalar(value);
  scalar.minFractionDigits = 1;
  return scalar;
};

export interface OptimizeConfigInput {
  parameters: SearchParameter[];
  trials: number;
  /** Bundle-relative path of the staged rows; omitted while they are still loading. */
  datasetPath?: string;
  /** Bare name of the virtual model that judges each trial, and the gateway URL it answers on. */
  judgeModel?: string;
  judgeModelUrl?: string;
  experimentId?: string;
}

/** Fabric model role the judge is registered under. Named for the study so it cannot collide with
 *  a role the agent already defines — the overlay's models are merged over the agent's. */
const JUDGE_ROLE = 'optimize_judge';

/** Evaluator name. Only its `average_score` output is read, so what it is called is cosmetic. */
const EVALUATOR_NAME = 'quality';

const JUDGE_PROMPT = `Score whether the generated answer satisfies the expected answer for the
instruction it was given. Judge only what the answer says, never how it is phrased. Return JSON
only.`;

const evalBlock = (datasetPath: string) => ({
  general: { dataset: { file_path: datasetPath }, max_concurrency: 4 },
  // Trajectories are per-trial intermediate evidence the study never reads back, and capturing
  // them multiplies the artifacts a sweep writes by its trial count.
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

/**
 * The Fabric-native optimization YAML the job runs.
 *
 * Written out in full rather than patched into an existing config: the study needs a search space
 * and an eval block, and the agent's own config carries neither. `metadata.experiment_id` is what
 * lands the trials alongside the evaluations they are being compared against.
 *
 * Shape is dictated by the Optuna backend's parser, not by convenience: `search_space` hangs off
 * `optimizer` (not off `numeric`, which only carries the sampler's own settings), every entry names
 * the applicator `type: fabric` plus the `path` it writes, and `eval_metrics` must declare at least
 * one metric or the study has no objective to create.
 *
 * The eval block is self-contained rather than a pointer at the evaluation it came from. The trial
 * path scores through its own `tunable_rag_evaluator` and reads rows from a JSON file, neither of
 * which is the shape Studio stores an evaluation in — so the study borrows that evaluation's
 * prompts and expected answers (staged separately as `datasetPath`) and re-scores them with the
 * judge named here. Trial scores are therefore comparable across trials, but not with the numbers
 * the original evaluation published.
 */
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
