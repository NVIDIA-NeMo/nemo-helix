// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Config prefix every swept parameter hangs off. The study overlays a Fabric agent config
 *  (`nemo-agents-spec-v1`), whose sampling knobs live under `models.<role>` — and `default` is the
 *  role the build path scaffolds, so it is the only one we can assume without reading the agent's
 *  config. */
export const LLM_CONFIG_PREFIX = 'models.default';

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
  parameters: SearchParameter[];
}

/**
 * What the user is tuning for, and the search space that follows from it.
 *
 * Picking an intent is the only way the form sets a search space, so the user never has to know
 * which config key moves their objective.
 *
 * Every intent sweeps temperature plus the one other knob its objective leans on. Fabric 0.4.0's
 * `ModelConfig` takes `temperature`, `top_p` and `max_tokens` and forbids anything else, and those
 * three are what the deepagents adapter forwards to the model. Add a parameter here only once the
 * adapter actually reads it.
 */
export const OPTIMIZATION_INTENTS: OptimizationIntent[] = [
  {
    id: 'accuracy',
    title: 'Accuracy',
    description:
      'Pick when a wrong answer costs more than a long one — triage, extraction, anything with a right answer.',
    parameters: [parameter('temperature', 'float', 0, 0.6), parameter('top_p', 'float', 0.7, 1)],
  },
  {
    id: 'brevity',
    title: 'Brevity',
    description:
      'Pick when answers are already correct but rambling, and length is driving your token bill or losing readers.',
    parameters: [
      parameter('temperature', 'float', 0, 0.8),
      parameter('max_tokens', 'int', 128, 768),
    ],
  },
  {
    id: 'creativity',
    title: 'Creativity',
    description:
      'Pick when outputs feel repetitive or templated and you want more variety across similar prompts.',
    parameters: [parameter('temperature', 'float', 0.3, 1.5), parameter('top_p', 'float', 0.8, 1)],
  },
  {
    id: 'cost',
    title: 'Cost & speed',
    description:
      'Pick when quality already clears the bar and you want the cheapest config that still holds the line.',
    parameters: [
      parameter('temperature', 'float', 0, 1),
      parameter('max_tokens', 'int', 256, 1024),
    ],
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
