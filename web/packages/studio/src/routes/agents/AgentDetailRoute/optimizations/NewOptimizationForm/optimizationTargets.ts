// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evaluatorLabel } from '@studio/routes/agents/AgentDetailRoute/evaluations/formatRollups';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';

export interface OptimizationTarget {
  experimentId: string;
  experimentName: string | null;
  evaluation: AgentEvaluationRow;
  /** Evaluator names, preferring the evaluation's own list and falling back to whatever actually
   *  published a score — a run in flight has the former but not yet the latter. */
  evaluators: string[];
}

/** Evaluators an evaluation ran, as the optimize config's `eval.evaluators` list wants them. */
const evaluatorNames = (evaluation: AgentEvaluationRow): string[] => {
  const declared = evaluation.evaluator_names ?? [];
  if (declared.length > 0) return declared;
  return Object.keys(evaluation.aggregate_scores ?? {}).map(evaluatorLabel);
};

/**
 * One target per experiment, carrying that experiment's most recent evaluation.
 */
export const optimizationTargets = (evaluations: AgentEvaluationRow[]): OptimizationTarget[] => {
  const byExperiment = new Map<string, OptimizationTarget>();

  for (const evaluation of evaluations) {
    for (const experiment of evaluation.experiments) {
      const existing = byExperiment.get(experiment.id);
      const isNewer =
        !existing || (evaluation.created_at ?? '') > (existing.evaluation.created_at ?? '');
      if (!isNewer) continue;
      byExperiment.set(experiment.id, {
        experimentId: experiment.id,
        experimentName: experiment.name,
        evaluation,
        evaluators: evaluatorNames(evaluation),
      });
    }
  }

  return [...byExperiment.values()].sort((a, b) =>
    (b.evaluation.created_at ?? '').localeCompare(a.evaluation.created_at ?? '')
  );
};
