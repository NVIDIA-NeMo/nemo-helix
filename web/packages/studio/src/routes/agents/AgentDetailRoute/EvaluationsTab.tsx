// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SegmentedControl, Stack } from '@nvidia/foundations-react-core';
import type { EvalJobRow } from '@studio/api/evaluation/utils';
import { EvaluationsTable } from '@studio/routes/agents/AgentDetailRoute/evaluations/EvaluationsTable';
import { ExperimentsTable } from '@studio/routes/agents/AgentDetailRoute/evaluations/ExperimentsTable';
import { groupByExperiment } from '@studio/routes/agents/AgentDetailRoute/evaluations/groupByExperiment';
import { VIEW_SEARCH_PARAM } from '@studio/routes/agents/AgentDetailRoute/tabs';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { type FC, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router';

const VIEW_EVALUATIONS = 'evaluations';
const VIEW_EXPERIMENTS = 'experiments';

const VIEW_ITEMS = [
  { value: VIEW_EVALUATIONS, children: 'Evaluations' },
  { value: VIEW_EXPERIMENTS, children: 'Experiments' },
];

interface EvaluationsTabProps {
  workspace: string;
  /** The agent whose evaluations these are, seeded into a re-run started from a row. */
  agentName?: string;
  evals: AgentEvaluationRow[];
  jobs: EvalJobRow[];
}

/** Two readings of the same work — flat evaluations or rolled up by experiment. The flat view takes
 *  both datasets because a run in flight is only a job: it has no published evaluation to show
 *  until it finishes, so `EvaluationsTable` merges the two rather than losing it. */
export const EvaluationsTab: FC<EvaluationsTabProps> = ({ workspace, agentName, evals, jobs }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const view =
    searchParams.get(VIEW_SEARCH_PARAM) === VIEW_EXPERIMENTS ? VIEW_EXPERIMENTS : VIEW_EVALUATIONS;
  const setView = useCallback(
    (next: string) => {
      setSearchParams(
        (previous) => {
          const params = new URLSearchParams(previous);
          if (next === VIEW_EXPERIMENTS) params.set(VIEW_SEARCH_PARAM, next);
          else params.delete(VIEW_SEARCH_PARAM);
          return params;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );
  const experiments = useMemo(() => groupByExperiment(evals), [evals]);

  return (
    <Stack gap="density-lg" className="w-full">
      <SegmentedControl
        className="w-fit"
        aria-label="Evaluation view"
        value={view}
        onValueChange={setView}
        items={VIEW_ITEMS}
      />
      {view === VIEW_EXPERIMENTS && (
        <ExperimentsTable workspace={workspace} experiments={experiments} />
      )}
      {view === VIEW_EVALUATIONS && (
        <EvaluationsTable
          workspace={workspace}
          agentName={agentName}
          evaluations={evals}
          jobs={jobs}
        />
      )}
    </Stack>
  );
};
