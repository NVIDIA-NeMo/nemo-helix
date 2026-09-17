// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG, JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListDeploymentsQueryKey,
  useAgentsDeleteDeployment,
  useAgentsListDeployments,
} from '@nemo/sdk/generated/agents/agent-deployments';
import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { getEvaluation, useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import { fetchEvaluatorJobs } from '@studio/api/evaluation/evaluator-jobs';
import { fetchExperimentsByIds } from '@studio/api/evaluation/experiments';
import {
  type EvalJobRow,
  publishedEvaluationName,
  targetNameForEvalJob,
  toEvalJobRow,
} from '@studio/api/evaluation/utils';
import { RECENT_EVAL_LIMIT } from '@studio/routes/agents/AgentDetailRoute/constants';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';

/** Statuses that will not change again, so polling can stop. */
const TERMINAL_JOB_STATUSES = new Set(['completed', 'error', 'cancelled']);

/** One of the experiments an evaluation belongs to: the name its detail route is nested under, the
 *  description the overview cards label it with, and the two display flags the overview groups and
 *  filters on. Name and description are null when the experiment fell outside the fetched page and
 *  could not be resolved, in which case both flags read false. */
export interface EvaluationExperiment {
  id: string;
  name: string | null;
  description: string | null;
  isFavorite: boolean;
  showsEvaluationsOverTime: boolean;
}

/** A published evaluation plus every experiment it belongs to.
 *
 *  Membership is many-to-many — `AddToGroupModal` adds an evaluation to further experiments by
 *  appending to `experiment_ids` — so this is a list, in `experiment_ids` order, with unresolved
 *  ids keeping their place. Consumers that roll evaluations up by experiment must visit every
 *  entry, or a shared evaluation goes missing from all but one of its experiments. */
export type AgentEvaluationRow = EvaluationResponse & {
  experiments: EvaluationExperiment[];
};

/** The experiment an evaluation's detail route is nested under. Any of them addresses the same
 *  evaluation, so this takes the first one whose name resolved; undefined when none did, which
 *  leaves the caller with no linkable route. */
export const primaryExperimentName = (evaluation: AgentEvaluationRow): string | undefined =>
  evaluation.experiments.find((experiment) => experiment.name)?.name ?? undefined;

interface UseAgentPanelParams {
  workspace: string;
  agentName?: string;
  selectedDeploymentName?: string;
}

export const useAgentDetails = ({
  workspace,
  agentName,
  selectedDeploymentName,
}: UseAgentPanelParams) => {
  const queryClient = useQueryClient();
  const toast = useToast();

  const { data: agent, isLoading: isAgentLoading } = useAgentsGetAgent(workspace, agentName ?? '', {
    query: { enabled: !!agentName && !!workspace },
  });

  const { data: deploymentsResponse, isLoading: isDeploymentsLoading } = useAgentsListDeployments(
    workspace,
    undefined,
    {
      query: {
        enabled: !!agentName,
        // Poll quickly while any deployment is mid-transition (pending/starting/deleting)
        // so the panel reflects controller-side progress; fall back to the long interval
        // otherwise to match the agents table.
        refetchInterval: (query) => {
          const deployments = query.state.data?.data ?? [];
          const transitional = deployments.some(
            (d) =>
              d.agent === agentName &&
              (d.status === 'pending' || d.status === 'starting' || d.status === 'deleting')
          );
          return transitional ? JOB_POLLING_INTERVAL_MS : JOB_POLLING_INTERVAL_LONG;
        },
      },
    }
  );

  const deploymentsData = deploymentsResponse?.data;

  const { data: agentEvalsResponse, isPending: isAgentEvalsPending } = useListEvaluations(
    workspace,
    {
      filter: { agent_name: agentName ?? '' },
      page_size: RECENT_EVAL_LIMIT,
      sort: '-created_at',
    },
    {
      query: {
        enabled: !!agentName && !!workspace,
        // The jobs query stops polling once nothing is live, which is when a run's results are
        // still on their way to Intake, so this one has to keep looking on its own.
        refetchInterval: JOB_POLLING_INTERVAL_LONG,
      },
    }
  );

  const { data: agentJobsData } = useQuery({
    queryKey: ['evaluator-jobs', workspace, 'agent-panel', agentName] as const,
    queryFn: ({ signal }) =>
      fetchEvaluatorJobs(workspace, signal, (all) => {
        const matched = all.filter((job) => targetNameForEvalJob(job) === agentName).length;
        return matched >= RECENT_EVAL_LIMIT;
      }),
    enabled: !!agentName && !!workspace,
    refetchInterval: (query) => {
      const rows = (query.state.data ?? []).map(toEvalJobRow);
      const live = rows.some((row) => !TERMINAL_JOB_STATUSES.has(row.status ?? ''));
      return live ? JOB_POLLING_INTERVAL_MS : false;
    },
  });

  // Evaluations a job says it publishes to that the agent-filtered query cannot see: Intake
  // denormalizes `agent_name` from ingested spans, so a run's evaluation stays invisible to that
  // filter even though the record and its experiment exist from the moment it is submitted.
  const pendingEvaluationNames = useMemo(() => {
    const known = new Set((agentEvalsResponse?.data ?? []).map((evaluation) => evaluation.name));
    const names = new Set<string>();
    for (const job of agentJobsData ?? []) {
      if (targetNameForEvalJob(job) !== agentName) continue;
      const name = publishedEvaluationName(job);
      if (name && !known.has(name)) names.add(name);
    }
    return [...names].sort();
  }, [agentEvalsResponse, agentJobsData, agentName]);

  const { data: pendingEvals } = useQuery({
    queryKey: ['agent-pending-evaluations', workspace, pendingEvaluationNames] as const,
    queryFn: async ({ signal }) => {
      // Settled, not all: a job names its evaluation at submit, so the record can 404 in the gap
      // before Intake creates it, and one missing row must not blank the others.
      const settled = await Promise.allSettled(
        pendingEvaluationNames.map((name) => getEvaluation(workspace, name, signal))
      );
      return settled.flatMap((result) => (result.status === 'fulfilled' ? [result.value] : []));
    },
    enabled: !!workspace && pendingEvaluationNames.length > 0,
    refetchInterval: JOB_POLLING_INTERVAL_LONG,
  });

  const evalResponses: EvaluationResponse[] = useMemo(
    () => [...(agentEvalsResponse?.data ?? []), ...(pendingEvals ?? [])],
    [agentEvalsResponse, pendingEvals]
  );

  const deleteDeploymentMutation = useAgentsDeleteDeployment({
    mutation: {
      onSuccess: () => {
        void queryClient.invalidateQueries({
          queryKey: getAgentsListDeploymentsQueryKey(workspace),
        });
      },
      onError: (error) => {
        toast.error(error.message);
      },
    },
  });

  const agentDeployments = useMemo(
    () => (deploymentsData ?? []).filter((d) => d.agent === agentName),
    [deploymentsData, agentName]
  );

  const experimentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const evaluation of evalResponses) {
      for (const id of evaluation.experiment_ids) ids.add(id);
    }
    return [...ids].sort();
  }, [evalResponses]);

  const { data: experimentsById } = useQuery({
    queryKey: ['experiments-by-id', workspace, experimentIds] as const,
    queryFn: ({ signal }) => fetchExperimentsByIds(workspace, experimentIds, signal),
    enabled: !!workspace && experimentIds.length > 0,
  });

  const agentEvals: AgentEvaluationRow[] = useMemo(() => {
    if (!agentName) return [];
    return evalResponses.map((evaluation) => ({
      ...evaluation,
      experiments: evaluation.experiment_ids.map((id) => {
        const experiment = experimentsById?.get(id);
        return {
          id,
          name: experiment?.name ?? null,
          description: experiment?.description ?? null,
          isFavorite: experiment?.is_favorite ?? false,
          showsEvaluationsOverTime: experiment?.show_evaluations_over_time ?? false,
        };
      }),
    }));
  }, [evalResponses, experimentsById, agentName]);

  const agentJobs: EvalJobRow[] = useMemo(() => {
    if (!agentName) return [];
    return (agentJobsData ?? [])
      .filter((job) => targetNameForEvalJob(job) === agentName)
      .slice(0, RECENT_EVAL_LIMIT)
      .map(toEvalJobRow);
  }, [agentJobsData, agentName]);

  const healthyDeployments = useMemo(
    () => agentDeployments.filter((d) => d.status === 'running'),
    [agentDeployments]
  );

  const isDeploying = useMemo(
    () => agentDeployments.some((d) => d.status === 'pending' || d.status === 'starting'),
    [agentDeployments]
  );

  const chatDeployment = useMemo(() => {
    if (selectedDeploymentName) {
      return healthyDeployments.find((d) => d.name === selectedDeploymentName);
    }
    return healthyDeployments[0];
  }, [healthyDeployments, selectedDeploymentName]);

  return {
    isDeploymentsLoading,
    agent,
    isAgentLoading,
    agentDeployments,
    agentEvals,
    isAgentEvalsPending,
    agentJobs,
    healthyDeployments,
    isDeploying,
    chatDeployment,
    deleteDeploymentMutation,
  };
};
