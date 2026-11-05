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
import { AGENT_EVALUATIONS_PAGE_SIZE } from '@studio/routes/agents/AgentDetailRoute/constants';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
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
  /** Fetch and poll evaluations; off for tabs that don't show them. */
  evalsEnabled?: boolean;
}

export const useAgentDetails = ({
  workspace,
  agentName,
  selectedDeploymentName,
  evalsEnabled = true,
}: UseAgentPanelParams) => {
  const queryClient = useQueryClient();
  const toast = useToast();

  const {
    data: agent,
    isLoading: isAgentLoading,
    isPending: isAgentPending,
  } = useAgentsGetAgent(workspace, agentName ?? '', {
    query: { enabled: !!agentName && !!workspace },
  });

  const {
    data: deploymentsResponse,
    isLoading: isDeploymentsLoading,
    refetch: refetchDeployments,
  } = useAgentsListDeployments(workspace, undefined, {
    query: {
      enabled: !!agentName,
      // Poll quickly while any deployment is mid-transition (pending/starting/deleting)
      // so the panel reflects controller-side progress; otherwise use the long interval
      // to match the agents table.
      refetchInterval: (query) => {
        const deployments = query.state.data?.data ?? [];
        const transitional = deployments.some(
          (d) =>
            d.agent === agentName &&
            (d.status === 'pending' || d.status === 'starting' || d.status === 'deleting')
        );
        if (transitional) {
          return JOB_POLLING_INTERVAL_MS;
        }
        return JOB_POLLING_INTERVAL_LONG;
      },
    },
  });

  const deploymentsData = deploymentsResponse?.data;

  const { data: agentEvalsResponse, isPending: isAgentEvalsPending } = useListEvaluations(
    workspace,
    {
      filter: { agent_name: agentName ?? '' },
      page_size: AGENT_EVALUATIONS_PAGE_SIZE,
      sort: '-created_at',
    },
    {
      query: {
        enabled: evalsEnabled && !!agentName && !!workspace,
        // The jobs query stops polling once nothing is live, which is when a run's results are
        // still on their way to Intake, so this one has to keep looking on its own.
        refetchInterval: JOB_POLLING_INTERVAL_LONG,
      },
    }
  );

  const { data: agentJobsData } = useQuery({
    queryKey: ['evaluator-jobs', workspace, 'agent-panel', agentName] as const,
    // Fetch every job page: a preview-sized prefix cannot resolve older Evaluation rows.
    queryFn: ({ signal }) => fetchEvaluatorJobs(workspace, signal),
    enabled: evalsEnabled && !!agentName && !!workspace,
    refetchInterval: (query) => {
      const rows = (query.state.data ?? [])
        .filter((job) => targetNameForEvalJob(job) === agentName)
        .map(toEvalJobRow);
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
    // Jobs are newest first. Hydrate only a page of this agent's recent publications,
    // while keeping the full job history above for linkage of older displayed evaluations.
    const recentJobs = (agentJobsData ?? [])
      .filter((job) => targetNameForEvalJob(job) === agentName)
      .slice(0, AGENT_EVALUATIONS_PAGE_SIZE);
    for (const job of recentJobs) {
      const name = publishedEvaluationName(job);
      if (name && !known.has(name)) names.add(name);
    }
    return [...names].sort();
  }, [agentEvalsResponse, agentJobsData, agentName]);

  const pendingEvals = useQueries({
    queries: pendingEvaluationNames.map((name) => ({
      queryKey: ['agent-pending-evaluation', workspace, name] as const,
      queryFn: ({ signal }: { signal: AbortSignal }) => getEvaluation(workspace, name, signal),
      enabled: evalsEnabled && !!workspace,
      // A missing publication must not blank other rows. Poll only unresolved records:
      // successful lookups stay cached even when the candidate page changes.
      retry: false,
      staleTime: Infinity,
      refetchInterval: (query: { state: { data: EvaluationResponse | undefined } }) =>
        query.state.data ? false : JOB_POLLING_INTERVAL_LONG,
    })),
    combine: (results) => results.flatMap((result) => (result.data ? [result.data] : [])),
  });

  const evalResponses: EvaluationResponse[] = useMemo(() => {
    const evaluations = new Map(
      (agentEvalsResponse?.data ?? []).map((evaluation) => [evaluation.name, evaluation])
    );
    // Prefer the regularly refreshed list once Intake has indexed the agent name.
    for (const evaluation of pendingEvals) {
      if (!evaluations.has(evaluation.name)) evaluations.set(evaluation.name, evaluation);
    }
    return [...evaluations.values()]
      .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
      .slice(0, AGENT_EVALUATIONS_PAGE_SIZE);
  }, [agentEvalsResponse, pendingEvals]);

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
    () =>
      (deploymentsData ?? [])
        .filter((d) => d.agent === agentName)
        .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? '')),
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
    enabled: evalsEnabled && !!workspace && experimentIds.length > 0,
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

  // Any status is selectable (a failed deployment still has logs); without a pick, prefer a healthy one.
  const selectedDeployment = useMemo(() => {
    const named = agentDeployments.filter((d) => d.name);
    return (
      named.find((d) => d.name === selectedDeploymentName) ??
      named.find((d) => d.status === 'running') ??
      named[0]
    );
  }, [agentDeployments, selectedDeploymentName]);

  return {
    isDeploymentsLoading,
    refetchDeployments,
    agent,
    isAgentLoading,
    isAgentPending,
    agentDeployments,
    agentEvals,
    isAgentEvalsPending,
    agentJobs,
    healthyDeployments,
    isDeploying,
    selectedDeployment,
    deleteDeploymentMutation,
  };
};
