// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { getAgentsListAgentsQueryKey } from '@nemo/sdk/generated/agents/agents';
import {
  cancelDistillationWorkflow,
  createDistillationWorkflow,
  getDistillationWorkflow,
  renderDistillationWorkflow,
  retryDistillationNode,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApi';
import type {
  CreateWorkflowBody,
  DistillationWorkflow,
  DistillInputs,
  WorkflowStatus,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import {
  type ArmMetrics,
  evaluationNamesByArm,
  fetchArmMetrics,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationArmMetrics';
import { promoteDistillation } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationPromote';
import type { ArmId } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

const TERMINAL_STATUSES: ReadonlySet<WorkflowStatus> = new Set(['completed', 'failed', 'cancelled']);

const workflowKey = (workspace: string, name: string) =>
  ['distillation-workflow', workspace, name] as const;

export const useDistillationWorkflow = (workspace: string, name: string) =>
  useQuery({
    queryKey: workflowKey(workspace, name),
    queryFn: () => getDistillationWorkflow(workspace, name),
    refetchInterval: (query) =>
      query.state.data && TERMINAL_STATUSES.has(query.state.data.status)
        ? false
        : JOB_POLLING_INTERVAL_MS,
  });

export const useRenderDistillation = (workspace: string, inputs: DistillInputs | undefined) =>
  useQuery({
    queryKey: ['distillation-render', workspace, inputs],
    queryFn: () => renderDistillationWorkflow(workspace, inputs as DistillInputs),
    enabled: !!inputs,
    placeholderData: keepPreviousData,
  });

export const useCreateDistillation = (workspace: string) =>
  useMutation({
    mutationFn: (body: CreateWorkflowBody) => createDistillationWorkflow(workspace, body),
  });

export const useCancelDistillation = (workspace: string, name: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => cancelDistillationWorkflow(workspace, name),
    onSuccess: (workflow) => queryClient.setQueryData(workflowKey(workspace, name), workflow),
  });
};

export const useRetryDistillationNode = (workspace: string, name: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (node: string) => retryDistillationNode(workspace, name, node),
    onSuccess: (workflow) => queryClient.setQueryData(workflowKey(workspace, name), workflow),
  });
};

export const usePromoteDistillation = (workspace: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (workflow: DistillationWorkflow) => promoteDistillation(workflow),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) }),
  });
};

/** Intake figures per arm, read once the workflow is done and used where the workflow reports none. */
export const useDistillationArmMetrics = (workspace: string, workflow: DistillationWorkflow | undefined) =>
  useQuery({
    queryKey: ['distillation-arm-metrics', workspace, workflow?.name],
    enabled: workflow?.status === 'completed',
    queryFn: async () => {
      if (!workflow) return {};
      const names = evaluationNamesByArm(workflow);
      const found = await Promise.all(
        Object.entries(names).map(async ([arm, name]) =>
          name ? ([arm, await fetchArmMetrics(workspace, name, workflow.created_at)] as const) : null
        )
      );
      return Object.fromEntries(found.filter((entry) => entry !== null)) as Partial<
        Record<ArmId, ArmMetrics>
      >;
    },
  });
