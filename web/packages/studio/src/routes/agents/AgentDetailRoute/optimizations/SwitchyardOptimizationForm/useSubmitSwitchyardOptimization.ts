// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  agentOptimizationCreateRunStrategyJob,
  getAgentOptimizationListRunStrategyJobsQueryKey,
} from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { RunStrategySubmitSpec } from '@nemo/sdk/generated/agent-optimization/schema';
import { SWITCHYARD_STRATEGY } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import type { SwitchyardFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import { getAgentOptimizationsTabRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useNavigate } from 'react-router';

/**
 * The `run-strategy` spec for a Switchyard study. The strategy's own schema forbids fields it does
 * not declare, so only what it takes is sent — each threshold only alongside the routing strategy
 * that reads it, and the judge only when the user picked one.
 */
export const buildSwitchyardSpec = (
  workspace: string,
  agentName: string,
  values: SwitchyardFormOutput
): RunStrategySubmitSpec => {
  const strategies = new Set(values.routingStrategies);
  return {
    strategy: SWITCHYARD_STRATEGY,
    agent: agentName,
    workspace,
    models: values.models,
    routing_strategies: values.routingStrategies,
    ...(strategies.has('random_routing') && { strong_probability: values.strong_probability }),
    ...(strategies.has('stage_router') && { confidence_threshold: values.confidence_threshold }),
    ...(strategies.has('llm_classifier') && {
      base_threshold: values.base_threshold,
      ...(values.judgeModel && { judge_model: values.judgeModel }),
    }),
  };
};

export interface UseSubmitSwitchyardOptimizationParams {
  workspace: string;
  agentName?: string;
}

/**
 * The submit path behind {@link SwitchyardOptimizationForm}: start a `switchyard` run-strategy
 * job and return to the jobs table. Nothing is staged first — the strategy reads the agent and
 * models from the platform itself.
 *
 * Rejects with a user-facing message on any failure; the form shows it beside the run button.
 */
export const useSubmitSwitchyardOptimization = ({
  workspace,
  agentName,
}: UseSubmitSwitchyardOptimizationParams) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useCallback(
    async (values: SwitchyardFormOutput): Promise<void> => {
      if (!agentName) throw new Error('No agent selected.');

      const job = await agentOptimizationCreateRunStrategyJob(workspace, {
        name: values.name,
        spec: buildSwitchyardSpec(workspace, agentName, values),
      });

      toast.success(`Optimization "${job.name ?? values.name}" submitted`);
      void queryClient.invalidateQueries({
        queryKey: getAgentOptimizationListRunStrategyJobsQueryKey(workspace),
      });
      navigate(getAgentOptimizationsTabRoute(workspace, agentName));
    },
    [workspace, agentName, toast, navigate, queryClient]
  );
};
