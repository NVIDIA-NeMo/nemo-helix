// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getAgentOptimizationListRunStrategyJobsQueryKey } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import { agentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { launchOptimizeStudy } from '@studio/api/agents/useLaunchOptimizeStudy';
import { evaluationConfigRef } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/evaluationConfigRef';
import type { OptimizationFormOutput } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { optimizationTargets } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import {
  buildStudyBundle,
  gatewayAgentModel,
  OPTIMIZE_CONFIG_PATH,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizeConfig';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { getAgentOptimizationsTabRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useNavigate } from 'react-router';

export interface UseSubmitOptimizationParams {
  workspace: string;
  agentName?: string;
  evals: AgentEvaluationRow[];
}

/**
 * The submit path behind {@link NewOptimizationForm}: stage a generated optimize config, start the
 * study scored by the selected evaluation's stored config, and return to the jobs table.
 *
 * Rejects with a user-facing message on any failure; the form shows it beside the run button.
 */
export const useSubmitOptimization = ({
  workspace,
  agentName,
  evals,
}: UseSubmitOptimizationParams) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useCallback(
    async (values: OptimizationFormOutput): Promise<void> => {
      if (!agentName) throw new Error('No agent selected.');
      const target = optimizationTargets(evals).find(
        (candidate) => candidate.experimentId === values.experimentId
      );
      if (!target) throw new Error('The selected evaluation is no longer available.');

      const [evaluationConfig, agent] = await Promise.all([
        evaluationConfigRef(workspace, target.evaluation),
        agentsGetAgent(workspace, agentName),
      ]);
      const agentModel = gatewayAgentModel(workspace, agent.config);
      const job = await launchOptimizeStudy({
        workspace,
        agentName,
        name: values.name,
        entries: buildStudyBundle({ values, agentModel }),
        optimizeConfig: OPTIMIZE_CONFIG_PATH,
        evaluationConfig,
      });

      toast.success(`Optimization study "${job.name ?? values.name}" submitted`);
      void queryClient.invalidateQueries({
        queryKey: getAgentOptimizationListRunStrategyJobsQueryKey(workspace),
      });
      navigate(getAgentOptimizationsTabRoute(workspace, agentName));
    },
    [workspace, agentName, evals, toast, navigate, queryClient]
  );
};
