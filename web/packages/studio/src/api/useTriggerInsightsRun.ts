// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getInsightsListAnalysisRunsQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-runs';
import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import {
  type AnalysisRunOptions,
  type InsightsModelOverrides,
  type InsightsTriggerResult,
  triggerInsightsRun,
} from '@studio/api/insightsAnalysis';
import { type UseMutationResult, useMutation, useQueryClient } from '@tanstack/react-query';

export interface TriggerInsightsRunVariables {
  agent: string;
  overrides?: InsightsModelOverrides;
  options?: AnalysisRunOptions;
}

export const useTriggerInsightsRun = (
  workspace: string
): UseMutationResult<InsightsTriggerResult, Error, TriggerInsightsRunVariables> => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agent, overrides, options }: TriggerInsightsRunVariables) =>
      triggerInsightsRun(workspace, agent, overrides, options),
    onSuccess: async ({ status }) => {
      if (status !== 'started') return;
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getInsightsListAnalysisRunsQueryKey(workspace),
        }),
        queryClient.invalidateQueries({ queryKey: getInsightsListInsightsQueryKey(workspace) }),
      ]);
    },
  });
};
