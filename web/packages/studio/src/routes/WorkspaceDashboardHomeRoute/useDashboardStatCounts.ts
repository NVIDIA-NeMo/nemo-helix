// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { useEvaluatorListEvaluateJobs } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { useInsightsListInsights } from '@nemo/sdk/generated/insights/insights-insights';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import type { ModelsListModelsParams } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_CUSTOM_MODELS_FILTER } from '@studio/components/dataViews/CustomModelsDataView/constants';

// Only the total count is needed for a StatTile, so every list is fetched at page_size 1.
const MINIMAL_PAGE = { page: 1, page_size: 1 } as const;

// Matches CustomModelsDataView's default filter — models without this scope
// down to `base_model`/adapters would otherwise count every base model too.
const CUSTOM_MODELS_FILTER = JSON.stringify(
  DEFAULT_CUSTOM_MODELS_FILTER
) as unknown as ModelsListModelsParams['filter'];

export interface DashboardStatCount {
  readonly value: number | undefined;
  readonly isLoading: boolean;
  /** A failed fetch, distinct from a genuine zero — callers must not treat this as "0". */
  readonly isError: boolean;
}

export interface DashboardStatCounts {
  readonly agents: DashboardStatCount;
  readonly insights: DashboardStatCount;
  readonly testCases: DashboardStatCount;
  readonly experiments: DashboardStatCount;
  readonly customModels: DashboardStatCount;
}

/** Which counts to actually fetch — mirrors the StatTiles a caller intends to render. */
export interface DashboardStatCountsEnabled {
  readonly agents: boolean;
  readonly insights: boolean;
  readonly testCases: boolean;
  readonly experiments: boolean;
  readonly customModels: boolean;
}

const toStatCount = (query: {
  data?: { pagination?: { total_results?: number } };
  isLoading: boolean;
  isError: boolean;
}): DashboardStatCount => ({
  value: query.data?.pagination?.total_results,
  isLoading: query.isLoading,
  isError: query.isError,
});

export const useDashboardStatCounts = (
  workspace: string,
  enabled: DashboardStatCountsEnabled
): DashboardStatCounts => {
  const agents = useAgentsListAgents(workspace, MINIMAL_PAGE, {
    query: { enabled: enabled.agents },
  });
  const insights = useInsightsListInsights(workspace, MINIMAL_PAGE, {
    query: { enabled: enabled.insights },
  });
  const testCases = useEvaluatorListEvaluateJobs(workspace, MINIMAL_PAGE, {
    query: { enabled: enabled.testCases },
  });
  const experiments = useListExperiments(workspace, MINIMAL_PAGE, {
    query: { enabled: enabled.experiments },
  });
  const customModels = useModelsListModels(
    workspace,
    { ...MINIMAL_PAGE, filter: CUSTOM_MODELS_FILTER },
    { query: { enabled: enabled.customModels } }
  );

  return {
    agents: toStatCount(agents),
    insights: toStatCount(insights),
    testCases: toStatCount(testCases),
    experiments: toStatCount(experiments),
    customModels: toStatCount(customModels),
  };
};
