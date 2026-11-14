// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvalsListAgentEvalResults } from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-results-routes';
import { useEvalsListEvalResults } from '@nemo/sdk/generated/evals/evals-plugin-eval-results-routes';
import { ResultSort } from '@nemo/sdk/generated/evals/schema';
import type { ScoredResult } from '@studio/api/metrics/metricScores';
import { useMemo } from 'react';

/**
 * How many of the newest dataset and agent results feed the trends. Results can't be filtered by
 * metric server-side, so this bounds the scan; older runs fall off the left of the chart.
 */
export const METRIC_HISTORY_LIMIT = 200;

const PARAMS = { page: 1, page_size: METRIC_HISTORY_LIMIT, sort: ResultSort['-created_at'] };

/** Recent eval results in the workspace, shared by the metrics list and detail pages. */
export const useMetricScoreHistory = (workspace: string) => {
  const evalResults = useEvalsListEvalResults(workspace, PARAMS);
  const agentResults = useEvalsListAgentEvalResults(workspace, PARAMS);

  const results = useMemo<ScoredResult[]>(
    () => [
      ...(evalResults.data?.data ?? []).map((result) => ({ ...result, kind: 'dataset' as const })),
      ...(agentResults.data?.data ?? []).map((result) => ({ ...result, kind: 'agent' as const })),
    ],
    [evalResults.data, agentResults.data]
  );

  return {
    results,
    isPending: evalResults.isPending || agentResults.isPending,
    error: evalResults.error ?? agentResults.error,
  };
};
