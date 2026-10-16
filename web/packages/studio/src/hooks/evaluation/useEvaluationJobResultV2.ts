// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvalsListEvaluateJobResults } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import { useQuery } from '@tanstack/react-query';

export interface AggregateScores {
  scores: Record<string, Record<string, number>>;
}

export const useEvaluationJobResultV2 = (workspace: string, jobName: string) => {
  const {
    data: resultsPage,
    isLoading: isLoadingMetadata,
    error: metadataError,
  } = useEvalsListEvaluateJobResults(workspace, jobName, {
    query: {
      enabled: !!workspace && !!jobName,
    },
  });

  const aggregateScoresResult = resultsPage?.data.find((r) => r.name === 'aggregate-scores');

  const {
    data: scores,
    isLoading: isLoadingScores,
    error: scoresError,
  } = useQuery({
    queryKey: [
      'evaluation-job-result-scores',
      workspace,
      jobName,
      aggregateScoresResult?.download_url,
    ],
    queryFn: async () => {
      if (!aggregateScoresResult?.download_url) {
        throw new Error('No download URL available');
      }

      const response = await platformFetch(aggregateScoresResult.download_url);
      if (!response.ok) {
        throw new Error(`Failed to download results: ${response.statusText}`);
      }

      const data: AggregateScores = await response.json();
      return data;
    },
    enabled: !!aggregateScoresResult?.download_url,
  });

  return {
    result: aggregateScoresResult,
    scores,
    isLoading: isLoadingMetadata || isLoadingScores,
    error: metadataError || scoresError,
  };
};
