// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvaluatorGetEvalResult } from '@nemo/sdk/generated/evaluator/evaluator-plugin-eval-results-routes';
import { useEvaluatorGetEvaluateJobResult } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { DatasetEvalRow } from '@studio/components/evaluation/Jobs/datasetEval/DatasetEvalRowResultsPanel';
import { useQuery } from '@tanstack/react-query';

/** Row scores have no dedicated endpoint — they live in the run's bundle, so this
 *  one artifact is still fetched by URL. A malformed line is skipped rather than
 *  discarding every other row alongside it. */
const parseRowScores = (text: string): DatasetEvalRow[] =>
  text
    .split('\n')
    .filter((line) => line.trim())
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as DatasetEvalRow];
      } catch {
        return [];
      }
    });

const downloadText = async (url: string, label: string): Promise<string> => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to download ${label}: ${response.statusText}`);
  return response.text();
};

export const useDatasetEvalResults = (workspace: string, jobName: string, status?: string) => {
  const isPending = status === 'pending' || status === 'active';
  const hasFailed =
    status === 'error' ||
    status === 'cancelled' ||
    status === 'canceled' ||
    status === 'failed' ||
    status === 'cancelling';
  const enabled = !!workspace && !!jobName && status === HelixJobStatus.completed;

  const {
    data: evalResult,
    isLoading: isLoadingScores,
    error: scoresError,
  } = useEvaluatorGetEvalResult(workspace, jobName, { query: { enabled, retry: 3 } });

  const {
    data: rowsMetadata,
    isLoading: isLoadingRowsMetadata,
    error: rowsMetadataError,
  } = useEvaluatorGetEvaluateJobResult(workspace, jobName, 'row-scores', {
    query: { enabled, retry: 3 },
  });

  const {
    data: rows,
    isLoading: isLoadingRowsDownload,
    error: rowsError,
  } = useQuery({
    queryKey: ['dataset-eval-row-scores', workspace, jobName, rowsMetadata?.download_url],
    queryFn: () =>
      downloadText(rowsMetadata?.download_url ?? '', 'row scores').then(parseRowScores),
    enabled: !!rowsMetadata?.download_url,
    retry: 3,
  });

  return {
    scores: evalResult?.scores?.scores ?? [],
    rows: rows ?? [],
    isPending,
    hasFailed,
    isLoadingScores,
    isLoadingRows: isLoadingRowsMetadata || isLoadingRowsDownload,
    scoresError,
    rowsError: rowsMetadataError || rowsError,
  };
};
