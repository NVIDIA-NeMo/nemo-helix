// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useEvalsGetEvalResult } from '@nemo/sdk/generated/evals/evals-plugin-eval-results-routes';
import { useEvalsGetEvaluateJobResult } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
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
  const hasFailed =
    status === 'error' ||
    status === 'cancelled' ||
    status === 'canceled' ||
    status === 'failed' ||
    status === 'cancelling';
  // Any non-terminal status (`created`, `paused`, ...) is still pending; `cancelling` is not
  // terminal yet but counts as failed above because it will never produce results.
  const isPending =
    !!status &&
    !hasFailed &&
    !HelixJobTerminalStatuses.some((terminalStatus) => terminalStatus === status);
  const enabled = !!workspace && !!jobName && status === HelixJobStatus.completed;

  const {
    data: evalResult,
    isPending: isScoresPending,
    error: scoresError,
  } = useEvalsGetEvalResult(workspace, jobName, { query: { enabled, retry: 3 } });

  const {
    data: rowsMetadata,
    isPending: isRowsMetadataPending,
    error: rowsMetadataError,
  } = useEvalsGetEvaluateJobResult(workspace, jobName, 'row-scores', {
    query: { enabled, retry: 3 },
  });

  const hasRowsDownloadUrl = !!rowsMetadata?.download_url;
  const {
    data: rows,
    isPending: isRowsDownloadPending,
    error: rowsError,
  } = useQuery({
    queryKey: ['dataset-eval-row-scores', workspace, jobName, rowsMetadata?.download_url],
    queryFn: () =>
      downloadText(rowsMetadata?.download_url ?? '', 'row scores').then(parseRowScores),
    enabled: hasRowsDownloadUrl,
    retry: 3,
  });

  // `isPending` rather than `isLoading`: a query that hasn't started yet is still loading
  // from the page's point of view, but `isLoading` is false until its first fetch begins.
  return {
    scores: evalResult?.scores?.scores ?? [],
    rows: rows ?? [],
    isPending,
    hasFailed,
    isLoadingScores: enabled && isScoresPending,
    isLoadingRows:
      enabled && (isRowsMetadataPending || (hasRowsDownloadUrl && isRowsDownloadPending)),
    scoresError,
    rowsError: rowsMetadataError || rowsError,
  };
};
