// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import type { OptimizationTarget } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import { loadStudyRows } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';
import { queryOptions, skipToken, useQuery } from '@tanstack/react-query';

/** The rows a study stages for one evaluation. Shared by the summary's count and the submit path,
 *  so a submit soon after picking the evaluation reuses the download instead of repeating it. */
export const studyRowsQueryOptions = (workspace: string, evaluation?: EvaluationResponse) =>
  queryOptions({
    queryKey: ['studyRows', workspace, evaluation?.id],
    queryFn: evaluation ? () => loadStudyRows(workspace, evaluation) : skipToken,
    retry: false,
    staleTime: 60_000,
  });

export interface StudyRowCount {
  /** How many rows each trial will score, or undefined while that is unknown. */
  rowCount?: number;
  /** Why the rows could not be loaded; submitting would fail with the same message. */
  error?: string;
}

/**
 * Counts the rows a submitted study actually stages (read from the evaluation's stored eval
 * config), because Intake's `test_case_count` is 0 for an evaluation whose traces have not been
 * ingested. Intake's count is only a fallback for when the config cannot be read.
 */
export const useStudyRowCount = (
  workspace: string,
  target: OptimizationTarget | undefined
): StudyRowCount => {
  const evaluation = target?.evaluation;
  const { data, error } = useQuery({
    ...studyRowsQueryOptions(workspace, evaluation),
    select: (rows) => rows.length,
  });

  return {
    rowCount: data ?? (evaluation?.test_case_count || undefined),
    error: error ? getErrorMessage(error, 'Could not load the evaluation rows.') : undefined,
  };
};
