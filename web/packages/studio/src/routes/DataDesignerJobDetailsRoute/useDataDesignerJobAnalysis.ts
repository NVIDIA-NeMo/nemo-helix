// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  dataDesignerDownloadCreateJobResult,
  getDataDesignerDownloadCreateJobResultQueryKey,
  useDataDesignerListCreateJobResults,
} from '@nemo/sdk/generated/data-designer/data-designer';
import type { DatasetProfilerResults } from '@studio/routes/DataDesignerJobDetailsRoute/datasetProfilerTypes';
import { useQuery } from '@tanstack/react-query';

/** Result name under which the profiler analysis JSON is registered for a job. */
const ANALYSIS_RESULT_NAME = 'analysis';

interface UseDataDesignerJobAnalysisOptions {
  /** Gate the download until the job has reached a terminal status. */
  enabled?: boolean;
}

/**
 * Loads and parses the {@link DatasetProfilerResults} for a Data Designer job.
 *
 * The profiler output is exposed as a downloadable JSON result named
 * `analysis`. We first list the job's results to confirm the analysis exists
 * (a failed job never produces one) and only then download + parse it, so a
 * missing profile surfaces as `hasAnalysis: false` rather than a 404.
 */
export const useDataDesignerJobAnalysis = (
  workspace: string,
  jobName: string,
  { enabled = true }: UseDataDesignerJobAnalysisOptions = {}
) => {
  const {
    data: resultsResponse,
    isLoading: isResultsLoading,
    isError: isResultsError,
    error: resultsError,
  } = useDataDesignerListCreateJobResults(workspace, jobName, {
    query: { enabled: enabled && Boolean(workspace && jobName) },
  });

  const hasAnalysis = Boolean(
    resultsResponse?.data?.some((result) => result.name === ANALYSIS_RESULT_NAME)
  );

  const analysisQuery = useQuery({
    queryKey: getDataDesignerDownloadCreateJobResultQueryKey(
      workspace,
      jobName,
      ANALYSIS_RESULT_NAME
    ),
    queryFn: async ({ signal }): Promise<DatasetProfilerResults> => {
      const blob = await dataDesignerDownloadCreateJobResult(
        workspace,
        jobName,
        ANALYSIS_RESULT_NAME,
        signal
      );
      const text = await blob.text();
      return JSON.parse(text) as DatasetProfilerResults;
    },
    enabled: enabled && hasAnalysis,
  });

  return {
    analysis: analysisQuery.data,
    hasAnalysis,
    isLoading: isResultsLoading || (hasAnalysis && analysisQuery.isLoading),
    isError: isResultsError || analysisQuery.isError,
    error: resultsError ?? analysisQuery.error,
  };
};
