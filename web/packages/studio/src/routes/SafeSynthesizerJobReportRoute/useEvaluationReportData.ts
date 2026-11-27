// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSafeSynthesizerDownloadJobResultQueryKey,
  safeSynthesizerDownloadJobResult,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import {
  type EvaluationReportData,
  evaluationReportDataSchema,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { useQuery } from '@tanstack/react-query';

export const EVALUATION_REPORT_DATA_RESULT = 'evaluation-report-data';

export const parseEvaluationReportData = (text: string): EvaluationReportData | null => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  const result = evaluationReportDataSchema.safeParse(parsed);
  if (!result.success) {
    return null;
  }
  return result.data;
};

const isNotFound = (error: unknown): boolean => {
  const status = (error as { response?: { status?: unknown } } | null)?.response?.status;
  return status === 404;
};

const fetchEvaluationReportData = async (
  workspace: string,
  job: string,
  signal: AbortSignal
): Promise<EvaluationReportData | null> => {
  try {
    const blob = await safeSynthesizerDownloadJobResult(
      workspace,
      job,
      EVALUATION_REPORT_DATA_RESULT,
      signal
    );
    return parseEvaluationReportData(await blob.text());
  } catch (error) {
    if (isNotFound(error)) {
      return null;
    }
    throw error;
  }
};

/** Resolves to null when the job has no report data or it cannot be read, so callers fall back to the summary. */
export const useEvaluationReportData = (workspace: string, job: string, enabled: boolean) =>
  useQuery({
    queryKey: getSafeSynthesizerDownloadJobResultQueryKey(
      workspace,
      job,
      EVALUATION_REPORT_DATA_RESULT
    ),
    queryFn: ({ signal }) => fetchEvaluationReportData(workspace, job, signal),
    enabled,
    staleTime: Infinity,
  });
