// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvalsGetEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import {
  useAggregatedEvaluationResults,
  useAggregatedEvaluations,
} from '@studio/api/evaluation/details/useAggregatedEvaluations';
import { useParams } from 'react-router';

export const useComparisonResults = () => {
  const { id = '', workspace = '' } = useParams();

  const { data: evaluationData, isLoading, error } = useEvalsGetEvaluateJob(workspace, id);

  // Extract first metric reference from v2 job spec
  const firstMetric = evaluationData?.spec?.metrics?.[0];
  const metricRef =
    firstMetric && typeof firstMetric === 'object' && 'metric_type' in firstMetric
      ? (firstMetric.metric_type as string)
      : undefined;
  const [metricWorkspace, metricName] = metricRef?.split('/') || ['', ''];

  const {
    data: aggData,
    error: aggError,
    isLoading: isAggregationLoading,
  } = useAggregatedEvaluations({
    workspace: metricWorkspace,
    name: metricName,
  });
  const {
    data: aggResults,
    error: aggResultsError,
    isLoading: isResultsLoading,
  } = useAggregatedEvaluationResults({
    workspace: metricWorkspace,
    name: metricName,
  });

  if (error || !evaluationData) {
    return {};
  }

  return {
    data: aggData,
    results: aggResults,
    error: aggError || aggResultsError,
    isLoading: isLoading || isAggregationLoading || isResultsLoading,
  };
};
