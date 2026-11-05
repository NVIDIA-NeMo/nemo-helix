// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  compareEvaluations,
  getCompareEvaluationsQueryKey,
} from '@nemo/sdk/generated/platform/evaluations';
import type {
  CompareEvaluationsParams,
  EvaluationCompareResponse,
} from '@nemo/sdk/generated/platform/schema';
import { useQueries } from '@tanstack/react-query';
import { useCallback } from 'react';

export interface ComparisonOfCandidate {
  readonly name: string;
  readonly result: EvaluationCompareResponse | undefined;
}

export interface ComparedEvaluations {
  /** One entry per candidate, in the order passed in, whether or not it has loaded. */
  readonly comparisons: readonly ComparisonOfCandidate[];
  readonly isLoading: boolean;
  readonly isError: boolean;
}

/**
 * The compare endpoint pairs one baseline against one candidate, so this fans out a query
 * per candidate. The pairing itself happens server-side; the browser never re-derives it,
 * so the chart, the CLI, and the API share one implementation.
 */
export function useComparedEvaluations(
  workspace: string,
  baseline: string | undefined,
  candidates: readonly string[],
  metric: string | undefined
): ComparedEvaluations {
  const combine = useCallback(
    (
      results: {
        data?: EvaluationCompareResponse;
        isLoading: boolean;
        isError: boolean;
      }[]
    ): ComparedEvaluations => {
      return {
        comparisons: candidates.map((name, index) => ({
          name,
          result: results[index]?.data,
        })),
        isLoading: results.some((result) => result.isLoading),
        isError: results.some((result) => result.isError),
      };
    },
    [candidates]
  );

  return useQueries({
    queries:
      baseline !== undefined && metric !== undefined
        ? candidates.map((candidate) => {
            const params: CompareEvaluationsParams = {
              baseline,
              candidate,
              metric,
            };
            return {
              queryKey: getCompareEvaluationsQueryKey(workspace, params),
              queryFn: ({ signal }: { signal: AbortSignal }) =>
                compareEvaluations(workspace, params, signal),
            };
          })
        : [],
    combine,
  });
}
