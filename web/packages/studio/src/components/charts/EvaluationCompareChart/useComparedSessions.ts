// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getListEvaluationSessionsQueryKey,
  listEvaluationSessions,
} from '@nemo/sdk/generated/platform/experiments';
import type {
  EvaluationSessionResponse,
  ListEvaluationSessionsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useQueries } from '@tanstack/react-query';
import { useCallback } from 'react';

const SESSIONS_PARAMS: ListEvaluationSessionsParams = { page_size: 1000, mode: 'summary' };

export interface ComparedSessions {
  readonly sessionsByEvaluation: ReadonlyMap<string, EvaluationSessionResponse[]>;
  /** Names of evaluations with more sessions than one page holds. */
  readonly truncated: readonly string[];
  readonly isLoading: boolean;
  readonly isError: boolean;
}

/** The sessions endpoint is scoped to one evaluation, so this fans out a query per evaluation. */
export function useComparedSessions(
  workspace: string,
  evaluationNames: readonly string[]
): ComparedSessions {
  const combine = useCallback(
    (
      results: {
        data?: Awaited<ReturnType<typeof listEvaluationSessions>>;
        isLoading: boolean;
        isError: boolean;
      }[]
    ): ComparedSessions => {
      const sessionsByEvaluation = new Map<string, EvaluationSessionResponse[]>();
      const truncated: string[] = [];
      evaluationNames.forEach((name, index) => {
        const page = results[index]?.data;
        const sessions = page?.data ?? [];
        sessionsByEvaluation.set(name, sessions);
        if ((page?.pagination?.total_results ?? 0) > sessions.length) truncated.push(name);
      });
      return {
        sessionsByEvaluation,
        truncated,
        isLoading: results.some((result) => result.isLoading),
        isError: results.some((result) => result.isError),
      };
    },
    [evaluationNames]
  );

  return useQueries({
    queries: evaluationNames.map((name) => ({
      queryKey: getListEvaluationSessionsQueryKey(workspace, name, SESSIONS_PARAMS),
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        listEvaluationSessions(workspace, name, SESSIONS_PARAMS, signal),
    })),
    combine,
  });
}
