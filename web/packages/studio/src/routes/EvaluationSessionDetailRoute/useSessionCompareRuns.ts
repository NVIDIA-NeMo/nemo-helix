// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import { useGetExperiment } from '@nemo/sdk/generated/platform/experiments';
import { useSessionTrajectories } from '@studio/components/IntakeDetail/useSessionTrajectories';
import { useTestCaseRuns } from '@studio/routes/EvaluationSessionDetailRoute/useTestCaseRuns';
import { useMemo } from 'react';

/**
 * Every run of the current session's test case across the group's evaluations, plus
 * the test case name the session is keyed on.
 *
 * Shared by the compare view and the single session view's "Compare against…" entry
 * point. React Query dedupes the session/trace fetches when both the surrounding
 * route and the SessionDetailView read the same session.
 */
export function useSessionCompareRuns(
  workspace: string,
  experimentName: string,
  sessionId: string
): {
  testCaseName: string | undefined;
  runs: ReturnType<typeof useTestCaseRuns>['runs'];
  isRunsLoading: boolean;
} {
  // The session's traces supply the test case name every run is matched on.
  const { testCaseName } = useSessionTrajectories(workspace, sessionId);

  const { data: group } = useGetExperiment(workspace, experimentName);
  const { data: evaluationsPage } = useListEvaluations(
    workspace,
    { filter: { experiment_id: group?.id }, page_size: 1000 },
    { query: { enabled: Boolean(group?.id) } }
  );
  const evaluationNames = useMemo(
    () => evaluationsPage?.data?.map((e) => e.name) ?? [],
    [evaluationsPage]
  );

  const { runs, isLoading: isRunsLoading } = useTestCaseRuns({
    workspace,
    evaluationNames,
    testCaseName,
  });

  return { testCaseName, runs, isRunsLoading };
}
