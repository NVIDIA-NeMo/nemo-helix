// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGetSessionSuspense } from '@nemo/sdk/generated/platform/sessions';
import { Flex } from '@nvidia/foundations-react-core';
import type { SessionDetailRouteContext } from '@studio/components/IntakeDetail/SessionDetailView';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { CompareRunSelect } from '@studio/routes/EvaluationSessionDetailRoute/CompareRunSelect';
import { TestCaseCompare } from '@studio/routes/EvaluationSessionDetailRoute/TestCaseCompare';
import { TestCaseScores } from '@studio/routes/EvaluationSessionDetailRoute/TestCaseScores';
import { TestCaseTabs } from '@studio/routes/EvaluationSessionDetailRoute/TestCaseTabs';
import { useSessionCompareRuns } from '@studio/routes/EvaluationSessionDetailRoute/useSessionCompareRuns';
import { SessionDetailContent } from '@studio/routes/IntakeSessionDetailRoute';
import {
  getEvaluationDetailRoute,
  getEvaluationSessionDetailRoute,
  getExperimentDetailRoute,
  getExperimentRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router';

/**
 * Compare mode: the current session on the left, a picked run of the same test case on the right.
 * Fans out over the group's sibling evaluations to gather every run of the test case (Option A).
 */
const EvaluationSessionCompare: FC<{
  workspace: string;
  experimentName: string;
  sessionId: string;
  compareWith: string;
  onSelectCompare: (sessionId: string) => void;
  onClearCompare: () => void;
}> = ({ workspace, experimentName, sessionId, compareWith, onSelectCompare, onClearCompare }) => {
  const { testCaseName, runs, isRunsLoading } = useSessionCompareRuns(
    workspace,
    experimentName,
    sessionId
  );
  const primaryRun = runs.find((run) => run.session_id === sessionId);
  const compareRun = runs.find((run) => run.session_id === compareWith);

  return (
    <TestCaseCompare
      workspace={workspace}
      experimentName={experimentName}
      testCaseName={testCaseName}
      primarySessionId={sessionId}
      primaryRun={primaryRun}
      compareSessionId={compareWith}
      compareRun={compareRun}
      isRunsLoading={isRunsLoading}
      slotHeaderActions={
        <Flex align="center" gap="density-md">
          <CompareRunSelect
            runs={runs}
            currentSessionId={sessionId}
            value={compareWith}
            onChange={onSelectCompare}
            isLoading={isRunsLoading}
          />
          <button
            type="button"
            className="text-sm text-secondary hover:underline"
            onClick={onClearCompare}
          >
            Clear
          </button>
        </Flex>
      }
    />
  );
};

/**
 * Single session (test case) view with the "Compare against…" entry point in its
 * header. Picking a run sets ?compareWith, which flips the route into compare mode.
 */
const EvaluationSessionDetail: FC<{
  workspace: string;
  experimentName: string;
  sessionId: string;
  routeContext: SessionDetailRouteContext;
  onSelectCompare: (sessionId: string) => void;
}> = ({ workspace, experimentName, sessionId, routeContext, onSelectCompare }) => {
  const { runs, isRunsLoading } = useSessionCompareRuns(workspace, experimentName, sessionId);
  const currentRun = runs.find((run) => run.session_id === sessionId);

  const contextWithCompare = useMemo<SessionDetailRouteContext>(
    () =>
      routeContext.kind === 'evaluation'
        ? {
            ...routeContext,
            renderBody: ({ trajectories, traceView }) => (
              <>
                {currentRun ? <TestCaseScores run={currentRun} /> : null}
                <TestCaseTabs
                  workspace={workspace}
                  trajectories={trajectories}
                  traceView={traceView}
                />
              </>
            ),
            headerActions: (
              <CompareRunSelect
                runs={runs}
                currentSessionId={sessionId}
                value={null}
                onChange={onSelectCompare}
                isLoading={isRunsLoading}
              />
            ),
          }
        : routeContext,
    [routeContext, currentRun, workspace, runs, sessionId, onSelectCompare, isRunsLoading]
  );

  return <SessionDetailContent sessionId={sessionId} routeContext={contextWithCompare} />;
};

export const EvaluationSessionDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [searchParams, setSearchParams] = useSearchParams();
  // compareWith carries the session_id of the run to show in the right column.
  const compareWith = searchParams.get(QUERY_PARAMETERS.compareWith);

  const { sessionId, experimentName, evaluationName } = useRequiredPathParams([
    ROUTE_PARAMS.sessionId,
    ROUTE_PARAMS.experimentName,
    ROUTE_PARAMS.evaluationName,
  ]);
  // Suspends until the session loads, in both views; a missing session renders the route's error state.
  useGetSessionSuspense(workspace, sessionId);

  const onSelectCompare = useCallback(
    (targetSessionId: string) => {
      setSearchParams((previous) => {
        const next = new URLSearchParams(previous);
        next.set(QUERY_PARAMETERS.compareWith, targetSessionId);
        return next;
      });
    },
    [setSearchParams]
  );

  const onClearCompare = useCallback(() => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      next.delete(QUERY_PARAMETERS.compareWith);
      return next;
    });
  }, [setSearchParams]);

  const routeContext = useMemo<SessionDetailRouteContext>(
    () => ({
      kind: 'evaluation',
      parentBreadcrumbs: [
        { slotLabel: 'Experiments', href: getExperimentRoute(workspace) },
        {
          slotLabel: experimentName,
          href: getExperimentDetailRoute(workspace, experimentName),
        },
        {
          slotLabel: evaluationName,
          href: getEvaluationDetailRoute(workspace, experimentName, evaluationName),
        },
      ],
      getSessionHref: (targetSessionId) =>
        getEvaluationSessionDetailRoute(workspace, experimentName, evaluationName, targetSessionId),
    }),
    [workspace, experimentName, evaluationName]
  );

  if (compareWith) {
    return (
      <EvaluationSessionCompare
        workspace={workspace}
        experimentName={experimentName}
        sessionId={sessionId}
        compareWith={compareWith}
        onSelectCompare={onSelectCompare}
        onClearCompare={onClearCompare}
      />
    );
  }

  return (
    <EvaluationSessionDetail
      workspace={workspace}
      experimentName={experimentName}
      sessionId={sessionId}
      routeContext={routeContext}
      onSelectCompare={onSelectCompare}
    />
  );
};
