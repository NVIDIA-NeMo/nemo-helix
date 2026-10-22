// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { IconError } from '@nvidia/react-gui-icons/icons';
import { Loading } from '@studio/components/Layouts/Loading';
import {
  type BreadcrumbsItemProps,
  useBreadcrumbs,
} from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { SessionCompareColumn } from '@studio/routes/EvaluationSessionDetailRoute/SessionCompareColumn';
import { getExperimentDetailRoute, getExperimentRoute } from '@studio/routes/utils';
import { type FC, type ReactNode, useEffect } from 'react';

interface TestCaseCompareProps {
  workspace: string;
  experimentName: string;
  testCaseName: string | null | undefined;
  /** The run shown in the left column (the session the route is on). */
  primarySessionId: string;
  primaryRun: EvaluationSessionResponse | undefined;
  /** The run picked for the right column, or null when none is selected yet. */
  compareSessionId: string | null;
  compareRun: EvaluationSessionResponse | undefined;
  /** True while the group's test-case runs are still loading. */
  isRunsLoading: boolean;
  /** Rendered in the header (the "Compare against…" run selector). */
  slotHeaderActions?: ReactNode;
}

const CompareEmpty: FC<{ heading: string; message: string }> = ({ heading, message }) => (
  <div className="flex h-full flex-col items-center justify-center gap-3 p-density-2xl text-center">
    <IconError className="h-10 w-10 text-feedback-warning" aria-hidden />
    <p className="font-semibold text-primary">{heading}</p>
    <p className="max-w-sm text-sm text-secondary">{message}</p>
  </div>
);

export const TestCaseCompare: FC<TestCaseCompareProps> = ({
  workspace,
  experimentName,
  testCaseName,
  primarySessionId,
  primaryRun,
  compareSessionId,
  compareRun,
  isRunsLoading,
  slotHeaderActions,
}) => {
  const { setBreadcrumbs } = useBreadcrumbs();

  useEffect(() => {
    const breadcrumbs: BreadcrumbsItemProps[] = [
      { slotLabel: 'Experiments', href: getExperimentRoute(workspace) },
      {
        slotLabel: experimentName,
        href: getExperimentDetailRoute(workspace, experimentName),
      },
      { slotLabel: 'Test case comparison' },
    ];
    setBreadcrumbs(breadcrumbs);
  }, [setBreadcrumbs, workspace, experimentName]);

  const heading = testCaseName
    ? `Test case comparison — Test case ${testCaseName}`
    : 'Test case comparison';

  const renderRightColumn = () => {
    if (!testCaseName) {
      return (
        <CompareEmpty
          heading="Cannot compare"
          message="This session has no test case name. Comparison requires a producer-supplied test case name."
        />
      );
    }
    if (compareSessionId && compareRun) {
      return (
        <SessionCompareColumn
          key={compareSessionId}
          workspace={workspace}
          sessionId={compareSessionId}
          run={compareRun}
        />
      );
    }
    if (isRunsLoading) return <Loading description="Loading comparison run…" />;
    if (compareSessionId) {
      return (
        <CompareEmpty
          heading="Run not available"
          message="The selected run could not be found for this test case."
        />
      );
    }
    return (
      <CompareEmpty
        heading="Pick a run to compare"
        message="Choose another run of this test case from the selector above to compare it side by side."
      />
    );
  };

  return (
    <AccessibleTitle title="Test case comparison">
      <Stack className="h-full min-h-0" gap="0">
        <div className="shrink-0 border-b border-base px-density-2xl py-density-lg">
          <PageHeader
            className="p-0"
            slotHeading={heading}
            slotDescription="See how this test case performed across different runs"
            slotActions={slotHeaderActions}
          />
        </div>
        <div className="flex min-h-0 flex-1 divide-x divide-base">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-auto">
              <SessionCompareColumn
                key={primarySessionId}
                workspace={workspace}
                sessionId={primarySessionId}
                run={primaryRun}
              />
            </div>
          </div>
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-auto">{renderRightColumn()}</div>
          </div>
        </div>
      </Stack>
    </AccessibleTitle>
  );
};
