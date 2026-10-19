// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IntakeAccordion } from '@nemo/common/src/components/IntakeAccordion';
import type { Trace } from '@nemo/sdk/generated/platform/schema';
import { useGetTrace } from '@nemo/sdk/generated/platform/traces';
import { Banner, Stack, StatusMessage, Text } from '@nvidia/foundations-react-core';
import { KeyValueRows } from '@studio/components/IntakeDetail/IntakeComponents/KeyValueRows';
import { RawJsonDebug } from '@studio/components/IntakeDetail/IntakeComponents/RawJsonDebug';
import {
  buildEvaluationContextEntries,
  buildTraceSummaryEntries,
} from '@studio/components/IntakeDetail/IntakeComponents/traceKeyValues';
import { Loading } from '@studio/components/Layouts/Loading';
import { NotFound } from '@studio/components/Layouts/NotFound';
import {
  type BreadcrumbsItemProps,
  useBreadcrumbs,
} from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getTraceDisplayName } from '@studio/util/intakeTelemetry';
import { CircleAlert } from 'lucide-react';
import { type FC, type ReactNode, useEffect, useMemo } from 'react';

const TRACE_SUMMARY_SECTION = 'trace-summary';
const EVALUATION_CONTEXT_SECTION = 'evaluation-context';

interface TraceDetailViewProps {
  workspace: string;
  traceId: string;
  sessionId: string;
  traceSummary?: Trace;
  traceSummaryStatus: 'loading' | 'resolved' | 'error';
  traceSummaryErrorMessage?: string;
  parentBreadcrumbs: BreadcrumbsItemProps[];
  children: (trace: Trace) => ReactNode;
}

/** Hydrates and renders the selected trace's metadata within a session detail page. */
export const TraceDetailView: FC<TraceDetailViewProps> = ({
  workspace,
  traceId,
  sessionId,
  traceSummary,
  traceSummaryStatus,
  traceSummaryErrorMessage,
  parentBreadcrumbs,
  children,
}) => {
  const {
    data: trace,
    error,
    isLoading,
  } = useGetTrace(workspace, traceId, {
    mode: 'detailed',
  });

  // The session already owns enough trace data to keep its explorer mounted
  // while the selected trace's full payload hydrates in the background.
  const resolvedTrace = trace ?? traceSummary;
  const traceNotFound = error?.response?.status === 404;

  const { setBreadcrumbs } = useBreadcrumbs();
  const traceBreadcrumbLabel = resolvedTrace ? getTraceDisplayName(resolvedTrace) : traceId;

  const summaryEntries = useMemo(
    () => (resolvedTrace ? buildTraceSummaryEntries(resolvedTrace, { workspace }) : []),
    [resolvedTrace, workspace]
  );
  const evaluationEntries = useMemo(
    () => (resolvedTrace ? buildEvaluationContextEntries(resolvedTrace.evaluation_context) : []),
    [resolvedTrace]
  );

  useEffect(() => {
    setBreadcrumbs([...parentBreadcrumbs, { slotLabel: `Trace ${traceBreadcrumbLabel}` }]);
  }, [parentBreadcrumbs, setBreadcrumbs, traceBreadcrumbLabel]);

  if (!resolvedTrace && (isLoading || (traceNotFound && traceSummaryStatus === 'loading'))) {
    return <Loading description="Loading trace..." />;
  }

  if (traceNotFound && !resolvedTrace && traceSummaryStatus === 'error') {
    return (
      <StatusMessage
        className="mx-auto mt-density-2xl"
        size="medium"
        slotMedia={<CircleAlert width={65} height={65} />}
        slotHeading="Error loading trace activity"
        slotSubheading={traceSummaryErrorMessage}
      />
    );
  }

  if (traceNotFound && !resolvedTrace) {
    return (
      <NotFound
        subheader="Trace Not Found"
        message="The trace does not exist or you do not have permission to view it."
      />
    );
  }

  if (error && !resolvedTrace) {
    return (
      <StatusMessage
        className="mx-auto mt-density-2xl"
        size="medium"
        slotMedia={<CircleAlert width={65} height={65} />}
        slotHeading="Error loading trace"
        slotSubheading={error.message}
      />
    );
  }

  if (!resolvedTrace) {
    return null;
  }

  if (resolvedTrace.session_id !== sessionId) {
    return (
      <NotFound
        subheader="Trace Not Found"
        message="The trace does not belong to this session or you do not have permission to view it."
      />
    );
  }

  return (
    <>
      {traceNotFound && (
        <Banner kind="inline" status="info">
          Trace details are still arriving. The activity received so far is shown below.
        </Banner>
      )}
      {children(resolvedTrace)}
      <IntakeAccordion
        variant="section"
        defaultValue={[]}
        items={[
          {
            value: TRACE_SUMMARY_SECTION,
            slotLabel: <Text kind="body/semibold/sm">Attributes</Text>,
            slotContent: (
              <Stack className="min-w-0">
                <KeyValueRows entries={summaryEntries} />
              </Stack>
            ),
          },
          ...(evaluationEntries.length > 0
            ? [
                {
                  value: EVALUATION_CONTEXT_SECTION,
                  slotLabel: <Text kind="body/semibold/sm">Evaluation Context</Text>,
                  slotContent: (
                    <Stack className="min-w-0">
                      <KeyValueRows entries={evaluationEntries} />
                    </Stack>
                  ),
                },
              ]
            : []),
        ]}
      />
      <RawJsonDebug value={resolvedTrace} />
    </>
  );
};
