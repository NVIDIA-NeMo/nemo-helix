// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import type { Session, Trace } from '@nemo/sdk/generated/platform/schema';
import { Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { IntakeTelemetryStatusBadge } from '@studio/components/IntakeDetail/IntakeComponents/IntakeTelemetryStatusBadge';
import type {
  HighlightMetric,
  HighlightMetricDetail,
} from '@studio/components/IntakeDetail/IntakeComponents/keyValueTypes';
import { TraceAgentLink } from '@studio/components/IntakeDetail/IntakeComponents/TraceAgentLink';
import {
  buildSessionHighlightMetrics,
  buildTraceHighlightMetrics,
} from '@studio/components/IntakeDetail/IntakeComponents/traceKeyValues';
import { type FC, type ReactNode, useMemo } from 'react';

const TraceSummaryMetricItem: FC<{
  label: string;
  value: ReactNode;
  details?: readonly HighlightMetricDetail[];
}> = ({ label, value, details }) => {
  const hasDetails = !!details && details.length > 0;
  const item = (
    <Stack gap="density-xs" className="w-max shrink-0">
      <Text kind="label/regular/md" className="whitespace-nowrap text-secondary">
        {label}
      </Text>
      {typeof value === 'string' ? (
        <Text
          kind="title/xs"
          className={`whitespace-nowrap text-primary${
            // Dotted underline signals an on-hover breakdown, matching the
            // click-to-expand timestamps in the span templates.
            hasDetails
              ? ' cursor-help underline decoration-dotted decoration-from-font underline-offset-2'
              : ''
          }`}
        >
          {value}
        </Text>
      ) : (
        value
      )}
    </Stack>
  );

  if (!hasDetails) {
    return item;
  }

  return (
    <Tooltip
      side="bottom"
      slotContent={
        <Stack gap="density-xs" className="min-w-[10rem]">
          {details.map((detail) => (
            <div key={detail.id} className="flex items-center justify-between gap-density-xl">
              <Text kind="label/regular/sm" className="whitespace-nowrap text-secondary">
                {detail.label}
              </Text>
              <Text kind="label/regular/sm" className="whitespace-nowrap font-mono">
                {detail.value}
              </Text>
            </div>
          ))}
        </Stack>
      }
    >
      {item}
    </Tooltip>
  );
};

// Timestamps read as data, not headline numbers — small mono, no title styling.
const dateValue = (value: string) => (
  <Text kind="body/regular/sm" className="whitespace-nowrap font-mono text-primary">
    {value}
  </Text>
);

interface SummaryTelemetry {
  workspace: string;
  started_at: string;
  ended_at?: string;
  status: Session['status'];
  agent_name?: string;
}

interface SummaryHeaderProps {
  telemetry: SummaryTelemetry;
  metrics: readonly HighlightMetric[];
}

const SummaryHeader: FC<SummaryHeaderProps> = ({ telemetry, metrics }) => {
  const started = telemetry.started_at ? formatAbsoluteTimestamp(telemetry.started_at) : undefined;
  const ended = telemetry.ended_at ? formatAbsoluteTimestamp(telemetry.ended_at) : undefined;

  return (
    <Flex align="stretch" gap="density-2xl" className="w-full min-w-0">
      <Flex gap="density-2xl" align="start" className="shrink-0">
        <TraceSummaryMetricItem
          label="Status"
          value={<IntakeTelemetryStatusBadge status={telemetry.status} />}
        />
        {started ? <TraceSummaryMetricItem label="Started" value={dateValue(started)} /> : null}
        {ended ? <TraceSummaryMetricItem label="Ended" value={dateValue(ended)} /> : null}
        {telemetry.agent_name ? (
          <TraceSummaryMetricItem
            label="Agent"
            value={
              <Text kind="title/xs" className="whitespace-nowrap text-primary">
                <TraceAgentLink workspace={telemetry.workspace} agentName={telemetry.agent_name} />
              </Text>
            }
          />
        ) : null}
      </Flex>
      <div className="ml-auto flex max-w-full flex-nowrap items-start gap-4 overflow-x-auto">
        {metrics.map((metric) => (
          <TraceSummaryMetricItem
            key={metric.id}
            label={metric.label}
            value={metric.value}
            details={metric.details}
          />
        ))}
      </div>
    </Flex>
  );
};

interface SessionSummaryHeaderProps {
  session: Session;
}

/** Inline session summary strip backed only by the session detail response. */
export const SessionSummaryHeader: FC<SessionSummaryHeaderProps> = ({ session }) => {
  const metrics = useMemo(() => buildSessionHighlightMetrics(session), [session]);
  return <SummaryHeader telemetry={session} metrics={metrics} />;
};

export const TraceSummaryHeader: FC<{ trace: Trace }> = ({ trace }) => {
  const metrics = useMemo(() => buildTraceHighlightMetrics(trace), [trace]);
  return <SummaryHeader telemetry={trace} metrics={metrics} />;
};
