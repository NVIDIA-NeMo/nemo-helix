// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { SegmentedControl, Stack } from '@nvidia/foundations-react-core';
import { useLatestAnalysisRun } from '@studio/api/useLatestAnalysisRun';
import { InsightJobsDataView } from '@studio/components/dataViews/InsightJobsDataView';
import { InsightsDataView } from '@studio/components/dataViews/InsightsDataView';
import { InsightsMetrics } from '@studio/routes/agents/AgentDetailRoute/InsightsMetrics';
import { InsightsView, VIEW_SEARCH_PARAM } from '@studio/routes/agents/AgentDetailRoute/tabs';
import { type FC, useCallback } from 'react';
import { useSearchParams } from 'react-router';

const VIEW_ITEMS = [
  { value: InsightsView.Insights, children: 'Insights' },
  { value: InsightsView.Jobs, children: 'Jobs' },
];

interface InsightsTabProps {
  workspace: string;
  agentName?: string;
  agent?: Agent;
}

/** The agent's insights, or the analysis jobs that produced them. */
export const InsightsTab: FC<InsightsTabProps> = ({ workspace, agentName, agent }) => {
  const agentWorkspace = agent?.workspace ?? workspace;
  const name = agent?.name ?? agentName;

  // Toasts when a run finishes and refreshes the insights it found, whichever view is showing.
  useLatestAnalysisRun(agentWorkspace, name);

  const [searchParams, setSearchParams] = useSearchParams();
  const view =
    searchParams.get(VIEW_SEARCH_PARAM) === InsightsView.Jobs
      ? InsightsView.Jobs
      : InsightsView.Insights;
  const setView = useCallback(
    (next: string) => {
      setSearchParams(
        (previous) => {
          const params = new URLSearchParams(previous);
          if (next === InsightsView.Jobs) params.set(VIEW_SEARCH_PARAM, next);
          else params.delete(VIEW_SEARCH_PARAM);
          return params;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

  return (
    <Stack gap="density-lg" className="w-full pb-6">
      {name ? <InsightsMetrics workspace={agentWorkspace} agent={name} /> : null}
      <SegmentedControl
        className="w-fit"
        aria-label="Insights view"
        value={view}
        onValueChange={setView}
        items={VIEW_ITEMS}
      />
      {name && view === InsightsView.Insights ? (
        <InsightsDataView workspace={agentWorkspace} agent={name} />
      ) : null}
      {name && view === InsightsView.Jobs ? (
        <InsightJobsDataView workspace={agentWorkspace} agent={name} />
      ) : null}
    </Stack>
  );
};
