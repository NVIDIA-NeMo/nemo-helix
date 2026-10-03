// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { AgentTraceStatistics } from '@studio/components/AgentTraceStatistics';
import type { TraceStatisticsRange } from '@studio/components/AgentTraceStatistics/types';
import { bucketAdverbForRange } from '@studio/components/AgentTraceStatistics/utils';
import {
  AGENT_OVERVIEW_ENABLED,
  INTAKE_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';
import { AgentConfigPanels } from '@studio/routes/agents/AgentDetailRoute/AgentConfigPanels';
import { AgentSummaryPanel } from '@studio/routes/agents/AgentDetailRoute/overview/AgentSummaryPanel';
import { GetStartedPanel } from '@studio/routes/agents/AgentDetailRoute/overview/GetStartedPanel';
import { OpenInsightsPanel } from '@studio/routes/agents/AgentDetailRoute/overview/OpenInsightsPanel';
import { toRecentExperiments } from '@studio/routes/agents/AgentDetailRoute/overview/recentExperiments';
import { RecentExperimentsPanel } from '@studio/routes/agents/AgentDetailRoute/overview/RecentExperimentsPanel';
import { useAgentTraceMetrics } from '@studio/routes/agents/AgentDetailRoute/overview/useAgentTraceMetrics';
import { useOpenInsights } from '@studio/routes/agents/AgentDetailRoute/overview/useOpenInsights';
import { SourcePanel } from '@studio/routes/agents/AgentDetailRoute/SourcePanel';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import {
  getExperimentDetailRoute,
  getIntakeTracesRoute,
  getOptimizerInsightRoute,
  getOptimizerRoute,
} from '@studio/routes/utils';
import { type FC, type ReactNode, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

const SHOW_ACTIVITY = AGENT_OVERVIEW_ENABLED && INTAKE_ENABLED;

interface SummaryTabProps {
  workspace: string;
  agentName?: string;
  agent?: Agent;
  modelNames: string[];
  /** The agent's published evaluations, which the experiment cards are rolled up from. */
  evals: AgentEvaluationRow[];
  isEvalsPending?: boolean;
  /** Jump to the chat tab so the agent emits its first traces. */
  onRunAgent: () => void;
  /** Open the submit-evaluation modal from the experiments empty state. */
  onRunEvaluation?: () => void;
  deploymentsPanel: ReactNode;
}

/** Landing view for an agent: its deployments and activity, next to what it is and how it's built. */
export const SummaryTab: FC<SummaryTabProps> = ({
  workspace,
  agentName,
  agent,
  modelNames,
  evals,
  isEvalsPending,
  onRunAgent,
  onRunEvaluation,
  deploymentsPanel,
}) => {
  const navigate = useNavigate();
  const [range, setRange] = useState<TraceStatisticsRange>('max');
  const {
    summary,
    buckets,
    isPending,
    isPlaceholderData,
    error: traceMetricsError,
  } = useAgentTraceMetrics({
    workspace,
    agentName: agent?.name,
    range,
    enabled: SHOW_ACTIVITY,
  });
  const experiments = useMemo(() => toRecentExperiments(evals), [evals]);
  const awaitingTelemetry =
    !isPending &&
    !isPlaceholderData &&
    !traceMetricsError &&
    range === 'max' &&
    (summary === null || summary.totalTraces === 0);
  const {
    insights,
    totalCount: insightCount,
    isPending: insightsPending,
    error: insightsError,
  } = useOpenInsights({
    workspace,
    agent: agent?.name,
    enabled: SHOW_ACTIVITY && OPTIMIZER_ENABLED,
  });

  return (
    <Flex gap="density-2xl" align="start" wrap="wrap" className="w-full pb-6">
      <Stack gap="density-2xl" className="min-w-0 flex-1 basis-[32rem]">
        {deploymentsPanel}
        {SHOW_ACTIVITY &&
          (awaitingTelemetry ? (
            <GetStartedPanel workspace={workspace} agentName={agent?.name} />
          ) : (
            <AgentTraceStatistics
              summary={summary}
              buckets={buckets}
              range={range}
              onRangeChange={setRange}
              onViewTraces={() =>
                navigate(getIntakeTracesRoute(workspace, { agentName: agent?.name }))
              }
              onRunAgent={onRunAgent}
              isPending={isPending}
              caption={bucketAdverbForRange(range)}
              error={traceMetricsError}
            />
          ))}
        {SHOW_ACTIVITY && (
          <RecentExperimentsPanel
            favorites={experiments.favorites}
            experiments={experiments.recent}
            isPending={isEvalsPending}
            onOpenExperiment={(experiment) =>
              experiment.name && navigate(getExperimentDetailRoute(workspace, experiment.name))
            }
            onRunEvaluation={onRunEvaluation}
          />
        )}
        <AgentConfigPanels agent={agent} />
      </Stack>
      <Stack gap="density-2xl" className="w-full shrink-0 lg:w-90">
        <AgentSummaryPanel agent={agent} modelNames={modelNames} />
        <SourcePanel workspace={workspace} agentName={agent?.name ?? agentName} />
        {SHOW_ACTIVITY && OPTIMIZER_ENABLED && (
          <OpenInsightsPanel
            insights={insights}
            totalCount={insightCount}
            isPending={insightsPending}
            hasError={!!insightsError}
            awaitingTelemetry={awaitingTelemetry}
            onOpenInsight={(insight) => navigate(getOptimizerInsightRoute(workspace, insight.id))}
            onViewAll={() => navigate(getOptimizerRoute(workspace))}
          />
        )}
      </Stack>
    </Flex>
  );
};
