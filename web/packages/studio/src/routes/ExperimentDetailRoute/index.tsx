// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ChartPerformance, ChartScatterplot, Pencil, Star, LayoutDetail } from '@nemo/icons/icons';
import { useCompassGetInsight } from '@nemo/sdk/generated/compass/compass-insights';
import { useGetExperimentSuspense } from '@nemo/sdk/generated/platform/experiments';
import { Button, Card, Flex, PageHeader, Stack, Text } from '@nvidia/foundations-react-core';
import {
  isTrendChoiceCurrent,
  resolveTrendVisible,
  type TrendVisibilityChoice,
  trendVisibilityStorageKey,
} from '@studio/components/charts/ExperimentTrendChart/visibility';
import { ExperimentDataView } from '@studio/components/dataViews/ExperimentDataView';
import { ExperimentDeleteButton } from '@studio/components/ExperimentDeleteButton';
import { ExperimentEditModal } from '@studio/components/ExperimentEditModal';
import { OriginatingInsightLink } from '@studio/components/OriginatingInsightLink';
import { OPTIMIZER_ENABLED } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ExperimentMetrics } from '@studio/routes/ExperimentDetailRoute/ExperimentMetrics';
import { getExperimentRoute } from '@studio/routes/utils';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC, useEffect, useState } from 'react';
import { useNavigate } from 'react-router';

export const ExperimentDetailRoute: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();
  const { experimentName } = useRequiredPathParams([ROUTE_PARAMS.experimentName]);
  const { data: group } = useGetExperimentSuspense(workspace, experimentName);
  // The insight is a group-level concept, reached via the group's insight_id.
  const { data: insight } = useCompassGetInsight(workspace, group.insight_id ?? '', {
    query: { enabled: OPTIMIZER_ENABLED && Boolean(group.insight_id) },
  });
  const [editOpen, setEditOpen] = useState(false);

  // Pareto (cost-vs-accuracy) view visibility, persisted per group. Hidden by default.
  const [storedParetoVisible, setParetoVisible] = useLocalStorage<boolean>(
    `nemo-studio:experiment-pareto:${group.id}`,
    false
  );
  const paretoVisible = storedParetoVisible ?? false;

  const [storedCompareVisible, setCompareVisible] = useLocalStorage<boolean>(
    `nemo-studio:experiment-compare:${group.id}`,
    false
  );
  const compareVisible = storedCompareVisible ?? false;

  // Over-time trend visibility. The default is the experiment's own `show_evaluations_over_time`
  // flag — a flagged experiment is one whose owner has said its evaluations are comparable, so the
  // chart is worth showing unasked. A viewer's own toggle wins, but only until the flag changes;
  // `resolveTrendVisible` retires the stored choice at that point so the owner's edit takes effect.
  const [storedTrendChoice, setTrendChoice, clearTrendChoice] =
    useLocalStorage<TrendVisibilityChoice>(trendVisibilityStorageKey(group.id));
  const trendFlag = Boolean(group.show_evaluations_over_time);
  const trendVisible = resolveTrendVisible(storedTrendChoice, trendFlag);

  // Drop a choice the flag has moved out from under, rather than leaving it suspended: the stamp
  // lines up again if the flag is turned off and back on, which would revive a choice the owner has
  // twice overruled.
  useEffect(() => {
    if (storedTrendChoice === undefined) return;
    if (!isTrendChoiceCurrent(storedTrendChoice, trendFlag)) clearTrendChoice();
  }, [storedTrendChoice, trendFlag, clearTrendChoice]);

  useBreadcrumbs({
    items: [
      { href: getExperimentRoute(workspace), slotLabel: 'Experiments' },
      { slotLabel: experimentName },
    ],
  });

  return (
    <AccessibleTitle title={experimentName}>
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            group.is_favorite ? (
              <Flex align="center" gap="density-sm">
                <Star
                  size="24"
                  className="text-brand shrink-0"
                  fill="currentColor"
                  aria-label="Favorite"
                />
                {experimentName}
              </Flex>
            ) : (
              experimentName
            )
          }
          slotDescription={group.description || undefined}
          slotActions={
            <Flex gap="density-sm">
              <Button kind="secondary" onClick={() => setEditOpen(true)}>
                <Pencil />
                Edit
              </Button>
              <ExperimentDeleteButton
                workspace={workspace}
                name={group.name}
                onDeleted={() => navigate(getExperimentRoute(workspace))}
              />
            </Flex>
          }
        />
        <ExperimentEditModal
          open={editOpen}
          onClose={() => setEditOpen(false)}
          workspace={workspace}
          group={group}
        />
        <ExperimentMetrics experimentName={experimentName} />
        {(insight?.description || group.summary) && (
          <div className="flex items-start gap-density-lg">
            {insight?.description ? (
              <Card className="min-w-0 flex-1">
                <Stack className="gap-density-md">
                  <Flex className="items-start justify-between gap-density-md">
                    <Text kind="label/bold/lg">Insight description</Text>
                    <OriginatingInsightLink insightId={insight.id} />
                  </Flex>
                  <Text kind="body/regular/md">{insight.description}</Text>
                </Stack>
              </Card>
            ) : null}
            {group.summary ? (
              <Card className="min-w-0 flex-1">
                <Stack className="gap-density-md">
                  <Text kind="title/sm">Summary</Text>
                  <Text kind="body/regular/md">{group.summary}</Text>
                </Stack>
              </Card>
            ) : null}
          </div>
        )}
        <div className="flex flex-col gap-4 border-t border-base pt-4">
          <div className="flex items-center gap-3">
            <Text kind="title/sm">Evaluations</Text>
            <Button
              kind="tertiary"
              aria-pressed={trendVisible}
              onClick={() => setTrendChoice({ visible: !trendVisible, flag: trendFlag })}
            >
              <ChartPerformance width={12} height={12} className="text-brand" />
              {trendVisible ? 'Hide over time' : 'Over time'}
            </Button>
            <Button
              kind="tertiary"
              aria-pressed={paretoVisible}
              onClick={() => setParetoVisible(!paretoVisible)}
            >
              <ChartScatterplot width={12} height={12} className="text-brand" />
              {paretoVisible ? 'Hide Pareto' : 'Pareto view'}
            </Button>
            <Button
              kind="tertiary"
              aria-pressed={compareVisible}
              onClick={() => setCompareVisible(!compareVisible)}
            >
              <LayoutDetail width={12} height={12} className="text-brand" />
              {compareVisible ? 'Hide scorecard' : 'Scorecard'}
            </Button>
          </div>
          <ExperimentDataView
            // The route does not remount between experiments; without this the next inherits this one's columns.
            key={group.id}
            group={group}
            paretoVisible={paretoVisible}
            trendVisible={trendVisible}
            compareVisible={compareVisible}
          />
        </div>
      </Stack>
    </AccessibleTitle>
  );
};
