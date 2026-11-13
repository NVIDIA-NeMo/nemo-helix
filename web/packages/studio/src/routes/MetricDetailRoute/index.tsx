// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import {
  getEvalsListMetricsQueryKey,
  useEvalsDeleteMetric,
  useEvalsGetMetric,
} from '@nemo/sdk/generated/evals/evals-plugin-metrics-routes';
import {
  Anchor,
  Button,
  Flex,
  PageHeader,
  PanelContent,
  PanelHeader,
  PanelRoot,
  Stack,
  Table,
  Tag,
  Text,
} from '@nvidia/foundations-react-core';
import {
  metricOutputNames,
  toMetricRuns,
  toMetricTrendSeries,
} from '@studio/api/metrics/metricScores';
import {
  METRIC_HISTORY_LIMIT,
  useMetricScoreHistory,
} from '@studio/api/metrics/useMetricScoreHistory';
import { MetricTrendPanel } from '@studio/components/charts/MetricTrendPanel';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { StackedSkeleton } from '@studio/components/StackedSkeleton';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { runHref, targetHref } from '@studio/routes/MetricDetailRoute/runLinks';
import { getMetricDetailRoute, getMetricsRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';

const RECENT_RUN_COUNT = 20;

/** Truncating text that links when there is somewhere to go. */
const RowLink: FC<{ href?: string; children: string }> = ({ href, children }) =>
  href ? (
    <Anchor asChild className="block truncate">
      <Link to={href}>{children}</Link>
    </Anchor>
  ) : (
    <Text className="truncate">{children}</Text>
  );

export const MetricDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { metricName = '' } = useParams<{ metricName: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const { mutateAsync: deleteMetric } = useEvalsDeleteMetric();

  useBreadcrumbs({
    items: [
      { href: getMetricsRoute(workspace), slotLabel: 'Metrics' },
      { href: getMetricDetailRoute(workspace, metricName), slotLabel: metricName },
    ],
  });

  const { data: metric, isPending, error } = useEvalsGetMetric(workspace, metricName);
  const history = useMetricScoreHistory(workspace);
  const runs = useMemo(
    () => (metric ? toMetricRuns(metric, history.results) : []),
    [metric, history.results]
  );
  const series = useMemo(() => (metric ? toMetricTrendSeries(metric, runs) : []), [metric, runs]);
  const recentRuns = useMemo(() => runs.slice(-RECENT_RUN_COUNT).reverse(), [runs]);

  const handleDelete = async () => {
    try {
      await deleteMetric({ workspace, name: metricName });
    } catch {
      return false;
    }
    await queryClient.invalidateQueries({ queryKey: getEvalsListMetricsQueryKey(workspace) });
    navigate(getMetricsRoute(workspace));
    return true;
  };

  if (error) return <ErrorPanel errorMessage={getErrorMessage(error)} />;
  if (isPending) return <StackedSkeleton count={3} height={120} className="p-density-2xl" />;

  const outputs = metricOutputNames(metric);

  return (
    <AccessibleTitle title={metric.name}>
      <Stack gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex align="center" gap="density-md">
              {metric.name}
              <Tag readOnly density="compact" color="gray">
                {metric.metric_type}
              </Tag>
              <FeatureFlagBadge flag="metricsEnabled" />
            </Flex>
          }
          slotDescription={metric.description}
          slotActions={
            <Button kind="secondary" color="danger" onClick={() => setIsDeleteOpen(true)}>
              Delete
            </Button>
          }
        />

        <MetricTrendPanel
          title="Score trend"
          description={`Aggregate score per run, across the newest ${METRIC_HISTORY_LIMIT} dataset and agent evaluations in this workspace.`}
          series={series}
          isPending={history.isPending}
          valueLabel="Latest run"
          comparisonLabel="vs. previous run"
          formatValue={formatEvaluatorScore}
        />

        <PanelRoot elevation="mid">
          <PanelHeader>
            <Text kind="label/bold/xl">Recent runs</Text>
          </PanelHeader>
          <PanelContent>
            {recentRuns.length === 0 ? (
              <Text className="text-secondary">
                No evaluation has scored this metric&apos;s outputs yet.
              </Text>
            ) : (
              <Table
                className="w-full bg-transparent"
                layout="fixed"
                align="left"
                columns={[
                  { children: 'Run' },
                  { children: 'Target' },
                  ...outputs.map((name) => ({ children: name })),
                  { children: 'Ran' },
                ]}
                rows={recentRuns.map((run) => ({
                  cells: [
                    { children: <RowLink href={runHref(workspace, run)}>{run.name}</RowLink> },
                    {
                      children: run.targetName ? (
                        <RowLink href={targetHref(workspace, run)}>{run.targetName}</RowLink>
                      ) : (
                        '-'
                      ),
                    },
                    ...outputs.map((name) => ({
                      children: formatEvaluatorScore(run.scores[name]),
                    })),
                    { children: <RelativeTime datetime={run.createdAt} /> },
                  ],
                }))}
              />
            )}
          </PanelContent>
        </PanelRoot>
      </Stack>

      <DeleteConfirmationModal
        open={isDeleteOpen}
        simpleConfirm
        onDelete={handleDelete}
        title={`Delete: ${metric.name}`}
        description="Evaluations that already ran keep their scores. New runs can no longer reference this metric."
        onClose={() => setIsDeleteOpen(false)}
      />
    </AccessibleTitle>
  );
};
