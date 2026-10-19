// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useInsightsGetInsight } from '@nemo/sdk/generated/insights/insights-insights';
import { useGetEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import { useGetExperiment } from '@nemo/sdk/generated/platform/experiments';
import { Badge, Card, Flex, PageHeader, Stack, Text } from '@nvidia/foundations-react-core';
import { EvaluationSessionsDataView } from '@studio/components/dataViews/EvaluationSessionsDataView';
import { OriginatingInsightLink } from '@studio/components/OriginatingInsightLink';
import { OPTIMIZER_ENABLED } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { EvaluationDetailMetrics } from '@studio/routes/EvaluationDetailRoute/EvaluationDetailMetrics';
import { getExperimentDetailRoute, getExperimentRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC } from 'react';

export const EvaluationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { experimentName, evaluationName } = useRequiredPathParams([
    ROUTE_PARAMS.experimentName,
    ROUTE_PARAMS.evaluationName,
  ]);
  const { data: evaluation } = useGetEvaluation(workspace, evaluationName);
  // Evaluations reach their originating insight through the owning group's insight_id.
  const { data: experiment } = useGetExperiment(workspace, experimentName);
  const insightId = experiment?.insight_id ?? '';
  const { data: insight } = useInsightsGetInsight(workspace, insightId, {
    query: { enabled: OPTIMIZER_ENABLED && Boolean(insightId) },
  });
  const showInsightCard = Boolean(insight?.description);

  useBreadcrumbs({
    items: [
      { href: getExperimentRoute(workspace), slotLabel: 'Experiments' },
      {
        href: getExperimentDetailRoute(workspace, experimentName),
        slotLabel: experimentName,
      },
      { slotLabel: evaluationName },
    ],
  });

  return (
    <AccessibleTitle title={evaluationName}>
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={evaluationName}
          slotDescription={showInsightCard ? undefined : evaluation?.description || undefined}
        />
        <EvaluationDetailMetrics evaluationName={evaluationName} />
        {showInsightCard ? (
          <Card className="!h-fit">
            <Stack className="gap-density-md">
              <Flex className="items-start justify-between gap-density-md">
                <Text kind="label/bold/lg">Insight description</Text>
                <OriginatingInsightLink insightId={insightId} />
              </Flex>
              <Text kind="body/regular/md">{insight?.description}</Text>
            </Stack>
          </Card>
        ) : null}
        <div className="flex flex-col gap-4 border-t border-base pt-4">
          <div className="flex items-center gap-3">
            <Text kind="title/sm">Test cases</Text>
            {evaluation?.run_count !== undefined && (
              <Badge color="gray" kind="solid" className="text-sm">
                {evaluation.run_count}
              </Badge>
            )}
          </div>
          <EvaluationSessionsDataView
            evaluationName={evaluationName}
            experimentName={experimentName}
          />
        </div>
      </Stack>
    </AccessibleTitle>
  );
};
