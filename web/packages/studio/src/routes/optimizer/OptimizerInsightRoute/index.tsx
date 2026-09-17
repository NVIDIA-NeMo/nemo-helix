// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getInsightsGetInsightQueryKey,
  getInsightsListInsightsQueryKey,
  useInsightsGetInsight,
  useInsightsUpdateInsight,
} from '@nemo/sdk/generated/insights/insights-insights';
import type { InsightStatus } from '@nemo/sdk/generated/insights/schema';
import {
  Anchor,
  Button,
  Card,
  Divider,
  Flex,
  PageHeader,
  Stack,
  Tag,
  Text,
} from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { Loading } from '@studio/components/Layouts/Loading';
import { LINK_DOCS_STUDIO_EVALUATION } from '@studio/constants/links';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { InsightOpenModal } from '@studio/routes/optimizer/InsightOpenModal';
import { insightActions, insightStatusColor } from '@studio/routes/optimizer/insightStatus';
import { InsightTracesTable } from '@studio/routes/optimizer/InsightTracesTable';
import { InsightExperiments } from '@studio/routes/optimizer/OptimizerInsightRoute/InsightExperiments';
import { getOptimizerRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { Link, useParams } from 'react-router';

export const OptimizerInsightRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { insightId = '' } = useParams<{ insightId: string }>();
  const queryClient = useQueryClient();
  const toast = useToast();

  const {
    data: insight,
    isLoading,
    isError,
    refetch,
  } = useInsightsGetInsight(workspace, insightId);

  const { mutate: updateInsight, isPending: isUpdating } = useInsightsUpdateInsight({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getInsightsGetInsightQueryKey(workspace, insightId),
        });
        queryClient.invalidateQueries({
          queryKey: getInsightsListInsightsQueryKey(workspace),
        });
      },
      onError: () => toast.error('Failed to update insight.'),
    },
  });

  const [openModalOpen, setOpenModalOpen] = useState(false);

  // The external agent changes the status after it creates the experiment.
  const handleAction = (target: InsightStatus) => {
    if (target === 'open') {
      setOpenModalOpen(true);
      return;
    }
    updateInsight({ workspace, insightId, data: { status: target } });
  };

  useBreadcrumbs({
    items: [
      { href: getOptimizerRoute(workspace), slotLabel: 'Insights' },
      { slotLabel: insight?.title ?? insightId },
    ],
  });

  if (isLoading && !insight) {
    return <Loading description="Loading insight..." />;
  }

  if (isError || !insight) {
    return (
      <AccessibleTitle title="Insight">
        <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
          <ErrorMessage
            header="Failed to load insight"
            message="The insight could not be loaded. It may have been deleted or you may not have access."
            slotFooter={
              <Flex gap="density-sm">
                <Button type="button" kind="tertiary" onClick={() => refetch()}>
                  Retry
                </Button>
                <Link to={getOptimizerRoute(workspace)}>
                  <Button kind="secondary">Back to Optimizer</Button>
                </Link>
              </Flex>
            }
          />
        </Stack>
      </AccessibleTitle>
    );
  }

  const traceRefs = insight.trace_refs ?? [];
  const status = insight.status ?? 'open';

  return (
    <AccessibleTitle title={`Insight - ${insight.title}`}>
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex className="items-center gap-density-md">
              {insight.title}
              <FeatureFlagBadge flag="optimizerEnabled" />
            </Flex>
          }
          slotDescription={
            <>
              Insight generated from observed sessions by the analyst agent.{' '}
              <Anchor href={LINK_DOCS_STUDIO_EVALUATION} target="_blank">
                Learn more
              </Anchor>
            </>
          }
          slotActions={
            <Flex gap="density-sm">
              {insightActions(status).map((action) => (
                <Button
                  key={action.target}
                  kind={action.kind}
                  color={action.color}
                  disabled={isUpdating}
                  onClick={() => handleAction(action.target)}
                >
                  {action.label}
                </Button>
              ))}
            </Flex>
          }
        />

        <div className="flex gap-8">
          <KVPair
            label="Status"
            orientation="vertical"
            value={
              <Tag kind="outline" color={insightStatusColor(status)} readOnly>
                {status}
              </Tag>
            }
          />
          <Divider orientation="vertical" className="grow-0 self-stretch" />
          <KVPair label="Agent" orientation="vertical" value={insight.agent || '—'} />
          <Divider orientation="vertical" className="grow-0 self-stretch" />
          <KVPair
            label="Created"
            orientation="vertical"
            value={insight.created_at ? <RelativeTime datetime={insight.created_at} /> : '—'}
          />
          <Divider orientation="vertical" className="grow-0 self-stretch" />
          <KVPair
            label="Updated"
            orientation="vertical"
            value={insight.updated_at ? <RelativeTime datetime={insight.updated_at} /> : '—'}
          />
        </div>

        <div className="flex items-stretch gap-density-lg">
          <Card className="w-1/2">
            <Stack className="h-full gap-density-sm">
              <Text kind="label/bold/md">Description</Text>
              {insight.description ? (
                <ExpandableMessage
                  message={insight.description}
                  attributes={{ Text: { kind: 'body/regular/md' } }}
                />
              ) : (
                <Text kind="body/regular/md">—</Text>
              )}
            </Stack>
          </Card>

          <Card className="w-1/2 min-w-0">
            <InsightExperiments
              workspace={workspace}
              insightId={insightId}
              onRunExperiment={() => handleAction('open')}
              runExperimentDisabled={isUpdating}
            />
          </Card>
        </div>

        <Stack className="gap-density-sm">
          <Text kind="label/bold/md">Observed Sessions ({traceRefs.length})</Text>
          <InsightTracesTable workspace={workspace} traceIds={traceRefs} />
        </Stack>
      </Stack>

      <InsightOpenModal
        open={openModalOpen}
        insight={insight}
        workspace={workspace}
        onClose={() => setOpenModalOpen(false)}
      />
    </AccessibleTitle>
  );
};
