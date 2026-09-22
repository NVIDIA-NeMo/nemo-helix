// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { PlatformJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useEvaluatorGetEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import type { PlatformJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  Badge,
  Block,
  Flex,
  Grid,
  PageHeader,
  Panel,
  Spinner,
  Stack,
} from '@nvidia/foundations-react-core';
import { EvalAggregateScoresTable } from '@studio/components/evaluation/EvalAggregateScoresTable';
import { DatasetEvalRowResultsPanel } from '@studio/components/evaluation/Jobs/datasetEval/DatasetEvalRowResultsPanel';
import { useDatasetEvalResults } from '@studio/components/evaluation/Jobs/datasetEval/useDatasetEvalResults';
import { DetailsPanel } from '@studio/components/evaluation/Jobs/DetailsPanel';
import { StatusLogsContent } from '@studio/components/evaluation/Jobs/StatusLogsContent';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { FlaskConical, ScrollText } from 'lucide-react';
import { FC } from 'react';

const isTerminal = (status?: string) =>
  !!status && PlatformJobTerminalStatuses.includes(status as never);

export const EvaluationResultDetailsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { id } = useRequiredPathParams([ROUTE_PARAMS.evaluationJobId]);

  const { data: job, error } = useEvaluatorGetEvaluateJob(workspace, id, {
    query: {
      refetchOnMount: 'always',
      refetchInterval: (query) => {
        const status = query.state.data?.status;
        return isTerminal(status) ? false : 5000;
      },
    },
  });

  const {
    scores,
    rows,
    isPending,
    hasFailed,
    isLoadingScores,
    isLoadingRows,
    scoresError,
    rowsError,
  } = useDatasetEvalResults(workspace, id, job?.status);

  useBreadcrumbs({
    items: [
      {
        href: getEvaluationResultsRoute(workspace),
        slotLabel: 'Model Evaluations',
      },
      {
        slotLabel: job?.name ?? id,
      },
    ],
  });

  return (
    <AccessibleTitle title={`Evaluation ${job?.name ?? id}`}>
      <Stack className="overflow-auto" gap="density-2xl" padding="density-2xl">
        <Flex align="center" justify="center" className="w-full">
          <Stack className="w-full max-w-[1200px]" gap="density-2xl">
            <PageHeader
              className="p-0"
              slotHeading={
                <Flex align="center" gap="2">
                  {job?.name ?? id}
                  <Badge kind="outline" color="gray">
                    Dataset-Driven
                  </Badge>
                </Flex>
              }
            />

            <Grid cols={{ base: 1, xl: 2 }} gap="density-2xl">
              <DetailsPanel evaluationJob={job} error={!!error} />
              <Panel
                slotHeading="Scores"
                slotIcon={<FlaskConical />}
                elevation="high"
                density="compact"
              >
                {isPending && (
                  <Block className="text-subtle">
                    Scores are computed once the job reaches a terminal state.
                  </Block>
                )}
                {hasFailed && (
                  <Block className="text-subtle">Scores are not available for this job.</Block>
                )}
                {!isPending && !hasFailed && isLoadingScores && (
                  <Flex justify="center" align="center" className="min-h-[120px] w-full">
                    <Spinner size="small" aria-label="Loading scores..." />
                  </Flex>
                )}
                {!isPending && !hasFailed && !isLoadingScores && !scoresError && (
                  <EvalAggregateScoresTable scores={scores} />
                )}
                {!isPending && !hasFailed && scoresError && (
                  <Block className="text-subtle">Scores could not be loaded.</Block>
                )}
              </Panel>
            </Grid>

            {isPending && (
              <Block className="text-subtle">
                Row results are computed once the job reaches a terminal state.
              </Block>
            )}
            {hasFailed && (
              <Block className="text-subtle">Row results are not available for this job.</Block>
            )}
            {!isPending && !hasFailed && isLoadingRows && (
              <Flex justify="center" align="center" className="min-h-[120px] w-full">
                <Spinner size="small" aria-label="Loading row results..." />
              </Flex>
            )}
            {!isPending && !hasFailed && !isLoadingRows && !rowsError && (
              <DatasetEvalRowResultsPanel rows={rows} />
            )}
            {!isPending && !hasFailed && rowsError && (
              <Block className="text-subtle">Row results could not be loaded.</Block>
            )}

            <AccordionPanel slotHeading="Logs" slotIcon={<ScrollText />}>
              <StatusLogsContent
                workspace={workspace}
                jobName={id}
                jobStatus={job?.status as PlatformJobStatus}
              />
            </AccordionPanel>
          </Stack>
        </Flex>
      </Stack>
    </AccessibleTitle>
  );
};
