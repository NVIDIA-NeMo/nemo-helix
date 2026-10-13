// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { renderMultipleSelectedValues } from '@nemo/common/src/utils/form';
import { getColorsFromLength } from '@nemo/common/src/utils/formatters';
import { useEvalsListEvaluateJobResults } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvaluateJob } from '@nemo/sdk/generated/evals/schema';
import {
  Flex,
  Panel,
  Select,
  Spinner,
  Stack,
  StatusMessage,
  Text,
} from '@nvidia/foundations-react-core';
import { prettifyName } from '@studio/util/evaluations';
import { useQuery } from '@tanstack/react-query';
import { ChartLine, ChartNetwork, TriangleAlert } from 'lucide-react';
import { useState, ReactNode } from 'react';

interface AggregateScores {
  scores: Record<string, Record<string, number>>;
}

interface ComparisonPanelProps {
  job: EvaluateJob;
  workspace: string;
  jobName: string;
}

export const ComparisonPanel = ({ job, workspace, jobName }: ComparisonPanelProps) => {
  const [filterMetrics, setFilterMetrics] = useState<string[]>([]);

  const { status } = job ?? {};
  const isPendingStatus = status === 'pending' || status === 'active';

  const { data: resultsPage, isLoading: isLoadingMetadata } = useEvalsListEvaluateJobResults(
    workspace,
    jobName,
    {
      query: {
        enabled: !!workspace && !!jobName,
      },
    }
  );

  const aggregateScoresResult = resultsPage?.data?.find((r) => r.name === 'aggregate-scores');

  const { data: scores, isLoading: isLoadingScores } = useQuery({
    queryKey: [
      'evaluation-job-result-scores',
      workspace,
      jobName,
      aggregateScoresResult?.download_url,
    ],
    queryFn: async () => {
      if (!aggregateScoresResult?.download_url) {
        throw new Error('No download URL available');
      }
      const response = await fetch(aggregateScoresResult.download_url);
      if (!response.ok) {
        throw new Error(`Failed to download results: ${response.statusText}`);
      }
      return response.json() as Promise<AggregateScores>;
    },
    enabled: !!aggregateScoresResult?.download_url,
  });

  const isPendingResult = isLoadingMetadata || isLoadingScores;

  // Inline useJobsAndResultsV2 logic typed for EvaluateJob
  const combined = [{ job, result: scores }];
  const colorList = getColorsFromLength(combined.length);
  const allRowsWithColors = combined.map((row, idx) => ({ ...row, color: colorList[idx] }));

  const metricOpts = Array.from(
    new Set(
      combined.flatMap((row) => {
        if (!row.result?.scores) return [];
        return Object.keys(row.result.scores).map((metricName) => prettifyName(metricName));
      })
    )
  );

  const metricsToShow = metricOpts.filter((opt) => !filterMetrics.includes(opt));

  const rows = allRowsWithColors.map((row) => {
    if (!row.result?.scores) return row;
    const filteredScores: Record<string, Record<string, number>> = {};
    Object.entries(row.result.scores).forEach(([metricName, metricScores]) => {
      const prettified = prettifyName(metricName);
      if (metricsToShow.includes(prettified)) {
        filteredScores[prettified] = metricScores;
      }
    });
    return { ...row, result: { scores: filteredScores } };
  });

  const allMetrics = metricOpts;
  const activeMetrics = metricsToShow;

  const handleFilterMetrics = (remaining: string[]) => {
    setFilterMetrics(allMetrics.filter((opt) => !remaining.includes(opt)));
  };

  const handleTagClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (e.currentTarget.textContent) {
      setFilterMetrics([...filterMetrics, e.currentTarget.textContent]);
    }
  };

  const isLoading = isPendingResult;
  const failedStatus = status === 'error' || status === 'cancelled' || status === 'cancelling';
  const greenStatus = status === 'completed';

  let content: ReactNode;
  if (isPendingStatus) {
    content = (
      <Stack className="max-w-[500px] mx-auto" gap="density-lg">
        <Spinner
          size="large"
          description="Evaluation in progress. Results will be available once complete."
        />
      </Stack>
    );
  } else if (failedStatus) {
    content = (
      <StatusMessage
        className="mx-auto"
        size="medium"
        slotMedia={<TriangleAlert width={65} height={65} />}
        slotHeading="Job Failed"
        slotSubheading="The evaluation job failed. No comparison data available."
      />
    );
  } else if (isLoading) {
    content = <Spinner size="large" description="Loading comparison data..." />;
  } else if (!scores || rows.length === 0) {
    content = (
      <TableEmptyState
        header="No Results"
        emptyMessage="No evaluation results available for comparison"
        icon={<ChartNetwork />}
      />
    );
  } else {
    const row = rows[0];
    content = (
      <Stack gap="4" className="w-full">
        <Text kind="title/sm">Metrics Comparison</Text>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border-default">
                <th className="text-left p-2">
                  <Text kind="label/semibold/sm">Metric</Text>
                </th>
                {Object.keys(row.result?.scores?.[Object.keys(row.result?.scores)[0]] || {}).map(
                  (scoreName) => (
                    <th key={scoreName} className="text-right p-2">
                      <Text kind="label/semibold/sm">{scoreName}</Text>
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {Object.entries(row.result?.scores || {}).map(([metricName, metricScores]) => (
                <tr key={metricName} className="border-b border-border-subtle">
                  <td className="p-2">
                    <Text kind="body/regular/sm">{metricName}</Text>
                  </td>
                  {Object.values(metricScores).map((value, idx) => (
                    <td key={idx} className="text-right p-2">
                      <Text kind="body/semibold/sm">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </Text>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Stack>
    );
  }

  return (
    <Panel
      elevation="high"
      slotIcon={<ChartLine />}
      slotHeading={
        <Flex align="center" justify="between" gap="density-lg">
          Detailed Metrics
          {greenStatus && allMetrics.length > 0 && (
            <Select
              id="select-metrics"
              multiple
              placeholder="Select metrics"
              items={allMetrics}
              value={activeMetrics}
              onValueChange={handleFilterMetrics}
              renderValue={(values) =>
                renderMultipleSelectedValues({
                  values,
                  placeholder: 'Select metrics',
                  tagProps: {
                    onClick: handleTagClick,
                  },
                  attributes: {
                    overflowGroup: {
                      kind: 'popover',
                    },
                  },
                })
              }
              attributes={{
                SelectTrigger: {
                  className: 'max-w-[300px] [&_.nv-input-content]:overflow-x-auto',
                },
              }}
            />
          )}
        </Flex>
      }
    >
      {content}
    </Panel>
  );
};
