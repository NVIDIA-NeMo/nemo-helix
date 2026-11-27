// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { ArrowRight, CheckCircle, Warning } from '@nemo/icons/icons';
import { Badge, Button, Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { GradeBadge } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/GradeBadge';
import { REPORT_OVERVIEW_INTRO } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type { DatasetCounts } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import type { FC } from 'react';

interface ScoreCardProps {
  title: string;
  score?: number | null;
  onViewDetails?: () => void;
}

const ScoreCard: FC<ScoreCardProps> = ({ title, score, onViewDetails }) => (
  <Panel elevation="high" density="standard" data-testid="overview-score-card">
    <Flex align="center" gap="density-lg">
      <ScoreGauge score={score ?? undefined} size="lg" />
      <Stack gap="density-md" align="start">
        <GradeBadge score={score} />
        <Text kind="body/bold/lg">{title}</Text>
        {onViewDetails ? (
          <Button kind="tertiary" size="small" onClick={onViewDetails}>
            View details
          </Button>
        ) : null}
      </Stack>
    </Flex>
  </Panel>
);

const DatasetCard: FC<{ label: string; rows: number; columns: number; missingPercent: number }> = ({
  label,
  rows,
  columns,
  missingPercent,
}) => (
  <Stack gap="density-md" className="flex-1">
    <Badge kind="outline" color="gray" className="w-fit">
      {label}
    </Badge>
    <Grid cols={3} gap="density-lg">
      <Stack gap="density-xs">
        <Text kind="body/regular/sm" className="text-secondary">
          Rows
        </Text>
        <Text kind="body/bold/lg">{rows.toLocaleString()}</Text>
      </Stack>
      <Stack gap="density-xs">
        <Text kind="body/regular/sm" className="text-secondary">
          Columns
        </Text>
        <Text kind="body/bold/lg">{columns}</Text>
      </Stack>
      <Stack gap="density-xs">
        <Text kind="body/regular/sm" className="text-secondary">
          Missing
        </Text>
        <Text kind="body/bold/lg">{Math.trunc(missingPercent)}%</Text>
      </Stack>
    </Grid>
  </Stack>
);

export const DatasetStatistics: FC<{ dataset: DatasetCounts }> = ({ dataset }) => {
  const memorized = dataset.memorized_lines;
  return (
    <Panel slotHeading="Dataset Statistics" elevation="high" density="standard">
      <Stack gap="density-lg" data-testid="dataset-statistics">
        <Flex align="center" gap="density-xl" wrap="wrap">
          <DatasetCard
            label="Training Data"
            rows={dataset.training_rows}
            columns={dataset.training_cols}
            missingPercent={dataset.training_missing_percent}
          />
          <ArrowRight className="text-secondary" />
          <DatasetCard
            label="Synthetic Data"
            rows={dataset.synthetic_rows}
            columns={dataset.synthetic_cols}
            missingPercent={dataset.synthetic_missing_percent}
          />
        </Flex>
        <Flex align="center" gap="density-sm" data-testid="memorization-status">
          {memorized > 0 ? (
            <Warning className="text-feedback-warning" />
          ) : (
            <CheckCircle className="text-brand" />
          )}
          <Text kind="body/regular/sm">
            Training rows duplicated in Synthetic Data: {memorized.toLocaleString()}
          </Text>
        </Flex>
      </Stack>
    </Panel>
  );
};

interface OverviewTabProps {
  jobId: string | null;
  sqsScore?: number | null;
  dpsScore?: number | null;
  showDps: boolean;
  dataset: DatasetCounts | null;
  onViewQuality?: () => void;
  onViewPrivacy?: () => void;
}

/** The library report's overview: only the two scores and the dataset statistics. */
export const OverviewTab: FC<OverviewTabProps> = ({
  jobId,
  sqsScore,
  dpsScore,
  showDps,
  dataset,
  onViewQuality,
  onViewPrivacy,
}) => (
  <Stack gap="density-2xl" data-testid="overview-tab">
    <Stack gap="density-sm">
      <Text kind="body/regular/md">{REPORT_OVERVIEW_INTRO}</Text>
      {jobId ? (
        <Text kind="body/regular/sm" className="text-secondary">
          Job ID {jobId}
        </Text>
      ) : null}
    </Stack>
    <Grid cols={{ xs: 1, lg: 2 }} gap="density-xl">
      <ScoreCard title="Synthetic Quality Score" score={sqsScore} onViewDetails={onViewQuality} />
      {showDps ? (
        <ScoreCard title="Data Privacy Score" score={dpsScore} onViewDetails={onViewPrivacy} />
      ) : null}
    </Grid>
    {dataset ? <DatasetStatistics dataset={dataset} /> : null}
  </Stack>
);
