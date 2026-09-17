// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { Grid } from '@nvidia/foundations-react-core';
import type { StudyResults } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import type { FC } from 'react';

const EM_DASH = '—';

/** Optuna's `TrialState.COMPLETE`, as written to the `state` column. */
const COMPLETE = 'COMPLETE';

const formatScore = (value: number | null): string =>
  value === null ? EM_DASH : value.toLocaleString(undefined, { maximumFractionDigits: 4 });

export interface StudyStatTilesProps {
  results: StudyResults;
}

export const StudyStatTiles: FC<StudyStatTilesProps> = ({ results }) => {
  const { summary, trials, metricNames } = results;

  const totalTrials = summary?.nTrials ?? trials.length;
  const frontierCount = trials.filter((trial) => trial.paretoOptimal).length;

  const primaryMetric = metricNames[0];
  const bestValue = summary?.bestValues[0] ?? null;

  const timedTrials = trials.filter(
    (trial) => trial.state === COMPLETE && trial.durationSeconds !== null
  );
  const averageDurationMs = timedTrials.length
    ? (timedTrials.reduce((total, trial) => total + (trial.durationSeconds ?? 0), 0) /
        timedTrials.length) *
      1_000
    : null;

  return (
    <Grid cols={{ base: 1, md: 2, xl: 4 }} gap="density-xl" className="shrink-0">
      <StatTile
        variant="metric"
        label="Best score"
        value={formatScore(bestValue)}
        trailingLabel={primaryMetric}
      />
      <StatTile
        variant="metric"
        label="Trials"
        value={totalTrials ? String(totalTrials) : EM_DASH}
      />
      <StatTile
        variant="metric"
        label="Trials on frontier"
        value={trials.length ? String(frontierCount) : EM_DASH}
      />
      <StatTile
        variant="metric"
        label="Avg trial duration"
        value={averageDurationMs === null ? EM_DASH : formatDurationMs(averageDurationMs)}
      />
    </Grid>
  );
};
