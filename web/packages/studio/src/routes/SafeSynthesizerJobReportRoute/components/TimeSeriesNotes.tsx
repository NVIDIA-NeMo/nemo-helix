// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { MetricScore } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import type { FC } from 'react';

interface ComparisonDetails {
  skipped?: readonly { group: string | null; column: string; reason: string }[];
  groups_only_in_training_count?: number;
  groups_only_in_synthetic_count?: number;
}

const Note: FC<{ children: string }> = ({ children }) => (
  <Text kind="body/regular/sm" className="text-secondary">
    {children}
  </Text>
);

/** Score notes, skipped comparisons, and one-sided group counts of a time-series metric. */
export const TimeSeriesNotes: FC<{ score: MetricScore; details: ComparisonDetails }> = ({
  score,
  details,
}) => {
  const skipped = details.skipped ?? [];
  return (
    <Stack gap="density-sm">
      {score.notes ? <Note>{score.notes}</Note> : null}
      {skipped.length ? (
        <Stack gap="density-xs">
          <Text kind="label/bold/md">Skipped comparisons</Text>
          <ul className="list-disc pl-density-xl">
            {skipped.map((item) => (
              <li key={`${item.column}-${item.group ?? ''}`}>
                <Text kind="body/regular/sm">
                  {item.column}
                  {item.group !== null ? `, group ${item.group}` : ''}: {item.reason}
                </Text>
              </li>
            ))}
          </ul>
        </Stack>
      ) : null}
      {details.groups_only_in_training_count ? (
        <Note>{`Groups found only in training data: ${details.groups_only_in_training_count}.`}</Note>
      ) : null}
      {details.groups_only_in_synthetic_count ? (
        <Note>{`Groups found only in synthetic data: ${details.groups_only_in_synthetic_count}.`}</Note>
      ) : null}
    </Stack>
  );
};

export const SummaryLines: FC<{ lines: readonly string[] }> = ({ lines }) => (
  <Stack gap="density-sm">
    {lines.map((line) => (
      <Note key={line}>{line}</Note>
    ))}
  </Stack>
);
