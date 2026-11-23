// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { scoreColor } from '@nemo/common/src/components/ScoreGauge/utils';
import { Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { gradeForScore } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC, ReactNode } from 'react';

interface TitledDialProps {
  title: string;
  score?: number | null;
  description: ReactNode;
}

export const TitledDial: FC<TitledDialProps> = ({ title, score, description }) => {
  const tierColor = scoreColor(score ?? Number.NaN);

  return (
    <Stack gap="density-sm" data-testid="titled-dial">
      <Flex gap="density-sm">
        <ScoreGauge score={score ?? undefined} size="lg" />
        <Stack align="start" justify="start" gap="density-lg" padding="density-md">
          <Text kind="body/bold/lg">{title}</Text>
          <span
            data-testid="score-grade"
            className="inline-flex w-fit items-center rounded-sm border px-density-sm text-xs font-bold"
            /* eslint-disable-next-line no-restricted-syntax */
            style={{ borderColor: tierColor, color: tierColor }}
          >
            {gradeForScore(score)}
          </span>
        </Stack>
      </Flex>
      <Text kind="body/regular/sm">{description}</Text>
      <Divider className="my-density-lg" />
    </Stack>
  );
};
