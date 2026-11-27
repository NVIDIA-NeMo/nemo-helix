// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { GradeBadge } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/GradeBadge';
import type { FC, ReactNode } from 'react';

interface TitledDialProps {
  title: string;
  score?: number | null;
  description: ReactNode;
}

export const TitledDial: FC<TitledDialProps> = ({ title, score, description }) => (
  <Stack gap="density-sm" data-testid="titled-dial">
    <Flex gap="density-sm">
      <ScoreGauge score={score ?? undefined} size="lg" />
      <Stack align="start" justify="start" gap="density-lg" padding="density-md">
        <Text kind="body/bold/lg">{title}</Text>
        <GradeBadge score={score} />
      </Stack>
    </Flex>
    <Text kind="body/regular/sm">{description}</Text>
    <Divider className="my-density-lg" />
  </Stack>
);
