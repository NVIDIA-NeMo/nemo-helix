// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreItem } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreItem';
import {
  gradeForScore,
  meetsGrade,
  type RecommendedUse,
} from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC } from 'react';

interface ScoreGuidanceProps {
  heading: string;
  paragraphs: readonly string[];
  uses: readonly RecommendedUse[];
  score?: number | null;
}

export const ScoreGuidance: FC<ScoreGuidanceProps> = ({ heading, paragraphs, uses, score }) => {
  const grade = gradeForScore(score);

  return (
    <Stack gap="density-lg">
      <Text kind="body/bold/md">{heading}</Text>
      {paragraphs.map((paragraph) => (
        <Text key={paragraph} kind="body/regular/sm">
          {paragraph}
        </Text>
      ))}
      <Stack gap="density-lg">
        {uses.map((use) => (
          <ScoreItem
            key={use.label}
            success={meetsGrade(grade, use.minimumGrade)}
            value={use.label}
          />
        ))}
      </Stack>
    </Stack>
  );
};
