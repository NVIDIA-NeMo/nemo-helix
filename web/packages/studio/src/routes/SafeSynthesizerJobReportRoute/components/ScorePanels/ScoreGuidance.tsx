// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreItem } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreItem';
import {
  gradeForScore,
  meetsGrade,
  type RecommendedUse,
} from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC } from 'react';

interface ScoreGuidanceProps {
  heading: string;
  intro: string;
  lowScoreGoal: string;
  lowScoreHref: string;
  uses: readonly RecommendedUse[];
  score?: number | null;
}

export const ScoreGuidance: FC<ScoreGuidanceProps> = ({
  heading,
  intro,
  lowScoreGoal,
  lowScoreHref,
  uses,
  score,
}) => {
  const grade = gradeForScore(score);

  return (
    <Stack gap="density-lg" data-testid="score-guidance">
      <Text kind="body/bold/md">{heading}</Text>
      <Text kind="body/regular/sm">{intro}</Text>
      <Text kind="body/regular/sm">
        Low scores suggest you should{' '}
        <Anchor href={lowScoreHref} textKind="body/regular/sm" external>
          adjust your configuration
        </Anchor>{' '}
        to improve {lowScoreGoal}.
      </Text>
      {grade === 'Unavailable' ? (
        <Text kind="body/regular/sm" className="text-secondary">
          Recommended uses appear once a score is available.
        </Text>
      ) : (
        <Stack gap="density-lg">
          {uses.map((use) => (
            <ScoreItem
              key={use.label}
              success={meetsGrade(grade, use.minimumGrade)}
              value={use.label}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
