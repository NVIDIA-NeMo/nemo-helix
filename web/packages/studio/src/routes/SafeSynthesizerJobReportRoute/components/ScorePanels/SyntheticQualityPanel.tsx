// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreItem } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreItem';
import { ScoreTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreTable';
import { TitledDial } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/TitledDial';
import {
  getSyntheticQualityGradeLabel,
  GRADE_VALUES,
  isPassingGrade,
} from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC } from 'react';

interface SyntheticQualityPanelProps {
  reportSummary?: SafeSynthesizerSummary;
  title: string;
  icon: React.ReactNode;
}

export const SyntheticQualityPanel: FC<SyntheticQualityPanelProps> = ({
  reportSummary,
  title,
  icon,
}) => {
  const sqsScore = reportSummary?.synthetic_data_quality_score ?? 0;
  const sqsGrade = sqsScore ? getSyntheticQualityGradeLabel(sqsScore) : GRADE_VALUES.UNAVAILABLE;

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl">
        <Stack>
          <Stack>
            <TitledDial
              title="Synthetic Quality Score (SQS)"
              score={reportSummary?.synthetic_data_quality_score}
              description="The Synthetic Quality Score is computed by taking a weighted combination of the individual quality metrics: Column Distribution Stability, Column Correlation Stability, Deep Structure Stability, Text Semantic Similarity and Text Structure Similarity."
              grade={sqsGrade}
            />
            <Stack gap="density-lg">
              <Text kind="body/bold/md">Understand what you can do with your data</Text>
              <Stack gap="density-lg">
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.MODERATE, sqsGrade)}
                  value="Analyze internally for directional guidance"
                />

                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.MODERATE, sqsGrade)}
                  value="Prototype, test, and run QA"
                />
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.GOOD, sqsGrade)}
                  value="Balance or augment real-world datasets"
                />
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.GOOD, sqsGrade)}
                  value="Train production models"
                />
              </Stack>
            </Stack>
          </Stack>
        </Stack>
        <Stack>
          <ScoreTable
            scores={[
              {
                name: 'Column Correlation Stability',
                description: 'Compares the correlation across every combination of two columns.',
                value: reportSummary?.column_correlation_stability_score ?? 0,
                displayValue: reportSummary?.column_correlation_stability_score?.toFixed(1) ?? '—',
              },
              {
                name: 'Deep Structure Stability',
                description:
                  'Compares the original and synthetic data using Principal Component Analysis to reduce the dimensionality.',
                value: reportSummary?.deep_structure_stability_score ?? 0,
                displayValue: reportSummary?.deep_structure_stability_score?.toFixed(1) ?? '—',
              },
              {
                name: 'Column Distribution Stability',
                description:
                  'Compares the distribution for each column in the original data to the matching column in the synthetic data.',
                value: reportSummary?.column_distribution_stability_score ?? 0,
                displayValue: reportSummary?.column_distribution_stability_score?.toFixed(1) ?? '—',
              },
              {
                name: 'Text Semantic Similarity',
                description:
                  'Compares the semantic meaning of the text columns between the original and synthetic data.',
                value: reportSummary?.text_semantic_similarity_score ?? 0,
                displayValue: reportSummary?.text_semantic_similarity_score?.toFixed(1) ?? '—',
              },
              {
                name: 'Text Structure Similarity',
                description:
                  'Compares the sentence, word, and character counts across text columns in the original and synthetic data.',
                value: reportSummary?.text_structure_similarity_score ?? 0,
                displayValue: reportSummary?.text_structure_similarity_score?.toFixed(1) ?? '—',
              },
            ]}
          />
        </Stack>
      </Grid>
    </Panel>
  );
};
