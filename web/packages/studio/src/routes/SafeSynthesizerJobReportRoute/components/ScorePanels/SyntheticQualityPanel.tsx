// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Grid, Panel, Stack } from '@nvidia/foundations-react-core';
import { ScoreGuidance } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreGuidance';
import { ScoreTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreTable';
import { TitledDial } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/TitledDial';
import { QUALITY_RECOMMENDED_USES } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
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
  const sqsScore = reportSummary?.synthetic_data_quality_score;

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl">
        <Stack>
          <TitledDial
            title="Synthetic Quality Score (SQS)"
            score={sqsScore}
            description="The SQS measures how well synthetic data preserves the statistical properties of your training data, scoring 0.0–10.0 (higher is better) across up to 5 subscores."
          />
          <ScoreGuidance
            heading="Understanding Quality Scores"
            paragraphs={[
              "Some quality loss is expected — and acceptable — since it's inherent to privacy preservation. Mid-range scores are often sufficient for model training. You should test your synthetic data on the downstream use case to assess true viability.",
              'Low scores suggest you should adjust your configuration to improve model learning.',
            ]}
            uses={QUALITY_RECOMMENDED_USES}
            score={sqsScore}
          />
        </Stack>
        <Stack>
          <ScoreTable
            scores={[
              {
                name: 'Column Correlation Stability',
                description: 'Measures how well correlations between columns are preserved.',
                value: reportSummary?.column_correlation_stability_score,
              },
              {
                name: 'Deep Structure Stability',
                description: 'Measures how well structural relationships are maintained.',
                value: reportSummary?.deep_structure_stability_score,
              },
              {
                name: 'Column Distribution Stability',
                description: 'Measures how well value distributions are preserved.',
                value: reportSummary?.column_distribution_stability_score,
              },
              {
                name: 'Text Semantic Similarity',
                description: 'Measures meaning preservation in text fields.',
                value: reportSummary?.text_semantic_similarity_score,
              },
              {
                name: 'Text Structure Similarity',
                description: 'Measures structural pattern preservation in text fields.',
                value: reportSummary?.text_structure_similarity_score,
              },
            ]}
          />
        </Stack>
      </Grid>
    </Panel>
  );
};
