// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Badge, Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { ScoreGuidance } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreGuidance';
import { ScoreTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreTable';
import { TitledDial } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/TitledDial';
import { PRIVACY_RECOMMENDED_USES } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC } from 'react';

interface DataPrivacyPanelProps {
  reportSummary?: SafeSynthesizerSummary;
  dpEnabled: boolean;
  title: string;
  icon: React.ReactNode;
}

export const DataPrivacyPanel: FC<DataPrivacyPanelProps> = ({
  reportSummary,
  dpEnabled,
  title,
  icon,
}) => {
  const dpsScore = reportSummary?.data_privacy_score;

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl" className="pb-density-2xl">
        <Stack>
          <TitledDial
            title="Data Privacy Score (DPS)"
            score={dpsScore}
            description="The DPS measures how well synthetic data protects sensitive information from the training data, scoring 0.0–10.0 (higher is better)."
          />
          <ScoreGuidance
            heading="Understanding Privacy Scores"
            paragraphs={[
              'High DPS scores mean adversaries rarely outperform random chance across hundreds of simulated attacks. Scores are penalized for even minor information leakage; mid-range scores still imply significant uncertainty for attackers.',
              'Low scores suggest you should adjust your configuration to improve privacy protection.',
            ]}
            uses={PRIVACY_RECOMMENDED_USES}
            score={dpsScore}
          />
        </Stack>
        <Stack gap="density-lg">
          <ScoreTable
            scores={[
              {
                name: 'Membership Inference Protection',
                description:
                  'Protection against attacks that determine whether a record was in training data.',
                value: reportSummary?.membership_inference_protection_score,
              },
              {
                name: 'Attribute Inference Protection',
                description: 'Protection against attacks that infer sensitive attributes.',
                value: reportSummary?.attribute_inference_protection_score,
              },
            ]}
          />
          <Stack gap="density-sm">
            <Flex gap="density-md" align="center">
              <Text kind="body/semibold/md">Differential Privacy</Text>
              <Badge data-testid="dp-status" kind="solid" color={dpEnabled ? 'green' : 'gray'}>
                {dpEnabled ? 'Enabled' : 'Disabled'}
              </Badge>
            </Flex>
            <Text kind="body/regular/sm" className="text-secondary">
              Differential Privacy offers supplemental protection for mathematical privacy
              guarantees beyond the inherent privacy of synthetic data. It typically comes with a
              reduction in quality and longer training time.
            </Text>
          </Stack>
        </Stack>
      </Grid>
    </Panel>
  );
};
