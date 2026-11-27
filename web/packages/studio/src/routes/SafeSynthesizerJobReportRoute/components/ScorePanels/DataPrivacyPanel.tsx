// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Anchor, Badge, Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  LINK_SAFE_SYNTHESIZER_DIFFERENTIAL_PRIVACY,
  LINK_SAFE_SYNTHESIZER_DPS,
  LINK_SAFE_SYNTHESIZER_DPS_INTERPRETATION,
} from '@studio/constants/links';
import { ScoreGuidance } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreGuidance';
import { ScoreTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreTable';
import { TitledDial } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/TitledDial';
import { METRIC_COPY } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import { PRIVACY_RECOMMENDED_USES } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC, ReactNode } from 'react';

interface DataPrivacyPanelProps {
  reportSummary?: SafeSynthesizerSummary;
  /** Overrides the summary's DPS, such as with the report data's. */
  score?: number | null;
  dpEnabled: boolean;
  epsilon?: number | null;
  delta?: number | string | null;
  title: string;
  icon: ReactNode;
  /** Per-metric cards; without them the panel lists the summary's subscores. */
  children?: ReactNode;
}

const DifferentialPrivacyStatus: FC<{
  dpEnabled: boolean;
  epsilon?: number | null;
  delta?: number | string | null;
}> = ({ dpEnabled, epsilon, delta }) => (
  <Stack gap="density-sm">
    <Flex gap="density-md" align="center">
      <Text kind="body/semibold/md">Differential Privacy</Text>
      <Badge data-testid="dp-status" kind="solid" color={dpEnabled ? 'green' : 'gray'}>
        {dpEnabled ? 'Enabled' : 'Disabled'}
      </Badge>
      {dpEnabled && epsilon !== undefined && epsilon !== null ? (
        <Text kind="body/regular/sm" className="text-secondary">
          epsilon {epsilon} · delta {delta ?? '—'}
        </Text>
      ) : null}
    </Flex>
    <Text kind="body/regular/sm" className="text-secondary">
      <Anchor href={LINK_SAFE_SYNTHESIZER_DIFFERENTIAL_PRIVACY} textKind="body/regular/sm" external>
        Differential Privacy
      </Anchor>{' '}
      offers supplemental protection for mathematical privacy guarantees beyond the inherent privacy
      of synthetic data. It typically comes with a reduction in quality and longer training time.
    </Text>
  </Stack>
);

export const DataPrivacyPanel: FC<DataPrivacyPanelProps> = ({
  reportSummary,
  score,
  dpEnabled,
  epsilon,
  delta,
  title,
  icon,
  children,
}) => {
  const dpsScore = score ?? reportSummary?.data_privacy_score;

  const dial = (
    <TitledDial
      title="Data Privacy Score (DPS)"
      score={dpsScore}
      description={
        <>
          The{' '}
          <Anchor href={LINK_SAFE_SYNTHESIZER_DPS} textKind="body/regular/sm" external>
            DPS
          </Anchor>{' '}
          measures how well synthetic data protects sensitive information from the training data,
          scoring 0.0–10.0 (higher is better).
        </>
      }
    />
  );
  const guidance = (
    <ScoreGuidance
      heading="Understanding Privacy Scores"
      intro="High DPS scores mean adversaries rarely outperform random chance across hundreds of simulated attacks. Scores are penalized for even minor information leakage; mid-range scores still imply significant uncertainty for attackers."
      lowScoreGoal="privacy protection"
      lowScoreHref={LINK_SAFE_SYNTHESIZER_DPS_INTERPRETATION}
      uses={PRIVACY_RECOMMENDED_USES}
      score={dpsScore}
    />
  );
  const dpStatus = (
    <DifferentialPrivacyStatus dpEnabled={dpEnabled} epsilon={epsilon} delta={delta} />
  );

  if (children) {
    return (
      <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
        <Stack gap="density-2xl">
          <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl">
            <Stack gap="density-lg">
              {dial}
              {dpStatus}
            </Stack>
            {guidance}
          </Grid>
          {children}
        </Stack>
      </Panel>
    );
  }

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl" className="pb-density-2xl">
        <Stack>
          {dial}
          {guidance}
        </Stack>
        <Stack gap="density-lg">
          <ScoreTable
            scores={[
              {
                name: METRIC_COPY.membership_inference_protection.name,
                description: `${METRIC_COPY.membership_inference_protection.description}.`,
                value: reportSummary?.membership_inference_protection_score,
              },
              {
                name: METRIC_COPY.attribute_inference_protection.name,
                description: `${METRIC_COPY.attribute_inference_protection.description}.`,
                value: reportSummary?.attribute_inference_protection_score,
              },
            ]}
          />
          {dpStatus}
        </Stack>
      </Grid>
    </Panel>
  );
};
