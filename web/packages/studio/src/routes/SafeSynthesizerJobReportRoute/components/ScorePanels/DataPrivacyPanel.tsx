// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import {
  Badge,
  Flex,
  Grid,
  Panel,
  Stack,
  TableBody,
  TableDataCell,
  TableRoot,
  TableRow,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { InfoCircle } from '@nvidia/react-gui-icons/icons';
import { ScoreItem } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreItem';
import { ScoreTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/ScoreTable';
import { TitledDial } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/TitledDial';
import {
  getDataPrivacyGradeLabel,
  GRADE_VALUES,
  isPassingGrade,
} from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import { tooltipClassName } from '@studio/styles/common';
import { FC } from 'react';

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
  const dpsScore = reportSummary?.data_privacy_score ?? 0;
  const dpsGrade = dpsScore ? getDataPrivacyGradeLabel(dpsScore) : GRADE_VALUES.UNAVAILABLE;

  return (
    <Panel slotHeading={title} slotIcon={icon} elevation="high" density="standard">
      <Grid cols={{ xs: 1, md: 2 }} gap="density-2xl" className="pb-density-2xl">
        <Stack>
          <Stack>
            <TitledDial
              title="Data Privacy Score (DPS)"
              score={reportSummary?.data_privacy_score}
              description="The Data Privacy Score is determined by the privacy mechanisms you've enabled in the synthetic configuration. The use of these mechanisms helps to ensure that your synthetic data is safe from adversarial attacks."
              grade={dpsGrade}
            />
            <Stack gap="density-lg">
              <Text kind="body/bold/md">Understand what you can do with your data</Text>
              <Stack gap="density-lg">
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.POOR, dpsGrade)}
                  value="Share internally for analytics and reporting"
                />

                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.MODERATE, dpsGrade)}
                  value="Share externally with trusted third-parties"
                />
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.GOOD, dpsGrade)}
                  value="Publish for research and community use"
                />
                <ScoreItem
                  success={isPassingGrade(GRADE_VALUES.VERY_GOOD, dpsGrade)}
                  value="Train production models"
                />
              </Stack>
            </Stack>
          </Stack>
        </Stack>
        <Stack>
          <div>
            <ScoreTable
              scores={[
                {
                  name: 'Membership Inference Protection',
                  description:
                    'Tests whether attackers can determine if specific records were in the training data',
                  value: reportSummary?.membership_inference_protection_score ?? 0,
                  displayValue:
                    reportSummary?.membership_inference_protection_score?.toFixed(1) ?? '—',
                },
                {
                  name: 'Attribute Inference Protection',
                  description:
                    'Tests whether sensitive attributes can be inferred by an attacker when other attributes are known',
                  value: reportSummary?.attribute_inference_protection_score ?? 0,
                  displayValue:
                    reportSummary?.attribute_inference_protection_score?.toFixed(1) ?? '—',
                },
              ]}
            />

            <TableRoot className="bg-transparent w-full" layout="fixed" align="left">
              <TableBody>
                <TableRow>
                  <TableDataCell>
                    <Flex gap="density-sm" align="center" className="overflow-visible">
                      <span className="truncate">Differential Privacy</span>
                      <Tooltip
                        slotContent="Differential Privacy is the gold standard of privacy. If applied, this process adds noise during training to provide mathematical guarantees of privacy, but could result in lower quality results and/or longer training time."
                        side="bottom"
                        className={tooltipClassName}
                      >
                        <InfoCircle className="inline shrink-0" />
                      </Tooltip>
                    </Flex>
                  </TableDataCell>
                  <TableDataCell data-testid="dp-status" className="w-[90px] overflow-visible">
                    {dpEnabled ? (
                      <Badge kind="solid" color="green">
                        On
                      </Badge>
                    ) : (
                      <Badge kind="solid" color="gray">
                        Off
                      </Badge>
                    )}
                  </TableDataCell>
                </TableRow>
              </TableBody>
            </TableRoot>
          </div>
        </Stack>
      </Grid>
    </Panel>
  );
};
