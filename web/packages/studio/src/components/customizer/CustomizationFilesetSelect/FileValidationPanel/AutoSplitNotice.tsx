// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { InfoCircle } from '@nvidia/react-gui-icons/icons';
import { CUSTOMIZER_AUTO_VAL_SPLIT_RATIO } from '@studio/constants/customization';
import { FC } from 'react';

const numberFormatter = new Intl.NumberFormat('en-US');
const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  maximumFractionDigits: 0,
});

export interface AutoSplitNoticeProps {
  trainingRowCount: number;
}

export const AutoSplitNotice: FC<AutoSplitNoticeProps> = ({ trainingRowCount }) => {
  const validationCount = Math.round(trainingRowCount * CUSTOMIZER_AUTO_VAL_SPLIT_RATIO);
  const trainingCount = trainingRowCount - validationCount;
  const trainingPct = percentFormatter.format(1 - CUSTOMIZER_AUTO_VAL_SPLIT_RATIO);
  const validationPct = percentFormatter.format(CUSTOMIZER_AUTO_VAL_SPLIT_RATIO);

  // Per design spec: neutral, not blue/info. Top + bottom borders only (no
  // left/right borders, no background fill) so the notice reads as a quiet
  // contextual aside rather than an alert.
  return (
    <Flex
      align="start"
      gap="density-md"
      className="border-y border-base py-density-md text-fg-subdued"
    >
      <InfoCircle width={16} height={16} className="mt-1 flex-shrink-0" />
      <Stack gap="density-xs">
        <Text kind="body/regular/md">
          No Validation Data found. Training data will be automatically split.
        </Text>
        {trainingRowCount > 0 && (
          <>
            <Text kind="body/regular/sm">
              {trainingPct} ({numberFormatter.format(trainingCount)}) examples will be used for
              training.
            </Text>
            <Text kind="body/regular/sm">
              {validationPct} ({numberFormatter.format(validationCount)}) examples will be used for
              validation.
            </Text>
          </>
        )}
      </Stack>
    </Flex>
  );
};
