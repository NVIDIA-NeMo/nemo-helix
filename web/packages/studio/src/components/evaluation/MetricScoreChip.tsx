// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Stack, Text } from '@nvidia/foundations-react-core';
import { formatScore, scoreColor } from '@studio/components/evaluation/utils';
import { isNaNScore } from '@studio/util/functions';
import { type FC } from 'react';

interface MetricScoreChipProps {
  label: string;
  value?: number | string | null;
}

const NOT_SCORED = 'not scored';

export const MetricScoreChip: FC<MetricScoreChipProps> = ({ label, value }) => {
  const numeric = typeof value === 'number' && Number.isFinite(value) ? value : null;
  const categorical =
    numeric === null && typeof value === 'string' && !isNaNScore(value) ? value : null;

  return (
    <Stack gap="density-xs" className="min-w-0">
      <Text kind="body/regular/sm" className="truncate text-secondary">
        {label}
      </Text>
      {categorical ? (
        <Badge kind="outline" color="gray">
          {categorical}
        </Badge>
      ) : (
        <Badge kind="solid" color={scoreColor(numeric)}>
          {numeric === null ? NOT_SCORED : formatScore(numeric)}
        </Badge>
      )}
    </Stack>
  );
};
