// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { EvalComparisonEntry } from '@studio/components/dataViews/EvalComparisonTable/types';
import type { FC } from 'react';

export interface ComparisonColumnHeaderProps {
  readonly evaluation: EvalComparisonEntry;
  readonly isBaseline?: boolean;
}

export const ComparisonColumnHeader: FC<ComparisonColumnHeaderProps> = ({
  evaluation,
  isBaseline = false,
}) => (
  <Stack gap="density-xxs">
    <Flex align="center" gap="density-sm">
      <Text className="truncate" kind="body/semibold/md">
        {evaluation.label}
      </Text>
      {isBaseline && (
        <Badge color="gray" kind="outline">
          Baseline
        </Badge>
      )}
    </Flex>
    <Text className="text-secondary" kind="body/regular/sm">
      {evaluation.createdAt ? <RelativeTime datetime={evaluation.createdAt} /> : '–'}
    </Text>
  </Stack>
);
