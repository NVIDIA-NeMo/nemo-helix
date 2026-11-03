// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelpCircle } from '@nemo/icons/icons';
import { Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import { FC } from 'react';

const PatternsTooltipContent: FC = () => (
  <div className={tooltipClassName}>
    <Stack gap="density-xs">
      <Text kind="label/bold/sm">Customizer file discovery</Text>
      <Text kind="body/regular/sm">
        Both <code>.jsonl</code> and <code>.json</code> are accepted everywhere below.
      </Text>
      <Text kind="body/regular/sm">
        <strong>Training (required):</strong> any <code>.jsonl</code>/<code>.json</code> directly
        inside <code>train/</code> or <code>training/</code>, OR root files whose names start with{' '}
        <code>train</code> or <code>training</code>.
      </Text>
      <Text kind="body/regular/sm">
        <strong>Validation (optional):</strong> any <code>.jsonl</code>/<code>.json</code> directly
        inside <code>val/</code>, <code>validation/</code>, or <code>dev/</code>, OR root files
        whose names start with <code>val</code>, <code>validation</code>, or <code>dev</code>.
      </Text>
      <Text kind="body/regular/sm">
        If nothing else matches, root-level <code>.jsonl</code> files (only) are claimed as training
        and auto-split 10% for validation.
      </Text>
    </Stack>
  </div>
);

export interface PatternsTooltipTriggerProps {
  label?: string;
}

export const PatternsTooltipTrigger: FC<PatternsTooltipTriggerProps> = ({
  label = 'Why are no files matching?',
}) => (
  <Tooltip slotContent={<PatternsTooltipContent />} side="top">
    <Flex align="center" gap="density-sm" className="cursor-help text-fg-subdued">
      <HelpCircle width={16} height={16} />
      <Text kind="body/regular/md">{label}</Text>
    </Flex>
  </Tooltip>
);
