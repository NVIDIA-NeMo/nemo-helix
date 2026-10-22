// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Switch, Text } from '@nvidia/foundations-react-core';
import { InfoCircle } from '@nvidia/react-gui-icons/icons';
import { type FC } from 'react';

/**
 * Presentation flags an Experiment carries, with the copy shown next to each toggle.
 * Shared by the create and edit modals so the two can't drift apart.
 */
export const EXPERIMENT_FLAGS = {
  show_evaluations_over_time: {
    label: 'Evaluate over time',
    hint: 'Graphed evaluations over time. Note: All evaluations should be run against the same dataset and evaluators.',
  },
  is_favorite: {
    label: 'Favorite',
    hint: 'Favorited experiments are displayed in critical areas of the application such as the agent page and dashboard.',
  },
} as const;

export type ExperimentFlag = keyof typeof EXPERIMENT_FLAGS;

export interface ExperimentFlagSwitchProps {
  flag: ExperimentFlag;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  onBlur?: () => void;
}

/**
 * A switch whose label sits after the control, with a muted info line beneath it. These flags are
 * opt-in and need a sentence of explanation that would not fit in a `FormField` label.
 */
export const ExperimentFlagSwitch: FC<ExperimentFlagSwitchProps> = ({
  flag,
  checked,
  onCheckedChange,
  disabled,
  onBlur,
}) => {
  const { label, hint } = EXPERIMENT_FLAGS[flag];

  return (
    <Stack gap="density-md">
      <Switch
        size="small"
        name={flag}
        checked={checked}
        onCheckedChange={onCheckedChange}
        onBlur={onBlur}
        disabled={disabled}
        slotLabel={label}
      />
      <Flex align="start" gap="density-md" className="text-fg-subdued">
        <InfoCircle width={12} height={12} className="mt-0.5 shrink-0" />
        <Text kind="label/regular/md">{hint}</Text>
      </Flex>
    </Stack>
  );
};
