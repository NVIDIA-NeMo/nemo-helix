// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { Badge, RadioGroupRoot, Stack } from '@nvidia/foundations-react-core';
import {
  TILE_DESCRIPTION_KIND,
  TILE_LABEL_KIND,
  TILE_RADIUS,
} from '@studio/components/StartOptions/tile';
import type { StartOptionCardsProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** KUI names the group from a wrapping FormField only, and the design draws no prompt. */
const GROUP_LABEL = 'How do you want to start?';

/** The stacked "How do you want to start?" cards, one per way in. */
export const StartOptionCards: FC<StartOptionCardsProps> = ({
  options,
  value,
  onChange,
  disabled = false,
}) => (
  <RadioGroupRoot
    name="start-option"
    aria-label={GROUP_LABEL}
    value={value ?? ''}
    onValueChange={onChange}
    disabled={disabled}
    className="w-full"
  >
    <Stack gap="density-md">
      {options.map((option) => (
        <RadioCard
          key={option.id}
          value={option.id}
          label={option.title}
          description={option.description}
          icon={<option.icon size={16} aria-hidden />}
          slotEnd={
            option.tag ? (
              <Badge
                id={`${option.id}-tag`}
                kind={option.tag.kind}
                color={option.tag.color}
                size="medium"
              >
                {option.tag.label}
              </Badge>
            ) : undefined
          }
          attributes={
            option.tag ? { RadioGroupInput: { 'aria-describedby': `${option.id}-tag` } } : undefined
          }
          compact
          labelKind={TILE_LABEL_KIND}
          descriptionKind={TILE_DESCRIPTION_KIND}
          showIndicator={false}
          className={TILE_RADIUS}
          disabled={disabled || !option.enabled}
        />
      ))}
    </Stack>
  </RadioGroupRoot>
);
