// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import {
  Badge,
  Block,
  Divider,
  Flex,
  PageHeader,
  RadioGroupRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  CONTENT_WIDTH,
  TILE_DESCRIPTION_KIND,
  TILE_LABEL_KIND,
  TILE_RADIUS,
} from '@studio/components/StartOptions/tile';
import type { StartPageProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** KUI names the group from a wrapping FormField only, and the design draws no prompt. */
const GROUP_LABEL = 'How do you want to start?';

/**
 * Pick a way in from the stacked cards, fill in whatever that way needs in the panel
 * below, then Continue. The panel is the caller's: only it knows what its options mean.
 */
export const StartPage: FC<StartPageProps> = ({
  heading,
  headingDescription,
  options,
  value,
  onChange,
  disabled = false,
  slotDetail,
  continueLabel = 'Continue',
  continueLoading = false,
  canContinue,
  onContinue,
  blockedHint,
  slotBanner,
}) => (
  <Stack className="h-full">
    {/* Header and cards hold their place; only the panel under them scrolls. */}
    <Stack gap="density-2xl" className="shrink-0 px-6 pt-6">
      <PageHeader slotHeading={heading} slotDescription={headingDescription} />

      <Flex justify="center" className="w-full">
        <Stack gap="density-2xl" className={CONTENT_WIDTH}>
          {slotBanner}

          {/* Needs its own full width: without it the cards sit narrower than the
              column, and stop lining up with the divider below. */}
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
                    option.tag
                      ? { RadioGroupInput: { 'aria-describedby': `${option.id}-tag` } }
                      : undefined
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

          {/* Sits in the fixed band, so the panel scrolls under it rather than past it. */}
          {slotDetail ? <Divider /> : null}
        </Stack>
      </Flex>
    </Stack>

    {/* The only scrolling region. The panel sizes to its content and scrolls past it. */}
    <Block className="min-h-0 flex-1 overflow-auto px-6 pb-6 pt-4 [scrollbar-gutter:stable]">
      <Flex justify="center" className="w-full">
        <Stack className={CONTENT_WIDTH}>{slotDetail}</Stack>
      </Flex>
    </Block>

    <Flex justify="center" className="shrink-0 border-t border-base bg-surface-base px-10 py-3">
      <Flex align="center" justify="end" gap="density-2xl" className={CONTENT_WIDTH}>
        {!canContinue && blockedHint ? (
          <Text kind="label/regular/md" className="text-secondary">
            {blockedHint}
          </Text>
        ) : null}
        <LoadingButton
          color="brand"
          kind="primary"
          loading={continueLoading}
          onClick={onContinue}
          disabled={!canContinue}
        >
          {continueLabel}
        </LoadingButton>
      </Flex>
    </Flex>
  </Stack>
);
