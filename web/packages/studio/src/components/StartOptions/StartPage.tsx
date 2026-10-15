// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Divider, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { StartFooter } from '@studio/components/StartOptions/StartFooter';
import { StartOptionCards } from '@studio/components/StartOptions/StartOptionCards';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import type { StartPageProps } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

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
  disabled,
  slotDetail,
  continueLabel,
  continueLoading,
  canContinue,
  onContinue,
  blockedHint,
  slotFooterStart,
  slotBanner,
}) => (
  <Stack className="h-full">
    {/* Header and cards hold their place; only the panel under them scrolls. */}
    <Stack gap="density-2xl" className="shrink-0 px-6 pt-6">
      <PageHeader slotHeading={heading} slotDescription={headingDescription} />

      <Flex justify="center" className="w-full">
        <Stack gap="density-2xl" className={CONTENT_WIDTH}>
          {slotBanner}

          <StartOptionCards
            options={options}
            value={value}
            onChange={onChange}
            disabled={disabled}
          />

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

    <StartFooter
      continueLabel={continueLabel}
      continueLoading={continueLoading}
      canContinue={canContinue}
      onContinue={onContinue}
      blockedHint={blockedHint}
      slotFooterStart={slotFooterStart}
    />
  </Stack>
);
