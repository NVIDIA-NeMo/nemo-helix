// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { StartFooter } from '@studio/components/StartOptions/StartFooter';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import type { StartSubPageProps } from '@studio/components/StartOptions/types';
import { ArrowLeft } from 'lucide-react';
import type { FC } from 'react';

/**
 * A start option that needs input before it can continue, on a page of its own: Back to
 * the options, the option's content, and Continue once that content is complete.
 */
export const StartSubPage: FC<StartSubPageProps> = ({
  heading,
  headingDescription,
  onBack,
  children,
  canContinue,
  onContinue,
  blockedHint,
}) => (
  <Stack className="h-full">
    <Stack className="shrink-0 px-6 pt-6">
      <PageHeader slotHeading={heading} slotDescription={headingDescription} />
    </Stack>

    {/* The only scrolling region. */}
    <Block className="min-h-0 flex-1 overflow-auto px-6 pb-6 pt-6 [scrollbar-gutter:stable]">
      <Flex justify="center" className="w-full">
        <Stack className={CONTENT_WIDTH}>{children}</Stack>
      </Flex>
    </Block>

    <StartFooter
      canContinue={canContinue}
      onContinue={onContinue}
      blockedHint={blockedHint}
      slotFooterStart={
        <Button kind="tertiary" onClick={onBack}>
          <ArrowLeft size={16} aria-hidden />
          Back
        </Button>
      }
    />
  </Stack>
);
