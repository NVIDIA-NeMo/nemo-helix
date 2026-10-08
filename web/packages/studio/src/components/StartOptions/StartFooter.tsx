// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import type { StartFooterProps } from '@studio/components/StartOptions/types';
import classNames from 'classnames';
import type { FC } from 'react';

/** The pinned bar under a start flow: actions on the selection, then Continue. */
export const StartFooter: FC<StartFooterProps> = ({
  continueLabel = 'Continue',
  continueLoading = false,
  canContinue,
  onContinue,
  blockedHint,
  slotFooterStart,
  attributes,
}) => (
  <Flex
    justify="center"
    {...attributes?.FlexContainer}
    className={classNames(
      'shrink-0 border-t border-base bg-surface-base px-10 py-3',
      attributes?.FlexContainer?.className
    )}
  >
    <Flex align="center" justify="between" gap="density-2xl" className={CONTENT_WIDTH}>
      <Flex align="center" gap="density-md">
        {slotFooterStart}
      </Flex>

      <Flex align="center" gap="density-2xl">
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
  </Flex>
);
