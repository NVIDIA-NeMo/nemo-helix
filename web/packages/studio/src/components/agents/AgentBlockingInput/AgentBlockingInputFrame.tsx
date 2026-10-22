// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { Paperplane } from '@nvidia/react-gui-icons/icons';
import type { AgentBlockingInputFrameProps } from '@studio/components/agents/AgentBlockingInput/types';
import type { FC } from 'react';

const subduedButtonFocusClassName =
  'focus-visible:[outline:1px_solid_var(--border-color-interaction-base)] focus-visible:[outline-offset:-1px] focus-visible:[box-shadow:none]';

export const AgentBlockingInputFrame: FC<AgentBlockingInputFrameProps> = ({
  children,
  isSubmitting = false,
  onSecondaryAction,
  onSkip,
  onSubmit,
  request,
  secondaryActions,
  secondaryActionLabel,
  submitDisabled = false,
  submitLabel = 'Submit',
}) => (
  <Flex
    direction="col"
    role="group"
    aria-label={request.title}
    className="w-full rounded-xl border border-base bg-surface-base p-density-md outline-none"
    data-testid="agent-blocking-input"
  >
    <Stack gap="density-md">
      <Stack gap="density-xs">
        <Text kind="body/semibold/md" className="block">
          {request.title}
        </Text>
        {request.description ? (
          <Text kind="body/regular/sm" className="block text-fg-secondary">
            {request.description}
          </Text>
        ) : null}
      </Stack>
      {children}
      <Flex align="center" justify="end" gap="density-sm">
        {secondaryActions?.map((action) => (
          <Button
            key={action.label}
            type="button"
            kind="tertiary"
            color="neutral"
            disabled={isSubmitting || action.disabled}
            className={subduedButtonFocusClassName}
            onClick={() => void action.onClick()}
          >
            <Text kind="label/regular/sm">{action.label}</Text>
          </Button>
        ))}
        {onSecondaryAction && secondaryActionLabel ? (
          <Button
            type="button"
            kind="tertiary"
            color="neutral"
            disabled={isSubmitting}
            className={subduedButtonFocusClassName}
            onClick={() => void onSecondaryAction()}
          >
            <Text kind="label/regular/sm">{secondaryActionLabel}</Text>
          </Button>
        ) : null}
        {onSkip ? (
          <Button
            type="button"
            kind="tertiary"
            color="neutral"
            disabled={isSubmitting}
            className={subduedButtonFocusClassName}
            onClick={() => void onSkip()}
          >
            <Text kind="label/regular/sm">Skip</Text>
          </Button>
        ) : null}
        <Button
          type="button"
          color="brand"
          size="small"
          disabled={isSubmitting || submitDisabled}
          className={subduedButtonFocusClassName}
          aria-label={submitLabel}
          onClick={() => void onSubmit()}
        >
          <Paperplane size="16" />
        </Button>
      </Flex>
    </Stack>
  </Flex>
);
