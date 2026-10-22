// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Card, Flex, Text } from '@nvidia/foundations-react-core';
import { Close } from '@nvidia/react-gui-icons/icons';
import type { FC, ReactNode } from 'react';

interface TourCardProps {
  title: string;
  body?: ReactNode;
  /** Left-aligned footer text, e.g. "2 of 3". */
  stepLabel?: ReactNode;
  /** Right-aligned footer controls (Next/Back/Skip/Got it…). */
  actions: ReactNode;
  onClose: () => void;
  closeLabel?: string;
}

/**
 * Shared visual chrome for guided-tour popovers (WelcomeTour's TourTooltip and
 * the AgentPanel Coachmark). Positioning/visibility is the caller's concern;
 * this only renders the card, close button, body, and footer.
 */
export const TourCard: FC<TourCardProps> = ({
  title,
  body,
  stepLabel,
  actions,
  onClose,
  closeLabel = 'Close',
}) => (
  <Card
    className="relative bg-surface-base shadow-lg"
    slotHeader={<h2 className="nv-modal-heading">{title}</h2>}
  >
    <Button
      aria-label={closeLabel}
      kind="tertiary"
      color="neutral"
      size="medium"
      className="absolute top-3 right-3"
      onClick={onClose}
    >
      <Close />
    </Button>
    {body && (
      <Text kind="label/regular/md" className="whitespace-normal break-words" lineHeight="125">
        {body}
      </Text>
    )}
    <Flex justify="between" align="center" className="mt-3">
      <Text kind="label/regular/sm" className="text-tertiary">
        {stepLabel ?? ''}
      </Text>
      <Flex gap="density-sm">{actions}</Flex>
    </Flex>
  </Card>
);
