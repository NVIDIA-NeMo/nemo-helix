// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text, type StackProps } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

import { Nebula } from '../../Nebula';

export interface ChatEmptyStateProps extends StackProps {
  slotHeading?: string;
  slotSubheading?: string;
  /**
   * Optional call to action rendered below the subheading — e.g. a button that
   * resolves whatever the empty state is describing.
   */
  slotAction?: ReactNode;
}

export const ChatEmptyState: FC<ChatEmptyStateProps> = ({
  className,
  slotHeading = 'Ready',
  slotSubheading = 'Prompt your model to get started.',
  slotAction,
  ...stackProps
}) => {
  const passedClasses = className?.split(' ') || [];
  return (
    <Stack
      {...stackProps}
      className={['relative'].concat(passedClasses).join(' ')}
      gap="density-md"
      align="center"
      justify="center"
    >
      <Text kind="label/bold/3xl" className="text-center">
        {slotHeading}
      </Text>
      <Text kind="label/regular/lg" className="text-center">
        {slotSubheading}
      </Text>
      {slotAction ? <div className="relative z-10">{slotAction}</div> : null}
      {/* Decorative only: must not intercept clicks on slotAction. */}
      <div className="pointer-events-none absolute top-0 left-0 w-full h-full">
        <Nebula variant="sphere" />
      </div>
    </Stack>
  );
};
