// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Label, Stack, Text } from '@nvidia/foundations-react-core';
import { TriangleAlert } from 'lucide-react';
import { FC } from 'react';

interface EmptyProps {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  /** Optional call to action rendered below the description. */
  slotAction?: React.ReactNode;
}

export const Empty: FC<EmptyProps> = ({
  title = 'No Content Available',
  description = 'There is currently no content to display.',
  icon = <TriangleAlert className="size-12" />,
  slotAction,
}) => {
  return (
    <Stack
      gap="density-md"
      align="center"
      justify="center"
      padding="density-md"
      className="text-center"
    >
      {icon}
      <header>
        <Text kind="title/sm">{title}</Text>
      </header>
      <Label color="textSecondary">{description}</Label>
      {slotAction}
    </Stack>
  );
};
