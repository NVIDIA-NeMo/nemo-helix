// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Divider, Stack, Text } from '@nvidia/foundations-react-core';
import { TemplateGrid } from '@studio/components/CreateCustomizationStart/TemplateGrid';
import type { StartOptionDetailProps } from '@studio/components/CreateCustomizationStart/types';
import type { FC, ReactNode } from 'react';

/** Whatever the chosen option still needs. "Build from scratch" needs nothing. */
export const StartOptionDetail: FC<StartOptionDetailProps> = ({
  option,
  selectedTemplateId,
  onSelectTemplate,
}) => {
  let content: ReactNode = null;

  if (option.id === 'template') {
    content = (
      <TemplateGrid selectedTemplateId={selectedTemplateId} onSelectTemplate={onSelectTemplate} />
    );
  }

  if (!content) return null;

  return (
    <Stack gap="density-md" className="w-full">
      <Divider />
      <Text kind="label/bold/sm" className="text-secondary">
        {option.title}
      </Text>
      {content}
    </Stack>
  );
};
