// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag } from '@nvidia/foundations-react-core';
import { getSourceDisplay } from '@studio/components/dataViews/JobsDataView/sourceDisplay';
import type { ComponentProps, FC } from 'react';

interface SourceTagProps extends Omit<ComponentProps<typeof Tag>, 'children'> {
  source: string;
}

export const SourceTag: FC<SourceTagProps> = ({ source, ...props }) => {
  const { icon, label } = getSourceDisplay(source);
  return (
    <Tag kind="outline" color="gray" density="compact" {...props}>
      <Flex align="center" gap="density-sm">
        {icon}
        {label}
      </Flex>
    </Tag>
  );
};
