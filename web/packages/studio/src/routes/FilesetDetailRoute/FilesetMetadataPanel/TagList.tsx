// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tag } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export interface TagListProps {
  items: string[];
}

export const TagList: FC<TagListProps> = ({ items }) => (
  <span className="inline-flex flex-wrap gap-density-xs">
    {items.map((item) => (
      <Tag key={item} kind="outline" color="gray" density="compact" readOnly>
        {item}
      </Tag>
    ))}
  </span>
);
