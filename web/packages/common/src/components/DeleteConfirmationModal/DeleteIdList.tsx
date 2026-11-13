// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { List, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

const VISIBLE_ID_COUNT = 3;

interface DeleteIdListProps {
  ids: readonly string[];
}

export const DeleteIdList: FC<DeleteIdListProps> = ({ ids }) => {
  const hiddenCount = ids.length - VISIBLE_ID_COUNT;

  const idItems = ids.slice(0, VISIBLE_ID_COUNT).map((id) => ({
    children: <Text kind="mono/md">{id}</Text>,
    slotMarker: '•',
  }));

  const items =
    hiddenCount > 0
      ? [
          ...idItems,
          { children: <Text kind="mono/md">and {hiddenCount} more</Text>, slotMarker: '•' },
        ]
      : idItems;

  return <List className="break-all" items={items} />;
};
