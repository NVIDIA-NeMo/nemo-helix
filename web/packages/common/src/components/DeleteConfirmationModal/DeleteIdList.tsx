// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { List, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

const VISIBLE_ID_COUNT = 3;

interface DeleteIdListProps {
  ids: readonly string[];
}

export const DeleteIdList: FC<DeleteIdListProps> = ({ ids }) => {
  const hiddenCount = ids.length - VISIBLE_ID_COUNT;

  return (
    <Stack gap="density-xs">
      <List
        className="break-all"
        items={ids.slice(0, VISIBLE_ID_COUNT).map((id) => ({
          children: <Text kind="mono/md">{id}</Text>,
        }))}
      />
      {hiddenCount > 0 && <Text>and {hiddenCount} more</Text>}
    </Stack>
  );
};
