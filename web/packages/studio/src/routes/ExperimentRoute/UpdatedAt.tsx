// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface UpdatedAtProps {
  datetime: string;
}

export const UpdatedAt: FC<UpdatedAtProps> = ({ datetime }) => {
  const relative = useRelativeTimeSince(datetime);
  return (
    <Text kind="label/regular/xs" className="text-tertiary">
      Updated {relative}
    </Text>
  );
};
