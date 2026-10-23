// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronLeft, ChevronRight } from '@nemo/icons/icons';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

const ICON_SIZE = '16';

interface RecordPagerProps {
  readonly index: number;
  readonly total: number;
  readonly onChange: (index: number) => void;
}

export const RecordPager: FC<RecordPagerProps> = ({ index, total, onChange }) => (
  <Flex align="center" gap="density-sm">
    <Button
      aria-label="Previous record"
      disabled={index === 0}
      kind="tertiary"
      onClick={() => onChange(index - 1)}
      type="button"
    >
      <ChevronLeft size={ICON_SIZE} />
    </Button>
    <Text kind="body/regular/md">
      Record {index + 1} of {total}
    </Text>
    <Button
      aria-label="Next record"
      disabled={index >= total - 1}
      kind="tertiary"
      onClick={() => onChange(index + 1)}
      type="button"
    >
      <ChevronRight size={ICON_SIZE} />
    </Button>
  </Flex>
);
