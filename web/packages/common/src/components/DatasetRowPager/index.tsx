// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { FC } from 'react';

export interface DatasetRowPagerProps {
  fileName?: string | null;
  rowIndex: number;
  rowCount: number;
  isPartial?: boolean;
  onChange: (rowIndex: number) => void;
  disabled?: boolean;
}

/** Steps through the rows of the selected dataset. */
export const DatasetRowPager: FC<DatasetRowPagerProps> = ({
  fileName,
  rowIndex,
  rowCount,
  isPartial,
  onChange,
  disabled,
}) => (
  <Stack gap="density-xs">
    {fileName ? (
      <Text kind="label/bold/sm" className="truncate text-center">
        {fileName}
      </Text>
    ) : null}
    <Flex align="center" justify="center" gap="density-sm">
      <Button
        kind="secondary"
        size="small"
        aria-label="Previous row"
        disabled={disabled || rowIndex === 0}
        onClick={() => onChange(Math.max(0, rowIndex - 1))}
      >
        <ChevronLeft size={16} />
      </Button>
      <Text kind="body/regular/sm" className="text-secondary">
        Row {rowIndex + 1} of {rowCount}
      </Text>
      <Button
        kind="secondary"
        size="small"
        aria-label="Next row"
        disabled={disabled || rowIndex >= rowCount - 1}
        onClick={() => onChange(rowIndex + 1)}
      >
        <ChevronRight size={16} />
      </Button>
    </Flex>
    {isPartial ? (
      <Text kind="body/regular/sm" className="text-center text-placeholder">
        First {rowCount} rows of a large file.
      </Text>
    ) : null}
  </Stack>
);
