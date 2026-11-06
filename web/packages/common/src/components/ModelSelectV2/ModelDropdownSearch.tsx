// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Filter } from '@nemo/icons/icons';
import { Block, TextInput } from '@nvidia/foundations-react-core';
import type { ChangeEvent, FC } from 'react';

interface ModelDropdownSearchProps {
  value: string;
  onChange: (value: string) => void;
  name?: string;
  testId?: string;
}

/** Sticky filter box at the top of a searchable dropdown. */
export const ModelDropdownSearch: FC<ModelDropdownSearchProps> = ({
  value,
  onChange,
  name = 'model-filter',
  testId = 'model-select-v2-filter',
}) => (
  <Block className="p-2 w-full sticky top-0 bg-surface z-10">
    <TextInput
      name={name}
      className="overflow-hidden"
      slotStart={<Filter />}
      placeholder="Filter"
      value={value}
      onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.value)}
      attributes={{
        Input: {
          ['data-testid' as never]: testId,
        },
      }}
    />
  </Block>
);
