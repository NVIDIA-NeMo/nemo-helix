// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, TextInput } from '@nvidia/foundations-react-core';
import { Filter } from '@nvidia/react-gui-icons/icons';
import { ChangeEvent, FC } from 'react';

interface ModelDropdownSearchProps {
  value: string;
  onChange: (value: string) => void;
}

export const ModelDropdownSearch: FC<ModelDropdownSearchProps> = ({ value, onChange }) => {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  };

  return (
    <Block className="p-2 w-full sticky top-0 bg-surface z-10">
      <TextInput
        name="model-filter"
        className="overflow-hidden"
        slotStart={<Filter />}
        placeholder="Filter"
        value={value}
        onChange={handleChange}
        attributes={{
          Input: {
            ['data-testid' as never]: 'model-select-v2-filter',
          },
        }}
      />
    </Block>
  );
};
