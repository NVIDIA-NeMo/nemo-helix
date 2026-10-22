// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tag, Text } from '@nvidia/foundations-react-core';
import { Close } from '@nvidia/react-gui-icons/icons';
import React, { ComponentProps } from 'react';

import { OverflowGroup } from '../components/OverflowGroup';

type OverflowGroupPropsWithChildren = ComponentProps<typeof OverflowGroup> & {
  children?: React.ReactNode;
};
const OverflowGroupTyped = OverflowGroup as React.ComponentType<OverflowGroupPropsWithChildren>;

interface RenderSelectedValuesProps {
  values: string[];
  placeholder?: string;
  tagProps?: ComponentProps<typeof Tag>;
  attributes?: {
    overflowGroup: ComponentProps<typeof OverflowGroup>;
  };
}

export const renderMultipleSelectedValues = ({
  values,
  tagProps,
  placeholder,
  attributes,
}: RenderSelectedValuesProps) => {
  return values.length ? (
    <OverflowGroupTyped
      onSeeMoreButtonClick={(e) => e.stopPropagation()}
      {...attributes?.overflowGroup}
    >
      {values.map((value) => (
        <Tag key={value} color="gray" density="compact" {...tagProps}>
          {value}
          <Close className="text-secondary" />
        </Tag>
      ))}
    </OverflowGroupTyped>
  ) : (
    <Text className="text-placeholder">{placeholder}</Text>
  );
};
