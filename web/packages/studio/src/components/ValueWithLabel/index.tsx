// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Stack,
  type StackProps,
  Text,
  Label,
  type LabelProps,
  Skeleton,
} from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import type { FC, ReactNode } from 'react';

interface Props {
  className?: string;
  label: string;
  labelProps?: LabelProps;
  value?: ReactNode;
  valueStyle?: React.CSSProperties;
  helperText?: ReactNode;
  gap?: StackProps['gap'];
  id?: string;
  loading?: boolean;
}

export const ValueWithLabel: FC<Props> = ({
  className,
  label,
  labelProps,
  value,
  valueStyle,
  helperText,
  gap = 'density-xs',
  id,
  loading,
}) => {
  return (
    <Stack className={className} gap={gap}>
      <Label id={id} {...labelProps}>
        {label}
      </Label>
      {/* eslint-disable-next-line no-restricted-syntax */}
      <span className="break-anywhere" style={valueStyle}>
        {loading ? <Skeleton /> : (value ?? EMPTY_FIELD_VALUE)}
      </span>
      {helperText && <Text fontWeight="light">{helperText}</Text>}
    </Stack>
  );
};
