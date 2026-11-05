// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Skeleton, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { ComponentProps, FC } from 'react';

type Props = {
  label: string;
  value: string | React.ReactNode;
  defaultValue?: string;
  loading?: boolean;
  size?: 'narrow' | 'medium' | 'wide';
  orientation?: 'horizontal' | 'vertical' | 'auto'; // auto will determine the orientation based on the screen width
  truncate?: boolean;
  attributes?: {
    label?: ComponentProps<typeof Text>;
    value?: ComponentProps<typeof Text>;
  };
};
/**
 * Displays a compact row with a label and a value.
 */
export const KVPair: FC<Props> = ({
  label,
  value,
  size = 'medium',
  defaultValue = '—',
  loading,
  truncate,
  orientation = 'auto',
  attributes,
}) => {
  const displayValue = loading ? (
    <Skeleton />
  ) : value == null || value === '' ? (
    defaultValue
  ) : (
    value
  );

  let orientationClass = '';
  let keyColumnClass = '';
  if (orientation === 'vertical') {
    orientationClass = 'flex-col items-start gap-1';
    keyColumnClass = 'w-auto';
  } else if (orientation === 'horizontal') {
    orientationClass = 'flex-row items-baseline gap-2';
    keyColumnClass = size === 'narrow' ? 'w-[96px]' : size === 'medium' ? 'w-[160px]' : 'w-[320px]';
  } else {
    orientationClass = 'flex-col items-start gap-1 sm:flex-row sm:items-baseline sm:gap-2';
    keyColumnClass =
      size === 'narrow' ? 'sm:w-[96px]' : size === 'medium' ? 'sm:w-[160px]' : 'sm:w-[320px]';
  }

  return (
    <Flex className={orientationClass}>
      <Text
        kind="label/regular/sm"
        className={`${keyColumnClass} shrink-0 text-secondary`}
        {...attributes?.label}
      >
        {label}
      </Text>
      <Text
        kind="body/semibold/md"
        className={cn({ truncate, 'text-wrap': !truncate })}
        {...attributes?.value}
      >
        {displayValue}
      </Text>
    </Flex>
  );
};
