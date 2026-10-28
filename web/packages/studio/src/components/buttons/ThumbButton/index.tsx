// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ThumbDown, ThumbUp } from '@nemo/icons/icons';
import { Button } from '@nvidia/foundations-react-core';
import { ComponentProps, FC } from 'react';

interface Props extends ComponentProps<typeof Button> {
  direction: 'up' | 'down';
  selected?: boolean;
}
export const ThumbButton: FC<Props> = ({ children, direction, selected = false, ...props }) => {
  const icon = direction === 'up' ? <ThumbUp /> : <ThumbDown />;
  const color = direction === 'up' ? 'brand' : 'danger';
  const activeColor = selected ? color : 'neutral';

  return (
    <Button type="button" color={activeColor} kind="secondary" {...props}>
      {icon}
      {children}
    </Button>
  );
};
