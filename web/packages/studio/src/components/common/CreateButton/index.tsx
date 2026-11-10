// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, type ButtonProps } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export const CreateButton: FC<Omit<ButtonProps, 'color'>> = ({ children, ...buttonProps }) => {
  return (
    <Button color="brand" {...buttonProps}>
      {children}
    </Button>
  );
};
