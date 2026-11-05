// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Spinner, type SpinnerProps } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

/**
 * A basic KUI Spinner centered in a full height container. It's most commonly
 * used in a route's `pendingComponent` property, and is used in the router's
 * `defaultPendingComponent`.
 */
export const Loading: FC<
  Omit<SpinnerProps, 'description'> & { description?: string; 'aria-label'?: never }
> = ({ size = 'medium', description = 'Loading...', ...spinnerProps }) => {
  return (
    <Stack justify="center" align="center" className="h-full">
      <Spinner data-testid="spinner" size={size} description={description} {...spinnerProps} />
    </Stack>
  );
};
