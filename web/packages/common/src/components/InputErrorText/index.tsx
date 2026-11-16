// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC, PropsWithChildren } from 'react';

/**
 * Sometimes we have to make our own input component that KUI just doesn't offer
 * but we want it to have the same styling. This component exists because KUI doesn't
 * expose the component it uses for `errorText` in its input components, but we want to
 * standardize the way we do it.
 */
export const InputErrorText: FC<PropsWithChildren<ComponentProps<typeof Text>>> = ({
  children,
  ...props
}) => {
  return (
    <Text className="text-feedback-danger" {...props}>
      {children}
    </Text>
  );
};
