// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FC, type PropsWithChildren, useEffect } from 'react';

interface AccessibleTitleProps {
  title?: string;
}

/**
 * AccessibleTitle is a small wrapper component that updates the document title, which
 * both makes it easier to keep track of if you have many tabs open, and also makes route
 * changes audible to screen readers.
 */
export const AccessibleTitle: FC<PropsWithChildren<AccessibleTitleProps>> = ({
  title,
  children,
}) => {
  useEffect(() => {
    document.title = title ? `${title} - Studio` : 'Studio';
  }, [title]);
  return children;
};
