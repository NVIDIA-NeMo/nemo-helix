// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { splitLastSegment } from '@studio/components/MiddleTruncatedPath/splitLastSegment';
import { FC } from 'react';

/** Ellipsizes a path in the middle so its last segment (the file name) stays visible at any width. */
export const MiddleTruncatedPath: FC<{ children: string }> = ({ children }) => {
  const [head, tail] = splitLastSegment(children);
  return (
    <span title={children} className="flex w-full min-w-0 text-left">
      {/* Flex items read as separate words, so assistive tech gets the path whole. */}
      <span className="sr-only">{children}</span>
      <span aria-hidden className="truncate">
        {head}
      </span>
      <span aria-hidden className="shrink-0 max-w-full truncate">
        {tail}
      </span>
    </span>
  );
};
