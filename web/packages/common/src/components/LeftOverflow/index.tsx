// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import classnames from 'classnames';
import type { FC } from 'react';

interface LeftOverflowProps {
  children: string;
  className?: string;
}

/**
 * Single-line text that overflows to the *left*, putting the ellipsis at the
 * start so the tail of the string stays visible. Use it for values whose
 * distinguishing part is at the end — file paths, versioned model names, IDs.
 *
 * `dir="rtl"` moves the ellipsis to the leading edge while `text-left` keeps
 * short values flush left. Bidi only reorders neutral characters at the very
 * edges of the string, so pass values that begin and end in a letter or digit —
 * a trailing `/` or `=` would render on the opposite side.
 */
export const LeftOverflow: FC<LeftOverflowProps> = ({ children, className }) => (
  <span className={classnames('block truncate text-left', className)} dir="rtl">
    {children}
  </span>
);
