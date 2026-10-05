// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextArea } from '@nvidia/foundations-react-core';
import { FC } from 'react';

/** Dataset text shown back verbatim, read-only. `rows` fixes the height at that
 *  many lines; omit it to grow with the content. */
export const PreviewBox: FC<{ value: string; label: string; rows?: number }> = ({
  value,
  label,
  rows,
}) => (
  <TextArea
    readOnly
    size="small"
    value={value}
    aria-label={label}
    className="font-mono text-xs"
    resizeable={rows ? undefined : 'auto'}
    attributes={{ TextAreaElement: { rows } }}
  />
);
