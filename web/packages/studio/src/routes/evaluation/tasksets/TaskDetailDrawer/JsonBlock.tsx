// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export const JsonBlock: FC<{ value: unknown }> = ({ value }) => (
  <CodeSnippet kind="block" language="json" value={JSON.stringify(value, null, 2)} />
);
