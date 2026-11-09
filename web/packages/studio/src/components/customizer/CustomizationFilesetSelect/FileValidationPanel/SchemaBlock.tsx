// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export interface SchemaBlockProps {
  /**
   * TypeScript-shaped string describing the inferred row schema (with nested
   * objects/arrays expanded). Empty string when nothing was detected — the
   * block hides itself in that case.
   */
  schemaShape: string;
}

export const SchemaBlock: FC<SchemaBlockProps> = ({ schemaShape }) => {
  if (!schemaShape) return null;
  return (
    <Stack gap="density-sm">
      <Text kind="label/bold/sm">Schema</Text>
      <CodeSnippet kind="block" language="typescript" value={schemaShape} />
    </Stack>
  );
};
