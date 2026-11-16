// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { Text } from '@nvidia/foundations-react-core';
import { formatValue } from '@studio/routes/evaluation/tasksets/helpers';
import type { FC } from 'react';

interface ConfigGridProps {
  section: Record<string, unknown>;
  /** Shown when the section has no entries. */
  empty: string;
}

export const ConfigGrid: FC<ConfigGridProps> = ({ section, empty }) => {
  const entries = Object.entries(section);
  if (entries.length === 0) {
    return <Text className="text-subtle">{empty}</Text>;
  }
  return (
    <KeyValueGrid
      items={entries.map(([key, value]) => ({ key, label: key, value: formatValue(value) }))}
    />
  );
};
