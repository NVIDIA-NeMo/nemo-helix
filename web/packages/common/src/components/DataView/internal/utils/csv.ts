// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IntentionalAny } from '@nemo/common/src/components/DataView/internal/types';
import { toCsvCell } from '@nemo/common/src/utils/download';
import { childrenToText } from '@nvidia/foundations-react-core/lib';
import type { Column, Row } from '@tanstack/react-table';
import type { ReactNode } from 'react';

export function generateCSV(
  rows: Row<IntentionalAny>[],
  columns: Column<IntentionalAny>[]
): string {
  const dataColumns = columns.filter((col) => !col.columnDef.meta?._isPrebuiltColumn);
  const headers = dataColumns.map((col) =>
    toCsvCell(childrenToText(col.columnDef.header as ReactNode))
  );
  const dataRows = rows.map((row) => dataColumns.map((col) => toCsvCell(row.getValue(col.id))));
  return [headers, ...dataRows].map((row) => row.join(',')).join('\n');
}
