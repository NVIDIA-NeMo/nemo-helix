// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SuiteRow } from '@agent-hardener/components/hitlTypes';
import { CSV_FIELDS } from '@agent-hardener/routes/AgentHardenerManifestDetailRoute/constants';

const escapeCsv = (value: string): string =>
  /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;

export const toRequestsCsv = (rows: SuiteRow[]): string => {
  const body = rows.map((row) =>
    CSV_FIELDS.map((field) => escapeCsv(String(row[field] ?? ''))).join(',')
  );
  return [CSV_FIELDS.join(','), ...body].join('\n');
};
