// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SuiteRow } from '@agent-hardener/components/hitlTypes';
import { CSV_FIELDS } from '@agent-hardener/routes/AgentHardenerManifestDetailRoute/constants';
import { toCsv } from '@nemo/common';

/** The suite as a requests.csv. It is re-uploaded as a benign suite, so cells are written verbatim. */
export const toRequestsCsv = (rows: SuiteRow[]): string =>
  toCsv(rows, { columns: CSV_FIELDS, formulaSafe: false });
