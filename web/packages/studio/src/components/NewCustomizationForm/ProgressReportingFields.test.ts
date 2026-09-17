// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customizationCreateAutomodelJobBodySpecScheduleProgressReportingMinReportIntervalSecondsMin as SCHEMA_MIN } from '@nemo/sdk/generated/customizer/zod/automodel-jobs/customizationCreateAutomodelJob';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * A slider's `min` silently decides which of the schema's values a user can reach. Setting
 * it above the schema's own minimum takes a documented value away with nothing to show for
 * it — here, `0`, which the backend documents as "report on every step the library logs".
 * Nothing else catches that: the tighter bound is still a valid number, so the types, the
 * linter and the rendered form are all satisfied.
 */
describe('min report interval', () => {
  it('lets the user reach every interval the schema accepts', () => {
    const source = readFileSync(
      join(dirname(fileURLToPath(import.meta.url)), 'ProgressReportingFields.tsx'),
      'utf8'
    );
    const min = source.match(/min=\{(-?[\d.]+)\}/)?.[1];

    expect(min, 'no min= found on the interval slider').toBeDefined();
    expect(Number(min)).toBe(SCHEMA_MIN);
  });
});
