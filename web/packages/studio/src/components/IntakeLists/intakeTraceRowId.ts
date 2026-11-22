// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trace } from '@nemo/sdk/generated/platform/schema';

// Loading placeholders have no identity; legacy producers can reuse IDs across sessions.
export const getIntakeTraceRowId = (trace: Trace, index: number): string =>
  trace.session_id === undefined || trace.id === undefined
    ? String(index)
    : JSON.stringify([trace.session_id, trace.id]);
