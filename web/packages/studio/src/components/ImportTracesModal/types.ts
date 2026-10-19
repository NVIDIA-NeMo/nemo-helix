// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ImportTraceResult {
  label: string;
  status: 'success' | 'error';
  message?: string;
  /**
   * A per-record failure listed under a file's outcome rather than an outcome of its own, so
   * counting files does not count every rejected record inside one.
   */
  detail?: boolean;
}

/** How the user chose to get traces into Intake. */
export type ImportMethod = 'skill' | 'files';
