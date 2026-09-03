// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ImportTraceResult {
  label: string;
  status: 'success' | 'error';
  message?: string;
}

/** How the user chose to get traces into Intake. */
export type ImportMethod = 'skill' | 'files';
