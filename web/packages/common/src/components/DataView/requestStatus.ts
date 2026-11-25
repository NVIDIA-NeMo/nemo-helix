// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { QueryStatus } from '@nemo/common/src/components/DataView/internal/types';

/** DataView request status: an error wins over an in-flight fetch. */
export const requestStatus = (error: unknown, isFetching: boolean): QueryStatus | undefined => {
  if (error) return 'error';
  return isFetching ? 'loading' : undefined;
};
