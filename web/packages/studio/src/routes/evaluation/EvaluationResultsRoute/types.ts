// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PaginationQueryState } from '@nemo/common/src/utils/useQueryFromSearchParams';
import type { DatetimeFilter, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

export interface EvaluationResultsFilterFields {
  name?: string;
  status?: HelixJobStatus[];
  created_at?: DatetimeFilter;
  updated_at?: DatetimeFilter;
}

export type EvaluationResultsFilterState = PaginationQueryState & {
  filter?: EvaluationResultsFilterFields;
};
