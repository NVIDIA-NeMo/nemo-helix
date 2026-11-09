// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useModelSearchCore,
  type UseModelSearchCoreOptions,
  type UseModelSearchResult,
} from '@nemo/common/src/api/models/useModelSearchCore';
import { modelsListModels } from '@nemo/sdk/generated/platform/models';

export {
  MODEL_SEARCH_PAGE_SIZE,
  type ModelSearchFilter,
  type ModelSearchProps,
  type UseModelSearchResult,
} from '@nemo/common/src/api/models/useModelSearchCore';

export type UseModelSearchOptions = Omit<UseModelSearchCoreOptions, 'listModels'>;

/** `useModelSearchCore` on Studio's SDK client. */
export const useModelSearch = (options: UseModelSearchOptions): UseModelSearchResult =>
  useModelSearchCore({ ...options, listModels: modelsListModels });
