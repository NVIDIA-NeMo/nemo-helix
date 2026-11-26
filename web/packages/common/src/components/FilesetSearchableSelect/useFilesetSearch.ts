// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useFilesetSearchCore,
  type UseFilesetSearchCoreOptions,
  type UseFilesetSearchResult,
} from '@nemo/common/src/api/filesets/useFilesetSearchCore';
import { filesListFilesets } from '@nemo/sdk/generated/platform/files';

export type { UseFilesetSearchResult } from '@nemo/common/src/api/filesets/useFilesetSearchCore';

export type UseFilesetSearchOptions = Omit<UseFilesetSearchCoreOptions, 'listFilesets'>;

/** `useFilesetSearchCore` on Studio's SDK client. */
export const useFilesetSearch = (options: UseFilesetSearchOptions): UseFilesetSearchResult =>
  useFilesetSearchCore({ ...options, listFilesets: filesListFilesets });
