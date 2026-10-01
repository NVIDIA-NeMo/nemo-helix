// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
import {
  customizationListJobTemplates,
  getCustomizationListJobTemplatesQueryKey,
} from '@nemo/sdk/generated/customizer/customization-job-templates';
import type { CustomizationJobTemplate } from '@nemo/sdk/generated/customizer/schema';
import { useQuery } from '@tanstack/react-query';

/**
 * Every saved template in the workspace.
 *
 * Drained rather than paged: the start page offers these for selection, and the list
 * endpoint defaults to 20 per page, so a 21st template would be unreachable — not
 * selectable and not deletable.
 *
 * Keyed with the generated list key, so creating or deleting one invalidates this too.
 */
export const useSavedTemplates = (workspace: string) =>
  useQuery({
    queryKey: getCustomizationListJobTemplatesQueryKey(workspace),
    queryFn: () =>
      fetchAllPages<CustomizationJobTemplate>((page, pageSize) =>
        customizationListJobTemplates(workspace, { page, page_size: pageSize })
      ),
  });
