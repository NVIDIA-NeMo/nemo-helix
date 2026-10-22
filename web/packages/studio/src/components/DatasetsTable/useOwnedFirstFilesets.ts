// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import {
  filesListFilesets,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import {
  type FilesetFilter,
  type FilesetOutput,
  type GenericSortField,
} from '@nemo/sdk/generated/platform/schema';
import { keepPreviousData, useQuery } from '@tanstack/react-query';

/** The slice of a list response the table reads. */
export interface DatasetsPage {
  data: FilesetOutput[];
  pagination?: { total_results?: number };
}

/**
 * Where a page of the "mine first, then everyone else" list continues in the second list.
 *
 * The API has no "current user first" sort, so the table is the concatenation of two lists:
 * filesets created by the user, then the rest. A page `[offset, offset + size)` of that
 * concatenation takes `min(size, ownTotal - offset)` rows from the first list; the remaining
 * `count` rows start at index `start` of the second.
 */
export const othersWindow = (ownTotal: number, offset: number, size: number) => {
  const ownOnPage = Math.max(0, Math.min(size, ownTotal - offset));
  return { start: Math.max(0, offset - ownTotal), count: size - ownOnPage };
};

interface UseOwnedFirstFilesetsArgs {
  workspace: string;
  /** Principal id recorded as `created_by` on the user's filesets. The query is off when unset. */
  ownerId: string | undefined;
  /** 1-based page number. */
  page: number;
  pageSize: number;
  sort: GenericSortField | undefined;
  filter: FilesetFilter;
}

export function useOwnedFirstFilesets({
  workspace,
  ownerId,
  page,
  pageSize,
  sort,
  filter,
}: UseOwnedFirstFilesetsArgs) {
  return useQuery<DatasetsPage>({
    // Shares the list-key prefix so `invalidateDatasetCaches(..., ['list'])` refreshes it.
    queryKey: [
      ...getFilesListFilesetsQueryKey(workspace),
      'owned-first',
      ownerId,
      { page, pageSize, sort, filter },
    ],
    enabled: Boolean(ownerId),
    placeholderData: keepPreviousData,
    queryFn: async ({ signal }) => {
      const mine = await filesListFilesets(
        workspace,
        { page, page_size: pageSize, sort, filter: { ...filter, created_by: ownerId } },
        signal
      );
      const ownTotal = mine.pagination?.total_results ?? mine.data.length;
      const { start, count } = othersWindow(ownTotal, (page - 1) * pageSize, pageSize);

      // `$nin` alone would drop filesets with no recorded creator, so match those explicitly.
      const othersFilter = jsonFilter<FilesetFilter>({
        $and: [
          filter,
          { $or: [{ created_by: { $nin: [ownerId] } }, { created_by: { $eq: null } }] },
        ],
      });
      const fetchOthers = (othersPage: number) =>
        filesListFilesets(
          workspace,
          { page: othersPage, page_size: pageSize, sort, filter: othersFilter },
          signal
        );

      // The second list's offset needn't be page-aligned, so a window can straddle two pages.
      const firstPage = Math.floor(start / pageSize) + 1;
      const skip = start % pageSize;
      const first = await fetchOthers(firstPage);
      const rest = skip + count > pageSize ? (await fetchOthers(firstPage + 1)).data : [];
      const others = [...first.data, ...rest].slice(skip, skip + count);

      return {
        data: [...mine.data, ...others],
        pagination: {
          total_results: ownTotal + (first.pagination?.total_results ?? first.data.length),
        },
      };
    },
  });
}
