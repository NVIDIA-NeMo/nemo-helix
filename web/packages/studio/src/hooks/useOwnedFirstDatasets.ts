// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import { type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { ownedFirst } from '@studio/util/ownedFirst';
import { useMemo } from 'react';

/**
 * Dataset filesets for a picker, newest first, with the signed-in user's own datasets on top.
 *
 * The user's datasets come from their own `created_by` query rather than from reordering the
 * newest `pageSize` rows, so an older dataset of theirs is never pushed out of a workspace
 * that holds more datasets than one page.
 */
export function useOwnedFirstDatasets(workspace: string, pageSize: number) {
  const ownerId = useAuthProfile()?.id;
  const params = {
    page_size: pageSize,
    sort: '-created_at',
    filter: { purpose: 'dataset' },
  } as const;

  const all = useFilesListFilesets(workspace, params);
  const mine = useFilesListFilesets(
    workspace,
    { ...params, filter: { ...params.filter, created_by: ownerId } },
    { query: { enabled: Boolean(ownerId) } }
  );

  const datasets = useMemo<FilesetOutput[] | undefined>(
    () => (all.data ? ownedFirst(mine.data?.data ?? [], all.data.data) : undefined),
    [all.data, mine.data]
  );

  return {
    datasets,
    isPending: all.isPending || (Boolean(ownerId) && mine.isPending),
    isFetching: all.isFetching || mine.isFetching,
    isError: all.isError || mine.isError,
    error: all.error ?? mine.error,
  };
}
