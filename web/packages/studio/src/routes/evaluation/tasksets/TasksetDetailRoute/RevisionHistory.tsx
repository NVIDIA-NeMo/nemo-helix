// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useEvalsListTasksetRevisions } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Taskset } from '@nemo/sdk/generated/evals/schema';
import { RevisionTable } from '@studio/routes/evaluation/tasksets/RevisionTable';
import { keepPreviousData } from '@tanstack/react-query';
import type { FC } from 'react';

/** Every published revision of a taskset. */
export const RevisionHistory: FC<{ taskset: Taskset }> = ({ taskset }) => {
  const dataViewState = useStudioDataViewState({ defaultSort: [] });
  const { data, isFetching, error } = useEvalsListTasksetRevisions(
    taskset.workspace,
    taskset.name,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
    },
    { query: { placeholderData: keepPreviousData } }
  );
  return (
    <RevisionTable
      dataViewState={dataViewState}
      revisions={data?.data}
      totalCount={data?.pagination?.total_results}
      isFetching={isFetching}
      error={error}
    />
  );
};
