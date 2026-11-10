// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useEvalsListTaskRevisions } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import { RevisionTable } from '@studio/routes/evaluation/tasksets/RevisionTable';
import { keepPreviousData } from '@tanstack/react-query';
import type { FC } from 'react';

/** Every published revision of a task. */
export const RevisionHistory: FC<{ task: Task }> = ({ task }) => {
  const dataViewState = useStudioDataViewState({ defaultSort: [] });
  const { data, isFetching, error } = useEvalsListTaskRevisions(
    task.workspace,
    task.name,
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
