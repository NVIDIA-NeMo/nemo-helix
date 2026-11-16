// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useEvalsListTasksetRevisions } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Revision, Taskset } from '@nemo/sdk/generated/evals/schema';
import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { requestStatus } from '@studio/routes/evaluation/tasksets/helpers';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, type FC, useMemo } from 'react';

type RevisionRow = Revision & { id: string };

const makeColumns: ComponentProps<typeof StudioDataView<RevisionRow>>['makeColumns'] = ({
  accessor,
}) => [
  accessor('revision', {
    header: 'Revision',
    size: 100,
    enableSorting: false,
    cell: ({ row }) => <Text>r{row.original.revision}</Text>,
  }),
  accessor((row) => row.tags?.join(',') ?? '', {
    id: 'tags',
    header: 'Tags',
    enableSorting: false,
    cell: ({ row }) => {
      const tags = row.original.tags ?? [];
      if (tags.length === 0) return <Text>{EMPTY_FIELD_VALUE}</Text>;
      return (
        <Flex gap="density-xs" wrap="wrap">
          {tags.map((tag) => (
            <Tag key={tag} kind="outline" color="gray" readOnly>
              {tag}
            </Tag>
          ))}
        </Flex>
      );
    },
  }),
  accessor('content_hash', {
    header: 'Content Hash',
    size: 160,
    enableSorting: false,
    cell: ({ row }) => (
      <Text kind="mono/sm" title={row.original.content_hash}>
        {row.original.content_hash.slice(0, 12)}
      </Text>
    ),
  }),
  accessor('created_at', {
    header: 'Published',
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <RelativeTime datetime={row.original.created_at} />,
  }),
];

/** Every published revision of a taskset, newest first (the server's order), with the tags
 *  that point at each and the content hash a pinned reference carries. */
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
  const revisions = useMemo<RevisionRow[]>(
    () => (data?.data ?? []).map((revision) => ({ ...revision, id: String(revision.revision) })),
    [data?.data]
  );

  return (
    <StudioDataView
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      attributes={{
        DataViewRoot: {
          data: revisions,
          totalCount: data?.pagination?.total_results,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(error ?? new Error('Failed to fetch revisions'))}
            />
          ),
        },
      }}
    />
  );
};
