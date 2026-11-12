// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { StudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { Revision } from '@nemo/sdk/generated/evals/schema';
import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { requestStatus } from '@studio/routes/evaluation/tasksets/helpers';
import { type ComponentProps, type FC, useMemo } from 'react';

type RevisionRow = Revision & { id: string };

interface RevisionTableProps {
  dataViewState: StudioDataViewState;
  /** One page of revisions, newest first (the server's order). */
  revisions?: Revision[];
  totalCount?: number;
  isFetching: boolean;
  error: Error | null;
}

/** Published revisions of a taskset or task, with the tags that point at each and the content
 *  hash a pinned reference carries. */
export const RevisionTable: FC<RevisionTableProps> = ({
  dataViewState,
  revisions: page,
  totalCount,
  isFetching,
  error,
}) => {
  const revisions = useMemo<RevisionRow[]>(
    () => (page ?? []).map((revision) => ({ ...revision, id: String(revision.revision) })),
    [page]
  );

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

  return (
    <StudioDataView
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      attributes={{
        DataViewRoot: {
          data: revisions,
          totalCount,
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
