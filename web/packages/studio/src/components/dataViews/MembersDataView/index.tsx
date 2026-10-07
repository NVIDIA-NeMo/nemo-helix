/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { WorkspaceMember } from '@nemo/sdk/generated/platform/schema';
import { type DropdownEntry, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceMembers } from '@studio/api/entity-store/useWorkspaceMembers';
import { Loading } from '@studio/components/Layouts/Loading';
import { Lock } from 'lucide-react';
import { ComponentProps, FC, useCallback, useMemo } from 'react';

export interface MembersDataViewProps {
  workspace: string;
  onAddMember: () => void;
  onEditMember: (member: WorkspaceMember) => void;
  onRemoveMember: (member: WorkspaceMember) => void;
}

type WorkspaceMemberWithId = WorkspaceMember & { id: string };

const grantedAtMs = (m: WorkspaceMember) => (m.granted_at ? new Date(m.granted_at).getTime() : 0);

export const MembersDataView: FC<MembersDataViewProps> = ({
  workspace,
  onAddMember,
  onEditMember,
  onRemoveMember,
}) => {
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'principal', desc: false }],
  });

  const { data, isLoading, error, isFetching, isForbidden } = useWorkspaceMembers(workspace);

  const membersWithId = useMemo<WorkspaceMemberWithId[]>(
    () => (data?.data ?? []).map((m) => ({ ...m, id: m.principal })),
    [data?.data]
  );

  const sortedMembers = useMemo(() => {
    const sort = dataViewState.sorting.state[0];
    if (!sort) {
      return membersWithId;
    }
    return [...membersWithId].sort((a, b) => {
      if (sort.id === 'principal') {
        const cmp = a.principal.localeCompare(b.principal);
        return sort.desc ? -cmp : cmp;
      }
      if (sort.id === 'granted_at') {
        const cmp = grantedAtMs(a) - grantedAtMs(b);
        return sort.desc ? -cmp : cmp;
      }
      return 0;
    });
  }, [membersWithId, dataViewState.sorting.state]);

  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageData = useMemo(() => {
    const start = pageIndex * pageSize;
    return sortedMembers.slice(start, start + pageSize);
  }, [sortedMembers, pageIndex, pageSize]);

  const makeColumns: ComponentProps<typeof StudioDataView<WorkspaceMemberWithId>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowActionsColumn }) => [
        accessor('principal', {
          id: 'principal',
          header: 'Member',
          enableSorting: true,
          cell({ row }) {
            const principal = row.original.principal;
            return (
              <Text kind="body/bold/md" className="truncate" title={principal}>
                {principal}
              </Text>
            );
          },
        }),
        accessor('roles', {
          id: 'roles',
          header: 'Roles',
          enableSorting: false,
          cell({ row }) {
            return row.original.roles.join(', ');
          },
        }),
        accessor((row) => row.granted_at ?? '', {
          id: 'granted_at',
          header: 'Granted',
          enableSorting: true,
          size: 220,
          cell({ row }) {
            const { granted_at: grantedAt, granted_by: grantedBy } = row.original;
            if (!grantedAt) {
              return <Text>—</Text>;
            }
            return (
              <Text>
                <RelativeTime datetime={grantedAt} />
                {grantedBy ? ` · by ${grantedBy}` : null}
              </Text>
            );
          },
        }),
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          rowActions: (member: WorkspaceMemberWithId): DropdownEntry[] => [
            {
              children: 'Edit Role',
              onSelect: () => onEditMember(member),
            },
            {
              children: 'Remove',
              danger: true,
              onSelect: () => onRemoveMember(member),
            },
          ],
        }),
      ],
      [onEditMember, onRemoveMember]
    );

  if (isLoading) {
    return <Loading description="Loading Members…" />;
  }

  // A 403 isn't a failure the user can retry their way out of, so skip the generic error panel
  // (and its "Refresh Page" call to action) and tell them what to do instead.
  if (isForbidden) {
    return (
      <ErrorMessage
        header="You don't have permission to view members"
        message="Managing members and access requires workspace admin permissions. Contact your workspace administrator to request access."
        slotMedia={<Lock className="size-16 stroke-2" />}
        slotFooter={<></>}
      />
    );
  }

  return (
    <StudioDataView<WorkspaceMemberWithId>
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      attributes={{
        DataViewRoot: {
          data: pageData,
          totalCount: sortedMembers.length,
          requestStatus: isFetching ? 'loading' : error ? 'error' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <EntityEmptyState entity="members" variant="first-use" onCreate={onAddMember} />
          ),
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(error ?? new Error('Failed to load members.'))}
            />
          ),
        },
      }}
    />
  );
};
