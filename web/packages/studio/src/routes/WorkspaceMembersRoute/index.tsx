/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  getEntitiesListWorkspaceMembersQueryKey,
  useEntitiesRemoveWorkspaceMember,
} from '@nemo/sdk/generated/platform/entity-store';
import type { WorkspaceMember } from '@nemo/sdk/generated/platform/schema';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { useWorkspaceMembers } from '@studio/api/entity-store/useWorkspaceMembers';
import { queryClient } from '@studio/api/queryClient';
import { MembersDataView } from '@studio/components/dataViews/MembersDataView';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getWorkspaceMembersRoute, getWorkspaceSettingsRoute } from '@studio/routes/utils';
import { MEMBERS_ROUTE_HEADER_DESCRIPTION } from '@studio/routes/WorkspaceMembersRoute/constants';
import {
  WorkspaceMemberModal,
  type WorkspaceMemberModalMode,
} from '@studio/routes/WorkspaceMembersRoute/WorkspaceMemberModal';
import { FC, useCallback, useState } from 'react';

export const WorkspaceMembersRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<WorkspaceMemberModalMode>('add');
  const [editingMember, setEditingMember] = useState<WorkspaceMember | null>(null);
  const [removingMember, setRemovingMember] = useState<WorkspaceMember | null>(null);

  const { mutateAsync: removeMember } = useEntitiesRemoveWorkspaceMember();
  const { data: membersData, isForbidden } = useWorkspaceMembers(workspace);

  useBreadcrumbs({
    items: [
      { href: getWorkspaceSettingsRoute(workspace), slotLabel: 'Settings' },
      { href: getWorkspaceMembersRoute(workspace), slotLabel: 'Members' },
    ],
  });

  const openAdd = useCallback(() => {
    setEditingMember(null);
    setModalMode('add');
    setModalOpen(true);
  }, []);

  const openEdit = useCallback((member: WorkspaceMember) => {
    setEditingMember(member);
    setModalMode('edit');
    setModalOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setModalOpen(false);
    setEditingMember(null);
  }, []);

  const handleRemove = useCallback(async () => {
    if (!removingMember) {
      return false;
    }
    try {
      await removeMember({ workspace, principalId: removingMember.principal });
      await queryClient.invalidateQueries({
        queryKey: getEntitiesListWorkspaceMembersQueryKey(workspace),
      });
      setRemovingMember(null);
      return true;
    } catch {
      return false;
    }
  }, [removeMember, removingMember, workspace]);

  return (
    <AccessibleTitle title="Workspace members">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <>
              Members & Access
              <FeatureFlagBadge flag="membersEnabled" />
            </>
          }
          slotDescription={MEMBERS_ROUTE_HEADER_DESCRIPTION}
          slotActions={
            // Without read access there is nothing to add members to from here.
            isForbidden ? undefined : (
              <Button color="brand" onClick={openAdd}>
                Add Member
              </Button>
            )
          }
        />

        <MembersDataView
          workspace={workspace}
          onAddMember={openAdd}
          onEditMember={openEdit}
          onRemoveMember={setRemovingMember}
        />
      </Stack>

      <WorkspaceMemberModal
        open={modalOpen}
        onClose={closeModal}
        workspace={workspace}
        mode={modalMode}
        member={editingMember}
        existingMembers={membersData?.data ?? []}
      />

      <DeleteConfirmationModal
        open={Boolean(removingMember)}
        onClose={() => setRemovingMember(null)}
        title="Remove Workspace Member"
        description={`Revoke all roles for ${removingMember?.principal ?? ''} in this workspace.`}
        simpleConfirm
        confirmationText={removingMember?.principal}
        successText="Member removed."
        errorText="Could not remove member."
        onDelete={handleRemove}
      />
    </AccessibleTitle>
  );
};
