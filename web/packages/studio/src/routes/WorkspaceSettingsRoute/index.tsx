// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  Button,
  Divider,
  Flex,
  PageHeader,
  Panel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { INFERENCE_PROVIDER_ENABLED, MEMBERS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getSecretsRoute,
  getWorkspaceInferenceProvidersRoute,
  getWorkspaceMembersRoute,
  getWorkspaceSettingsRoute,
} from '@studio/routes/utils';
import { MEMBERS_ROUTE_HEADER_DESCRIPTION } from '@studio/routes/WorkspaceMembersRoute/constants';
import { DeleteWorkspaceModal } from '@studio/routes/WorkspaceSettingsRoute/DeleteWorkspaceModal';
import { EditDescriptionModal } from '@studio/routes/WorkspaceSettingsRoute/EditDescriptionModal';
import { FC, ReactNode, useState } from 'react';
import { useNavigate } from 'react-router';

interface SettingsSectionProps {
  label: ReactNode;
  body: string;
  action: React.ReactNode;
}

const SettingsSection: FC<SettingsSectionProps> = ({ label, body, action }) => (
  <Flex justify="between" align="center" gap="density-2xl">
    <Stack gap="density-xs">
      <Text kind="body/bold/xl">{label}</Text>
      <Text kind="body/regular/md" className="text-subtle">
        {body}
      </Text>
    </Stack>
    {action}
  </Flex>
);

export const WorkspaceSettingsRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const isDefaultWorkspace = workspace === DEFAULT_WORKSPACE;
  const navigate = useNavigate();
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  useBreadcrumbs({
    items: [{ href: getWorkspaceSettingsRoute(workspace), slotLabel: 'Settings' }],
  });

  return (
    <AccessibleTitle title="Settings">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={`${workspace} Settings`}
          slotDescription="Personalize your workspace, manage API secrets, and add inference providers."
        />
        <Panel elevation="high">
          <Stack gap="10">
            <SettingsSection
              label="Description"
              body="Modify your workspace description to reflect its purpose."
              action={
                <Button
                  color="neutral"
                  kind="secondary"
                  onClick={() => setEditModalOpen(true)}
                  className="shrink-0"
                >
                  Edit Description
                </Button>
              }
            />

            {MEMBERS_ENABLED && (
              <>
                <Divider />
                <SettingsSection
                  label={
                    <>
                      Members & Access
                      <FeatureFlagBadge flag="membersEnabled" />
                    </>
                  }
                  body={MEMBERS_ROUTE_HEADER_DESCRIPTION}
                  action={
                    <Button
                      color="neutral"
                      kind="secondary"
                      onClick={() => navigate(getWorkspaceMembersRoute(workspace))}
                      className="shrink-0"
                    >
                      Manage Members
                    </Button>
                  }
                />
              </>
            )}

            <>
              <Divider />
              <SettingsSection
                label="Secrets"
                body="Manage user-defined secrets to securely store API keys to integrate with other providers."
                action={
                  <Button
                    color="neutral"
                    kind="secondary"
                    onClick={() => navigate(getSecretsRoute(workspace))}
                    className="shrink-0"
                  >
                    Manage Secrets
                  </Button>
                }
              />
            </>

            {INFERENCE_PROVIDER_ENABLED && (
              <>
                <Divider />
                <SettingsSection
                  label="Inference Providers"
                  body="Manage inference endpoints (NVIDIA Build, OpenAI, NIMs). Create an API key secret and reference it when adding a provider."
                  action={
                    <Button
                      color="neutral"
                      kind="secondary"
                      onClick={() => navigate(getWorkspaceInferenceProvidersRoute(workspace))}
                      className="shrink-0"
                    >
                      Manage Providers
                    </Button>
                  }
                />
              </>
            )}

            <Divider />
            <SettingsSection
              label="Delete Workspace"
              body={
                isDefaultWorkspace
                  ? 'The default workspace cannot be deleted.'
                  : 'Permanently delete your Workspace and all of its contents from NeMo Studio. Once deleted, it cannot be recovered.'
              }
              action={
                <Button
                  color="danger"
                  disabled={isDefaultWorkspace}
                  onClick={() => setDeleteModalOpen(true)}
                  className="shrink-0"
                >
                  Delete Workspace
                </Button>
              }
            />
          </Stack>
        </Panel>
      </Stack>

      <EditDescriptionModal
        workspace={workspace}
        open={editModalOpen}
        onClose={() => setEditModalOpen(false)}
      />
      <DeleteWorkspaceModal
        workspace={workspace}
        open={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
      />
    </AccessibleTitle>
  );
};
