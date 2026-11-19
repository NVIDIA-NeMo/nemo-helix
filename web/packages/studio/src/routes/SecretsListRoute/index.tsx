/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { SecretsDataView } from '@studio/components/dataViews/SecretsDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { getSecretsRoute } from '@studio/routes/utils';
import { type FC, useState } from 'react';

export const SecretsListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  useBreadcrumbs({ items: [{ href: getSecretsRoute(workspace), slotLabel: 'Secrets' }] });

  return (
    <AccessibleTitle title="Secrets">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Secrets"
          slotDescription="Manage user-defined secrets to securely store API keys to integrate with other providers."
          slotActions={
            <Button color="brand" onClick={() => setIsCreateModalOpen(true)}>
              Create Secret
            </Button>
          }
        />
        <SecretsDataView
          workspace={workspace}
          onCreate={() => setIsCreateModalOpen(true)}
          attributes={{
            Stack: {
              className: 'flex-1 min-h-0',
            },
          }}
        />
      </Stack>
      <CreateSecretModal
        workspace={workspace}
        open={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
      />
    </AccessibleTitle>
  );
};
