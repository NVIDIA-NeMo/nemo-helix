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
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { ModelDeployment } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { DeploymentsDataView } from '@studio/components/dataViews/DeploymentsDataView';
import { DocumentationButton } from '@studio/components/DocumentationButton';
import { LINK_DOCS_DEPLOYMENTS } from '@studio/constants/links';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { DeploymentDetailsSidePanel } from '@studio/routes/DeploymentsListRoute/DeploymentDetailsSidePanel';
import { useDeleteDeploymentAndConfig } from '@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig';
import {
  DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS,
  getWorkspaceDeploymentDetailsRoute,
  getWorkspaceDeploymentsRoute,
  getWorkspaceNewDeploymentRoute,
} from '@studio/routes/utils';
import { FC, useCallback, useState } from 'react';
import { Navigate, useNavigate, useParams } from 'react-router';

export const DeploymentsListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { deploymentName: deploymentNameParam, deploymentPanelView } = useParams<{
    deploymentName?: string;
    deploymentPanelView?: string;
  }>();

  // `useParams()` already returns decoded values; route helpers handle encoding.
  const deploymentNameFromPath = deploymentNameParam ?? '';

  const [deploymentToDelete, setDeploymentToDelete] = useState<ModelDeployment | null>(null);

  const detailsPanelOpen =
    Boolean(deploymentNameFromPath) &&
    deploymentPanelView === DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS;

  const { deleteDeploymentAndConfig } = useDeleteDeploymentAndConfig(workspace);

  const handleCloseDetailsPanel = useCallback(() => {
    navigate(getWorkspaceDeploymentsRoute(workspace), { replace: true, flushSync: true });
  }, [navigate, workspace]);

  const handleDeleteDeployment = useCallback(async () => {
    if (!deploymentToDelete) return false;
    try {
      await deleteDeploymentAndConfig(deploymentToDelete);
      if (detailsPanelOpen && deploymentToDelete.name === deploymentNameFromPath) {
        navigate(getWorkspaceDeploymentsRoute(workspace), { replace: true, flushSync: true });
      }
      return true;
    } catch {
      return false;
    }
  }, [
    deleteDeploymentAndConfig,
    deploymentNameFromPath,
    deploymentToDelete,
    detailsPanelOpen,
    navigate,
    workspace,
  ]);

  const handleModalClose = useCallback(() => setDeploymentToDelete(null), []);

  const goToCreate = useCallback(() => {
    navigate(getWorkspaceNewDeploymentRoute(workspace));
  }, [navigate, workspace]);

  useBreadcrumbs({
    items: [
      {
        href: getWorkspaceDeploymentsRoute(workspace),
        slotLabel: 'Deployments',
      },
    ],
  });

  const docsButton = <DocumentationButton href={LINK_DOCS_DEPLOYMENTS} />;
  const createDeploymentButton = (
    <Button color="brand" onClick={goToCreate}>
      Create Deployment
    </Button>
  );

  /** Only `details` is supported; normalize unknown segments to avoid broken URLs. */
  if (
    deploymentNameFromPath &&
    deploymentPanelView &&
    deploymentPanelView !== DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS
  ) {
    return (
      <Navigate
        replace
        to={getWorkspaceDeploymentDetailsRoute(
          workspace,
          deploymentNameFromPath,
          DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS
        )}
      />
    );
  }

  return (
    <AccessibleTitle title="Deployments">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Deployments"
          slotDescription="Manage NIM deployments and their configurations."
          slotActions={
            <Flex gap="2">
              {docsButton}
              {createDeploymentButton}
            </Flex>
          }
        />
        <DeploymentsDataView
          workspace={workspace}
          onCreate={goToCreate}
          onDeploymentRowClick={(row) =>
            navigate(
              getWorkspaceDeploymentDetailsRoute(
                workspace,
                row.name,
                DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS
              ),
              { replace: true }
            )
          }
          onRequestDeleteDeployment={setDeploymentToDelete}
          attributes={{
            Stack: {
              className: 'h-full',
            },
          }}
        />
      </Stack>
      <DeploymentDetailsSidePanel
        open={detailsPanelOpen}
        deploymentName={deploymentNameFromPath}
        onClose={handleCloseDetailsPanel}
        onRequestDelete={setDeploymentToDelete}
      />
      {deploymentToDelete ? (
        <DeleteConfirmationModal
          open
          simpleConfirm
          onDelete={handleDeleteDeployment}
          title={`Delete deployment: ${deploymentToDelete.name}`}
          confirmationText={deploymentToDelete.name}
          successText="Deployment deletion started."
          errorText="Failed to start deleting the deployment. Please try again later."
          onClose={handleModalClose}
        />
      ) : null}
    </AccessibleTitle>
  );
};
