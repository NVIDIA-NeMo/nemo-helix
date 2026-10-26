// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Grid } from '@nvidia/foundations-react-core';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import type { FilesetRef } from '@studio/hooks/useModelLineage';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getFilesetRoute, getWorkspaceDeploymentDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

/** What the deployment lineage card shows for each deployment state. */
const getDeploymentCard = (
  state: DeploymentIndicatorState,
  isAdapter: boolean
): { value: string; note?: string; to?: string } => {
  switch (state.kind) {
    case 'served': {
      if (state.deploymentRef) {
        const { workspace, name } = getPartsFromReference(state.deploymentRef);
        return { value: name, to: getWorkspaceDeploymentDetailsRoute(workspace, name) };
      }
      return {
        value: state.providerRef ? getPartsFromReference(state.providerRef).name : 'External',
        note: 'Served by an external provider',
      };
    }
    case 'not-serving': {
      const { workspace, name, status_message: statusMessage } = state.deployment;
      return {
        value: name,
        to: getWorkspaceDeploymentDetailsRoute(workspace, name),
        note: statusMessage || 'Not serving yet',
      };
    }
    case 'adapter-not-loaded':
      return { value: 'Not loaded', note: "The model's deployment hasn't loaded this adapter" };
    case 'not-deployed':
      return isAdapter
        ? { value: 'Base not deployed', note: 'Deploy the base model to serve this adapter' }
        : { value: 'Not deployed', note: 'Deploy it to chat and serve inference' };
    default:
      return { value: 'Unknown', note: "Couldn't check where this is served" };
  }
};

export interface LineageSectionProps {
  /**
   * Base model card: the model an adapter belongs to, or a checkpoint's `base_model`. This and
   * `job` are omitted for base models, which have neither.
   */
  base?: { name: string; to?: string; note: string };
  job?: {
    jobName: string | undefined;
    isLoading: boolean;
    status?: HelixJobStatus;
    createdAt?: string;
    to: string;
  };
  fileset: FilesetRef | null;
  deployment: DeploymentIndicatorState;
  /** Where the deployment card leads; defaults to the serving deployment's own page. */
  deploymentTo?: string;
  isAdapter: boolean;
}

/** Where the model came from and where it runs, each card opening the full detail. */
export const LineageSection: FC<LineageSectionProps> = ({
  base,
  job,
  fileset,
  deployment,
  deploymentTo,
  isAdapter,
}) => {
  const deploymentCard = getDeploymentCard(deployment, isAdapter);

  return (
    <Grid colMinWidth="220px" gap="density-lg">
      {base && <LineageCard label="Base Model" value={base.name} to={base.to} note={base.note} />}

      {job && (
        <LineageCard
          label="Training Job"
          loading={job.isLoading}
          value={job.jobName ?? 'Not found'}
          to={job.jobName ? job.to : undefined}
          badge={job.status && <StatusBadge status={job.status} />}
          note={
            job.jobName
              ? job.createdAt && formatAbsoluteTimestamp(job.createdAt)
              : 'No job lists this model as its output'
          }
        />
      )}

      <LineageCard
        label="Files"
        value={fileset?.name ?? 'None'}
        to={
          fileset
            ? getFilesetRoute(fileset.workspace, `${fileset.workspace}/${fileset.name}`)
            : undefined
        }
        note={fileset ? `Fileset in ${fileset.workspace}` : 'No fileset'}
      />

      <LineageCard
        label="Deployment"
        loading={deployment.kind === 'loading'}
        value={deploymentCard.value}
        to={deploymentTo ?? deploymentCard.to}
        badge={<DeploymentStatusBadge state={deployment} isAdapter={isAdapter} />}
        note={deploymentCard.note}
      />
    </Grid>
  );
};
