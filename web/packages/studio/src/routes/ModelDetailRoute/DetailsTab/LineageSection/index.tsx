// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { Grid } from '@nvidia/foundations-react-core';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import type { FilesetRef } from '@studio/hooks/useModelLineage';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getFilesetRoute, getWorkspaceDeploymentDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

/** What the deployment lineage card shows for each deployment state. */
const getDeploymentCard = (
  state: DeploymentIndicatorState
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
    case 'adapter-not-loaded':
      return { value: 'Not loaded', note: "The model's deployment hasn't loaded this adapter" };
    case 'not-deployed':
      return { value: 'Not deployed', note: 'Deploy it to chat and serve inference' };
    default:
      return { value: 'Unknown', note: "Couldn't check where this is served" };
  }
};

export interface LineageSectionProps {
  /**
   * Base model card: the model an adapter belongs to, or a checkpoint's `base_model`. This and
   * `job` are omitted for base models, which have neither.
   */
  base?: { name: string | undefined; to?: string; note: string };
  job?: { jobName: string | undefined; workspace: string; isLoading: boolean; to: string };
  fileset: FilesetRef | null;
  deployment: DeploymentIndicatorState;
  isAdapter: boolean;
}

/** Where the model came from and where it runs, each card opening the full detail. */
export const LineageSection: FC<LineageSectionProps> = ({
  base,
  job,
  fileset,
  deployment,
  isAdapter,
}) => {
  const { job: jobRecord, isLoading: isJobRecordLoading } = useCustomizationJob(
    job?.workspace ?? '',
    job?.jobName ?? '',
    { enabled: Boolean(job?.jobName) }
  );
  const deploymentCard = getDeploymentCard(deployment);

  return (
    <Grid asChild colMinWidth="220px" gap="density-lg" aria-label="Lineage">
      <section>
        {base && (
          <LineageCard
            label="Base Model"
            value={base.name ?? 'Unknown'}
            to={base.name ? base.to : undefined}
            note={base.note}
          />
        )}

        {job && (
          <LineageCard
            label="Training Job"
            loading={job.isLoading || (Boolean(job.jobName) && isJobRecordLoading)}
            value={job.jobName ?? 'Not found'}
            to={job.jobName ? job.to : undefined}
            badge={jobRecord?.status && <StatusBadge status={jobRecord.status} />}
            note={
              job.jobName
                ? jobRecord?.created_at && formatAbsoluteTimestamp(jobRecord.created_at)
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
          to={deploymentCard.to}
          badge={<DeploymentStatusBadge state={deployment} isAdapter={isAdapter} />}
          note={deploymentCard.note}
        />
      </section>
    </Grid>
  );
};
