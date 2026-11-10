// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Button } from '@nvidia/foundations-react-core';
import type { ModelLineage } from '@studio/hooks/useModelLineage';
import { type ModelDetailParent, ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { DeploymentActions } from '@studio/routes/ModelDetailRoute/DeploymentTab/DeploymentActions';
import { DetailsActions } from '@studio/routes/ModelDetailRoute/DetailsTab/DetailsActions';
import { getFilesetRoute, getWorkspaceCustomizationJobDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

export interface ModelDetailActionsProps {
  selectedTab: ModelDetailTab;
  parent: ModelDetailParent;
  model: ModelEntity;
  adapter: Adapter | null;
  lineage: ModelLineage;
  runEvaluationRoute: string;
}

export const ModelDetailActions: FC<ModelDetailActionsProps> = ({
  selectedTab,
  parent,
  model,
  adapter,
  lineage,
  runEvaluationRoute,
}) => {
  const { job, fileset } = lineage;

  const linkAction = (label: string, to: string) => (
    <Button color="brand" asChild>
      <Link to={to}>{label}</Link>
    </Button>
  );

  switch (selectedTab) {
    case ModelDetailTab.Details:
      // A base model can only be deleted while no provider serves it.
      return parent === 'catalog' ? (
        <DetailsActions model={model} canDelete={!adapter && !model.model_providers?.length} />
      ) : null;
    case ModelDetailTab.Deployment:
      return (
        <DeploymentActions model={model} adapter={adapter} deployment={lineage.baseDeployment} />
      );
    case ModelDetailTab.Evaluations:
      return linkAction('Run Evaluation', runEvaluationRoute);
    case ModelDetailTab.Files:
      return fileset
        ? linkAction(
            'Open Fileset',
            getFilesetRoute(fileset.workspace, `${fileset.workspace}/${fileset.name}`)
          )
        : null;
    case ModelDetailTab.Training:
    case ModelDetailTab.Logs:
      return job.jobName
        ? linkAction(
            'Open Job',
            getWorkspaceCustomizationJobDetailsRoute(job.workspace, job.jobName)
          )
        : null;
    default:
      return null;
  }
};
