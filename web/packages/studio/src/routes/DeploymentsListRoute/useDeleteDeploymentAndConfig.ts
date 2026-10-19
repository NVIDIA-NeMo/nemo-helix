/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import {
  filesDeleteFileset,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import {
  getModelsListDeploymentConfigsQueryKey,
  modelsDeleteAllDeploymentConfigVersions,
  modelsGetLatestDeploymentConfig,
} from '@nemo/sdk/generated/platform/model-deployment-configs';
import {
  getModelsGetLatestDeploymentQueryKey,
  getModelsListDeploymentsQueryKey,
  modelsDeleteAllDeploymentVersions,
  modelsGetLatestDeployment,
} from '@nemo/sdk/generated/platform/model-deployments';
import {
  getModelsListModelsQueryKey,
  modelsDeleteModel,
  modelsGetModel,
} from '@nemo/sdk/generated/platform/models';
import {
  type ModelDeployment,
  type ModelDeploymentConfig,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import {
  HUGGING_FACE_DEPLOYMENT_SOURCE_FIELD,
  HUGGING_FACE_DEPLOYMENT_SOURCE_VALUE,
  huggingFaceSourceFilesetName,
} from '@studio/routes/DeploymentsListRoute/huggingFaceDeploymentArtifacts';
import { type QueryClient, useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

const POLL_INTERVAL_MS = 2000;
const MAX_WAIT_MS = 120_000;

function delay(ms: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * The delete endpoint marks the deployment DELETING before backend teardown finishes.
 * Wait only for that visible state so callers can close their UI promptly.
 */
async function waitForDeploymentDeleteStarted(
  workspace: string,
  deploymentName: string
): Promise<void> {
  const deadline = Date.now() + MAX_WAIT_MS;
  while (Date.now() < deadline) {
    try {
      const latest = await modelsGetLatestDeployment(workspace, deploymentName);
      if (
        latest.status === ModelDeploymentStatus.DELETING ||
        latest.status === ModelDeploymentStatus.DELETED
      ) {
        return;
      }
    } catch {
      // 404 / network: treat as released so the modal does not hang.
      return;
    }
    await delay(POLL_INTERVAL_MS);
  }
  throw new Error('Timed out waiting for deployment to start deleting.');
}

/**
 * Config delete returns 409 while any deployment still references the config and is not DELETED.
 * After starting deployment delete, poll until the deployment is DELETED (or gone) before deleting the config.
 */
async function waitForDeploymentReleasedFromConfig(
  workspace: string,
  deploymentName: string
): Promise<void> {
  const deadline = Date.now() + MAX_WAIT_MS;
  while (Date.now() < deadline) {
    try {
      const latest = await modelsGetLatestDeployment(workspace, deploymentName);
      if (latest.status === ModelDeploymentStatus.DELETED) {
        return;
      }
    } catch {
      // 404 / network: treat as released so we can attempt config delete
      return;
    }
    await delay(POLL_INTERVAL_MS);
  }
  throw new Error('Timed out waiting for deployment to finish deleting.');
}

type HuggingFaceCleanup = { modelName: string; filesetName: string };
type ModelRef = { workspace: string; name: string };

function getDeploymentConfigModelRef(config: ModelDeploymentConfig): ModelRef | null {
  const modelEntityId = config.model_entity_id?.trim();
  if (modelEntityId) {
    const parsed = getPartsFromReference(modelEntityId);
    if (parsed.workspace && parsed.name) {
      return { workspace: parsed.workspace, name: parsed.name };
    }
  }

  const modelName = config.model_spec?.model_name?.trim();
  const modelNamespace = config.model_spec?.model_namespace?.trim();
  if (!modelNamespace || !modelName) return null;

  return {
    workspace: modelNamespace,
    name: modelName,
  };
}

function isStudioHuggingFaceSourceModel(model: ModelEntity): boolean {
  return (
    model.custom_fields?.[HUGGING_FACE_DEPLOYMENT_SOURCE_FIELD] ===
    HUGGING_FACE_DEPLOYMENT_SOURCE_VALUE
  );
}

async function readHuggingFaceCleanupPlan(
  workspace: string,
  configName: string,
  deploymentName: string
): Promise<HuggingFaceCleanup | null> {
  try {
    const cfg = await modelsGetLatestDeploymentConfig(workspace, configName);
    const modelRef = getDeploymentConfigModelRef(cfg);
    if (!modelRef || modelRef.workspace !== workspace) return null;

    const model = await modelsGetModel(modelRef.workspace, modelRef.name);
    if (!isStudioHuggingFaceSourceModel(model)) return null;

    return {
      modelName: modelRef.name,
      filesetName: huggingFaceSourceFilesetName(deploymentName),
    };
  } catch {
    return null;
  }
}

/** Best-effort: deployment + config are already removed; avoid failing the whole flow. */
async function deleteHuggingFaceModelAndFileset(
  workspace: string,
  { modelName, filesetName }: HuggingFaceCleanup
): Promise<void> {
  try {
    await modelsDeleteModel(workspace, modelName);
  } catch {
    /* model may already be gone */
  }
  try {
    await filesDeleteFileset(workspace, filesetName);
  } catch {
    /* fileset may already be gone */
  }
}

async function deleteRelatedResourcesAfterDeploymentReleased(
  workspace: string,
  deployment: ModelDeployment
): Promise<void> {
  const configName = deployment.config?.trim();
  let hfCleanup: HuggingFaceCleanup | null = null;

  await waitForDeploymentReleasedFromConfig(workspace, deployment.name);

  if (configName) {
    hfCleanup = await readHuggingFaceCleanupPlan(workspace, configName, deployment.name);
    await modelsDeleteAllDeploymentConfigVersions(workspace, configName);
  }

  if (hfCleanup) {
    await deleteHuggingFaceModelAndFileset(workspace, hfCleanup);
  }
}

function invalidateDeploymentDeleteQueries(
  queryClient: QueryClient,
  workspace: string,
  deploymentName: string
) {
  queryClient.invalidateQueries({ queryKey: getModelsListDeploymentsQueryKey(workspace) });
  queryClient.invalidateQueries({
    queryKey: getModelsListDeploymentConfigsQueryKey(workspace),
  });
  queryClient.invalidateQueries({
    queryKey: getModelsGetLatestDeploymentQueryKey(workspace, deploymentName),
  });
  queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) });
  queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) });
}

export function useDeleteDeploymentAndConfig(workspace: string) {
  const queryClient = useQueryClient();
  const toast = useToast();

  const mutation = useMutation({
    mutationFn: async (deployment: ModelDeployment) => {
      await modelsDeleteAllDeploymentVersions(workspace, deployment.name);
      await waitForDeploymentDeleteStarted(workspace, deployment.name);

      void deleteRelatedResourcesAfterDeploymentReleased(workspace, deployment)
        .catch((error: unknown) => {
          logger.error('Failed to delete deployment related resources', error);
          toast.error('Failed to delete related deployment resources. Please try again later.');
        })
        .finally(() => {
          invalidateDeploymentDeleteQueries(queryClient, workspace, deployment.name);
        });
    },
    onSettled: (_data, _err, deployment) => {
      invalidateDeploymentDeleteQueries(queryClient, workspace, deployment.name);
    },
  });

  const deleteDeploymentAndConfig = useCallback(
    (deployment: ModelDeployment) => mutation.mutateAsync(deployment),
    [mutation]
  );

  return {
    deleteDeploymentAndConfig,
    isDeleting: mutation.isPending,
  };
}
