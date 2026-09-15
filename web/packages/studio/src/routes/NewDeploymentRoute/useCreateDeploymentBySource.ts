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

import { getErrorMessage, isNotFoundError } from '@nemo/common/src/api/common/utils';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  filesCreateFileset,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import {
  getModelsListDeploymentConfigsQueryKey,
  modelsCreateDeploymentConfig,
} from '@nemo/sdk/generated/platform/model-deployment-configs';
import {
  getModelsListDeploymentsQueryKey,
  modelsCreateDeployment,
} from '@nemo/sdk/generated/platform/model-deployments';
import {
  getModelsListModelsQueryKey,
  modelsCreateModel,
  modelsGetModel,
} from '@nemo/sdk/generated/platform/models';
import {
  Engine,
  type ContainerExecutorConfig,
  type CreateFilesetRequest,
} from '@nemo/sdk/generated/platform/schema';
import {
  HUGGING_FACE_DEPLOYMENT_SOURCE_FIELD,
  HUGGING_FACE_DEPLOYMENT_SOURCE_VALUE,
  huggingFaceSourceFilesetName,
} from '@studio/routes/DeploymentsListRoute/huggingFaceDeploymentArtifacts';
import {
  additionalEnvsFormToApi,
  configNameFromWizardBaseName,
  deploymentNameFromWizardBaseName,
  WORKSPACE_PICKER_MODEL,
  SOURCE_HF,
  SOURCE_WORKSPACE,
  SOURCE_NGC,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { NO_SECRET_SELECT_VALUE } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';

type ReportStage = (message: string) => void;

/**
 * Image overrides for the engines that accept one.
 *
 * Blank is meaningful: the platform resolves the engine's own default image
 * (model-agnostic for vLLM), so an empty field must be omitted rather than sent
 * as an empty string.
 */
function imageOverrides(
  values: WizardFormValues
): Partial<Pick<ContainerExecutorConfig, 'image_name' | 'image_tag'>> {
  const imageName = values.imageName?.trim();
  const imageTag = values.imageTag?.trim();
  return {
    ...(imageName ? { image_name: imageName } : {}),
    ...(imageTag ? { image_tag: imageTag } : {}),
  };
}

async function createNgcDeployment(
  workspace: string,
  values: WizardFormValues,
  deploymentName: string,
  configName: string,
  reportStage: ReportStage
) {
  const additionalEnvs = additionalEnvsFormToApi(values.additionalEnvs);
  const modelName = values.name.trim();

  reportStage('Creating deployment configuration…');
  await modelsCreateDeploymentConfig(workspace, {
    name: configName,
    // The NGC source deploys a NIM container by definition; it has no engine picker.
    engine: Engine.nim,
    model_spec: {
      model_name: modelName,
      lora_enabled: values.loraEnabled,
    },
    executor_config: {
      gpu: values.gpu,
      image_name: values.imageName!.trim(),
      image_tag: values.imageTag!.trim(),
      disk_size: values.diskSize?.trim() || '50Gi',
      ...(additionalEnvs ? { additional_envs: additionalEnvs } : {}),
    },
  });

  reportStage('Creating deployment…');
  await modelsCreateDeployment(workspace, {
    name: deploymentName,
    config: configName,
  });
}

/**
 * Whether a Model Entity already exists under this name.
 *
 * Only a 404 proves the name is free. Any other failure (network, 5xx, auth) leaves
 * the answer unknown, and guessing "not taken" would let the chain create a fileset
 * that a later 409 strands with no rollback — so those errors propagate instead.
 */
async function isModelNameTaken(workspace: string, name: string): Promise<boolean> {
  try {
    await modelsGetModel(workspace, name);
    return true;
  } catch (error) {
    if (isNotFoundError(error)) return false;
    throw error;
  }
}

async function createHuggingFaceDeployment(
  workspace: string,
  values: WizardFormValues,
  deploymentName: string,
  configName: string,
  reportStage: ReportStage
) {
  const baseName = values.name.trim();
  const filesetName = huggingFaceSourceFilesetName(deploymentName);
  const modelEntityName = baseName;
  const tokenSecret =
    values.hfTokenSecret && values.hfTokenSecret !== NO_SECRET_SELECT_VALUE
      ? values.hfTokenSecret
      : undefined;

  // Checked before anything is created. The name now defaults from the repo id, so
  // redeploying the same model is a normal action and would otherwise fail on the
  // fileset — reporting the wrong resource and leaving that fileset orphaned, since
  // this chain has no rollback.
  reportStage('Checking name availability…');
  if (await isModelNameTaken(workspace, modelEntityName)) {
    throw new Error(
      `A model named "${modelEntityName}" already exists in ${workspace}. ` +
        'Choose a different name, or delete the existing deployment first.'
    );
  }

  const storage: CreateFilesetRequest['storage'] = {
    type: 'huggingface',
    repo_id: values.repoId!.trim(),
    repo_type: 'model',
    ...(tokenSecret ? { token_secret: tokenSecret } : {}),
  };

  reportStage('Creating Hugging Face fileset…');
  await filesCreateFileset(workspace, {
    name: filesetName,
    storage,
  });

  reportStage('Registering model…');
  await modelsCreateModel(workspace, {
    name: modelEntityName,
    fileset: `${workspace}/${filesetName}`,
    custom_fields: {
      [HUGGING_FACE_DEPLOYMENT_SOURCE_FIELD]: HUGGING_FACE_DEPLOYMENT_SOURCE_VALUE,
    },
  });

  reportStage('Creating deployment configuration…');
  await modelsCreateDeploymentConfig(workspace, {
    name: configName,
    engine: values.engine,
    model_spec: {
      model_namespace: workspace,
      model_name: modelEntityName,
      lora_enabled: values.loraEnabled,
    },
    executor_config: {
      gpu: values.gpu,
      ...imageOverrides(values),
    },
    model_entity_id: `${workspace}/${modelEntityName}`,
  });

  reportStage('Creating deployment…');
  await modelsCreateDeployment(workspace, {
    name: deploymentName,
    config: configName,
  });
}

async function createWorkspaceDeployment(
  workspace: string,
  values: WizardFormValues,
  deploymentName: string,
  configName: string,
  reportStage: ReportStage
) {
  let modelNamespace: string;
  let modelName: string;

  if (values.workspacePickerType === WORKSPACE_PICKER_MODEL) {
    if (!values.modelRef) {
      throw new Error('Select a model');
    }
    const parsed = getPartsFromReference(values.modelRef);
    modelNamespace = parsed.workspace;
    modelName = parsed.name;
  } else {
    if (!values.fileset) {
      throw new Error('Select a fileset');
    }
    const fs = getPartsFromReference(values.fileset);
    const baseName = values.name.trim();
    modelNamespace = workspace;
    modelName = baseName;

    reportStage('Registering model from fileset…');
    await modelsCreateModel(workspace, {
      name: modelName,
      fileset: `${fs.workspace}/${fs.name}`,
    });
  }

  reportStage('Creating deployment configuration…');
  await modelsCreateDeploymentConfig(workspace, {
    name: configName,
    engine: values.engine,
    model_spec: {
      model_namespace: modelNamespace,
      model_name: modelName,
      lora_enabled: values.loraEnabled,
    },
    executor_config: {
      gpu: values.gpu,
      ...imageOverrides(values),
    },
    model_entity_id: `${modelNamespace}/${modelName}`,
  });

  reportStage('Creating deployment…');
  await modelsCreateDeployment(workspace, {
    name: deploymentName,
    config: configName,
  });
}

export function useCreateDeploymentBySource(workspace: string) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const invalidateDeploymentQueries = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: getModelsListDeploymentsQueryKey(workspace) });
    queryClient.invalidateQueries({ queryKey: getModelsListDeploymentConfigsQueryKey(workspace) });
    queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) });
    queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) });
  }, [queryClient, workspace]);

  const clearStatusMessage = useCallback(() => setStatusMessage(null), []);

  const createDeploymentFromWizard = useCallback(
    async (values: WizardFormValues, onSuccess: () => void) => {
      setSubmitError(null);
      setStatusMessage(null);
      setIsSubmitting(true);
      const baseName = values.name.trim();
      const deploymentName = deploymentNameFromWizardBaseName(baseName);
      const configName = configNameFromWizardBaseName(baseName);
      const reportStage = (message: string) => setStatusMessage(message);

      try {
        if (values.source === SOURCE_NGC) {
          await createNgcDeployment(workspace, values, deploymentName, configName, reportStage);
        } else if (values.source === SOURCE_HF) {
          await createHuggingFaceDeployment(
            workspace,
            values,
            deploymentName,
            configName,
            reportStage
          );
        } else if (values.source === SOURCE_WORKSPACE) {
          await createWorkspaceDeployment(
            workspace,
            values,
            deploymentName,
            configName,
            reportStage
          );
        }

        toast.success('Deployment created successfully.');
        invalidateDeploymentQueries();
        onSuccess();
      } catch (e) {
        setSubmitError(e instanceof Error ? getErrorMessage(e) : 'An unexpected error occurred');
        toast.error('Failed to create deployment. Check the message below and try again.');
      } finally {
        setIsSubmitting(false);
        setStatusMessage(null);
      }
    },
    [invalidateDeploymentQueries, toast, workspace]
  );

  return {
    createDeploymentFromWizard,
    isSubmitting,
    submitError,
    setSubmitError,
    statusMessage,
    clearStatusMessage,
  };
}
