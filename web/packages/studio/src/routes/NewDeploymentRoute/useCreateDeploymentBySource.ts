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

import {
  getErrorMessage,
  isNotFoundError,
  isVersionConflictError,
} from '@nemo/common/src/api/common/utils';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  filesCreateFileset,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import {
  getModelsListDeploymentConfigsQueryKey,
  modelsCreateDeploymentConfig,
  modelsGetLatestDeploymentConfig,
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
  type ModelDeploymentConfig,
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

export type ReportStage = (message: string) => void;

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

/**
 * Create the `ModelDeploymentConfig` for a model that already exists in the workspace.
 *
 * Split out from `createWorkspaceDeployment` because a config is useful on its own:
 * the fine-tuning form creates one for the adapter's **base model** and hands its
 * name to the job as `deployment_config`, letting the job create the deployment
 * once training finishes rather than idling a GPU for the whole run.
 *
 * Validation is the reason this is worth doing up front. `_validate_engine_config`
 * runs synchronously inside `create_deployment_config`, so a missing image for the
 * NIM engine fails here in milliseconds — before the caller commits to anything
 * expensive.
 *
 * Callers outside the wizard own their own error surface and query invalidation.
 */
export async function createWorkspaceDeploymentConfig(
  workspace: string,
  values: WizardFormValues,
  configName: string,
  reportStage: ReportStage
): Promise<ModelDeploymentConfig> {
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
  return await modelsCreateDeploymentConfig(workspace, {
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
}

/**
 * Create an **unbound** `ModelDeploymentConfig` — an engine and an executor, and no model.
 *
 * For a run that produces its own Model Entity (`all_weights`, `lora_merged`, DPO),
 * the model this config will serve does not exist at submit time and cannot be named.
 * A config carrying neither `model_entity_id` nor `model_spec.model_name` is a
 * template: `is_unbound_deployment_config` recognises it, every compiler accepts it
 * from any job, and the model_entity task binds it to the trained model when the run
 * finishes — copying this engine, executor and serving options onto a derived config
 * that names the new model. The template itself is left alone, so it stays reusable.
 *
 * `lora_enabled` is still sent. It is a serving option rather than a model link, it
 * survives the copy onto the derived config, and it is what decides whether adapters
 * trained against the output model can later be served alongside it.
 *
 * The reason to create anything up front is the same as for a bound config:
 * `_validate_engine_config` runs synchronously inside `create_deployment_config`, so a
 * missing image for the NIM engine is rejected in milliseconds — before the caller
 * commits to a training run.
 */
export async function createUnboundDeploymentConfig(
  workspace: string,
  values: WizardFormValues,
  configName: string,
  reportStage: ReportStage
): Promise<ModelDeploymentConfig> {
  reportStage('Creating deployment configuration…');
  return await modelsCreateDeploymentConfig(workspace, {
    name: configName,
    engine: values.engine,
    model_spec: {
      lora_enabled: values.loraEnabled,
    },
    executor_config: {
      gpu: values.gpu,
      ...imageOverrides(values),
    },
  });
}

/** Create a config + deployment for a model that already exists in the workspace. */
export async function createWorkspaceDeployment(
  workspace: string,
  values: WizardFormValues,
  deploymentName: string,
  configName: string,
  reportStage: ReportStage
) {
  await createWorkspaceDeploymentConfig(workspace, values, configName, reportStage);

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

/** Whether `ensureWorkspaceDeploymentConfig` created the config or found one already there. */
export interface EnsuredDeploymentConfig {
  /** Always the config the caller should reference by name. */
  config: ModelDeploymentConfig;
  /** True when an existing config was adopted rather than created. */
  reused: boolean;
}

/**
 * Create the config, or adopt the one already under that name.
 *
 * The fine-tuning form derives the config name from the **base model**, deliberately:
 * one LoRA-enabled deployment of a base can serve every adapter trained against it, so
 * a name that varies per run would invite duplicates. The cost is that the name is not
 * free the second time — and it stays taken for the hours the first job trains, because
 * readiness only reports `serving-lora` once that job has actually deployed. Without
 * this, a second adapter run against the same base fails at submit with a 409 and never
 * starts training.
 *
 * Adopting the existing config is the intent rather than a fallback: one config per base
 * is what the naming scheme is *for*. But the adopted config was created by an earlier
 * run and may specify a different engine or GPU count than the caller just filled in, so
 * `reused` is returned rather than swallowed — the caller is expected to say so.
 *
 * Deliberately not an update-to-a-new-version: `get_deployment_config` resolves the
 * latest, and the earlier job has not resolved its config yet — it does that when
 * training ends. Bumping the version here would silently change what that job deploys.
 */
export async function ensureWorkspaceDeploymentConfig(
  workspace: string,
  values: WizardFormValues,
  configName: string,
  reportStage: ReportStage
): Promise<EnsuredDeploymentConfig> {
  return ensureDeploymentConfig(
    workspace,
    configName,
    () => createWorkspaceDeploymentConfig(workspace, values, configName, reportStage),
    reportStage
  );
}

/**
 * Create the unbound template, or adopt the one already under that name.
 *
 * The output flow derives the name from the run's output model, so a collision only
 * happens when that name is reused — which the backend also tolerates, updating the
 * existing Model Entity rather than failing. Failing the submit over the config alone
 * would be a worse answer than deploying with the settings already recorded, so this
 * adopts for the same reason the adapter flow does, and reports it the same way.
 */
export async function ensureUnboundDeploymentConfig(
  workspace: string,
  values: WizardFormValues,
  configName: string,
  reportStage: ReportStage
): Promise<EnsuredDeploymentConfig> {
  return ensureDeploymentConfig(
    workspace,
    configName,
    () => createUnboundDeploymentConfig(workspace, values, configName, reportStage),
    reportStage
  );
}

/** Shared create-or-adopt: only a 409 on the name is an adoption; everything else propagates. */
async function ensureDeploymentConfig(
  workspace: string,
  configName: string,
  create: () => Promise<ModelDeploymentConfig>,
  reportStage: ReportStage
): Promise<EnsuredDeploymentConfig> {
  try {
    return { config: await create(), reused: false };
  } catch (error) {
    if (!isVersionConflictError(error)) throw error;

    reportStage('Reusing the existing deployment configuration…');
    const existing = await modelsGetLatestDeploymentConfig(workspace, configName);
    return { config: existing, reused: true };
  }
}
