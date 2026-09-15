// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResourceRef } from '@nemo/common/src/types';
import {
  defaultWizardValues,
  SOURCE_WORKSPACE,
  WORKSPACE_PICKER_MODEL,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';

/**
 * Deployment is opt-out, for every run and both targets.
 *
 * One constant rather than one per flow: a user who fine-tunes a model means to
 * use it, and having to discover a switch before that becomes possible makes the
 * common case the one that needs extra work. Opting out stays a single click for
 * the runs trained only to be evaluated.
 *
 * Defaulting on costs little because Studio does not create the deployment. On
 * submit it creates the ModelDeploymentConfig and passes its name to the job as
 * `deployment_config`; the job's own model_entity task resolves that name once
 * training finishes and creates the ModelDeployment then — see `launch_model` in
 * `nmp.customization_common.tasks.model_entity.run`. So no GPU is claimed while
 * the run is in progress, and the up-front config still fails fast:
 * `_validate_engine_config` runs synchronously inside `create_deployment_config`,
 * so a bad engine or missing image is rejected before the job is submitted.
 */
export const DEPLOY_BY_DEFAULT = true;

/**
 * The deployment form nested inside the fine-tuning form reuses `WizardFormValues`
 * unchanged rather than a narrowed subset.
 *
 * Two reasons. `EngineFields` / `GPULoraFields` / `AdvancedSettingsAccordion` are
 * typed `Control<WizardFormValues>`, and react-hook-form's `Control<A>` is not
 * assignable to `Control<B>` — a subset type would force either a cast or three
 * genericised components. And pinning `source` to the Workspace branch makes
 * `createDeploymentWizardSchema`'s `superRefine` validate exactly what this flow
 * needs (image-for-engine, `modelRef` present) and skip the rest.
 */
export function baseDeploymentDefaults(modelRef?: string): WizardFormValues {
  return {
    ...defaultWizardValues(),
    source: SOURCE_WORKSPACE,
    workspacePickerType: WORKSPACE_PICKER_MODEL,
    modelRef: (modelRef ?? '') as ResourceRef,
    // Not a preference here: an adapter served by a base deployment without LoRA
    // support is unservable, so this is an invariant of the flow.
    loraEnabled: true,
  };
}

/**
 * Defaults for deploying the **model this run produces**.
 *
 * `modelRef` is a forward reference: the output Model Entity does not exist when
 * the config is created, and will not until the job finishes. That is deliberate
 * and supported — `create_deployment_config` never requires the referenced entity
 * to exist, and only looks one up at all when `model_entity_id` is absent, which
 * `createWorkspaceDeploymentConfig` always supplies. Pointing the config forward
 * is what lets the job deploy the moment training ends.
 *
 * `loraEnabled` is left at the wizard's default rather than pinned. For a
 * full-weight model it decides whether adapters trained against it later can be
 * served alongside it, which is a genuine preference and nothing this run
 * depends on.
 */
export function outputDeploymentDefaults(workspace: string, outputName?: string): WizardFormValues {
  return {
    ...defaultWizardValues(),
    source: SOURCE_WORKSPACE,
    workspacePickerType: WORKSPACE_PICKER_MODEL,
    modelRef: (outputName ? `${workspace}/${outputName}` : '') as ResourceRef,
  };
}

/**
 * Deployment base name derived from a model name or `workspace/name` reference.
 *
 * `deploymentNameFromWizardBaseName` appends `-deployment` and
 * `configNameFromWizardBaseName` appends `-config`, matching the wizard so the two
 * entry points produce consistently-named assets. Derived rather than
 * user-editable: one deployment per model is the point, and a free-form name
 * invites duplicates.
 *
 * Used for both flows. The adapter flow passes the base model's ref, so repeated
 * runs against one base collide on the name; the output flow passes the run's
 * output name, which is unique per job and therefore cannot.
 */
export function baseDeploymentName(modelRef: string | undefined): string {
  if (!modelRef) return '';
  const name = modelRef.includes('/') ? modelRef.slice(modelRef.indexOf('/') + 1) : modelRef;
  return name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}
