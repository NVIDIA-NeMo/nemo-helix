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
 * Deploying is the default: without it the run produces nothing servable.
 *
 * On submit Studio creates the ModelDeploymentConfig and passes its name to the
 * job as `deployment_config`. The job's own model_entity task resolves that name
 * once training finishes and creates the ModelDeployment then — see
 * `launch_model` in `nmp.customization_common.tasks.model_entity.run`.
 *
 * Studio deliberately does not create the deployment itself. That is the same
 * end state reached hours earlier, with a serving GPU idling through the entire
 * training run for nothing. Handing the job a config name costs nothing while it
 * trains and still fails fast: `_validate_engine_config` runs synchronously
 * inside `create_deployment_config`, so a bad engine or missing image is
 * rejected before the job is submitted.
 */
export const DEFAULT_DEPLOY_BASE_MODEL = true;

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
 * Deployment base name derived from the base model.
 *
 * `deploymentNameFromWizardBaseName` appends `-deployment` and
 * `configNameFromWizardBaseName` appends `-config`, matching the wizard so the two
 * entry points produce consistently-named assets. Derived rather than
 * user-editable: one deployment per base model is the point, and a free-form name
 * invites duplicates.
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
