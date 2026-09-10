// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResourceRef } from '@nemo/common/src/types';
import {
  defaultWizardValues,
  SOURCE_WORKSPACE,
  WORKSPACE_PICKER_MODEL,
  type WizardFormValues,
} from '@studio/routes/DeploymentsListRoute/CreateDeploymentSidePanel/schema';

/**
 * Whether Studio deploys the base model as part of starting this job.
 *
 * - `deploy` — create the ModelDeploymentConfig **and** the ModelDeployment
 *   before submitting the job. The deployment converges while training runs.
 * - `skip` — start the job only. The adapter will not be servable until someone
 *   deploys the base model, which they can do from the Deployments page at any
 *   time, including after the job finishes.
 *
 * There is deliberately no "deploy when the job finishes" option. That is the
 * GPU-correct point in time, but it needs `deployment_config` on the job, which
 * exists only on `UnslothJobInput` — `AutomodelJobInput` and `RlJobInput` have
 * no such field and their compilers hardcode it to `None`. Offering it before
 * the backend carries it would promise something the API cannot honour.
 */
export const BASE_DEPLOYMENT_DEPLOY = 'deploy' as const;
export const BASE_DEPLOYMENT_SKIP = 'skip' as const;
export type BaseDeploymentChoice = typeof BASE_DEPLOYMENT_DEPLOY | typeof BASE_DEPLOYMENT_SKIP;

/** Deploying is the default: without it the run produces nothing servable. */
export const DEFAULT_BASE_DEPLOYMENT_CHOICE: BaseDeploymentChoice = BASE_DEPLOYMENT_DEPLOY;

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
