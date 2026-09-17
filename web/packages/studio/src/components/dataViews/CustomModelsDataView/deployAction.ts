// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelDeploymentStatus, type ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { getWorkspaceNewDeploymentRoute } from '@studio/routes/utils';

export interface DeployActionTarget {
  /** Menu label. Always "Deploy", including on adapter rows. */
  label: string;
  /** Create Deployment wizard route, with the model prefilled. */
  href: string;
}

/**
 * Whether a row should offer Deploy, and what it should say.
 *
 * Offered only when nothing is currently serving the row, so the action never
 * appears next to a healthy deployment:
 *
 * | Status                    | Action                                      |
 * | ------------------------- | ------------------------------------------- |
 * | Not deployed              | Deploy                                      |
 * | Failed / Unavailable      | Deploy — nothing is serving it; redeploying is the fix |
 * | Not served (adapter)      | Deploy — points at the base model, with LoRA, which the wizard enables by default |
 * | Deployed / Served         | none — already serving                      |
 * | Deploying                 | none — a deployment is already on its way    |
 * | Available                 | none — reachable through an external provider |
 * | Unknown                   | none — we could not read the providers, so offering Deploy might duplicate a live one |
 *
 * Returns null when deployments are disabled: the wizard route is gated on the
 * same flag, so the entry would lead nowhere. Mirrors `DeployModelCta`.
 *
 * @param state - Resolved status for the row.
 * @param model - The model to deploy. For an adapter row this is the parent, since
 *   an adapter is loaded by whatever serves its base model.
 * @returns The action to render, or null when the row should not offer one.
 */
export function getDeployAction(
  state: DeploymentIndicatorState | undefined,
  model: ModelEntity
): DeployActionTarget | null {
  if (!DEPLOYMENTS_ENABLED) return null;
  if (!state) return null;

  const offer =
    state.kind === 'not-deployed' ||
    state.kind === 'adapter-not-loaded' ||
    (state.kind === 'served' &&
      (state.status === ModelDeploymentStatus.ERROR ||
        state.status === ModelDeploymentStatus.DELETED ||
        state.status === ModelDeploymentStatus.LOST));

  if (!offer) return null;

  // An adapter has no deployment of its own: it is loaded by whatever serves its
  // base model, so that is what the wizard is pointed at. The label stays "Deploy"
  // either way; callers pass the parent model for adapter rows.
  const modelRef = `${model.workspace}/${model.name}`;

  return {
    label: 'Deploy',
    href: getWorkspaceNewDeploymentRoute(model.workspace, { model: modelRef }),
  };
}
