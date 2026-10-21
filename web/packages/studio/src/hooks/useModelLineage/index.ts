// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useCustomizationJobForModel } from '@studio/hooks/useCustomizationJobForModel';
import {
  type DeploymentIndicatorState,
  type DeploymentStatusTarget,
  useModelDeploymentStatuses,
} from '@studio/hooks/useModelDeploymentStatuses';
import { useMemo } from 'react';

export interface FilesetRef {
  workspace: string;
  name: string;
}

/** Parses a `workspace/name` fileset reference; a bare name lives in `fallbackWorkspace`. */
export const parseFilesetRef = (
  filesetRef: string | undefined,
  fallbackWorkspace: string
): FilesetRef | null => {
  if (!filesetRef) return null;
  if (!filesetRef.includes('/')) return { workspace: fallbackWorkspace, name: filesetRef };
  const { workspace, name } = getPartsFromReference(filesetRef);
  return workspace && name ? { workspace, name } : null;
};

export interface ModelLineage {
  model: ModelEntity | undefined;
  /** Null on a model page, or when the named adapter is missing from the model. */
  adapter: Adapter | null;
  isLoading: boolean;
  isError: boolean;
  /** The model loaded but has no adapter with the requested name. */
  isAdapterMissing: boolean;
  /** Link to the job in `workspace`: it is the entity's own, not necessarily the page's. */
  job: { jobName: string | undefined; workspace: string; isLoading: boolean };
  /** Null when the entity records no fileset. */
  fileset: FilesetRef | null;
  deployment: DeploymentIndicatorState;
}

const TARGET_KEY = 'lineage';

/**
 * Resolves what a model or adapter is linked to: the customization job that produced it,
 * the fileset holding its weights, and the deployment serving it. Each link resolves on its
 * own, so callers can show each one as soon as it arrives.
 *
 * Deployment state is the same one the models list shows, so the two never disagree; an
 * adapter is served through its model's providers and can be `adapter-not-loaded`.
 */
export const useModelLineage = (
  workspace: string,
  modelName: string,
  adapterName?: string
): ModelLineage => {
  const {
    data: model,
    isLoading,
    isError,
  } = useModelsGetModel(workspace, modelName, undefined, {
    query: { enabled: Boolean(workspace && modelName), retry: false },
  });

  const adapter = adapterName
    ? (model?.adapters?.find((candidate) => candidate.name === adapterName) ?? null)
    : null;
  const subject = adapterName ? adapter : model;
  const isAdapterMissing = Boolean(adapterName && model && !adapter);

  const job = useCustomizationJobForModel(subject);

  const targets = useMemo<DeploymentStatusTarget[]>(
    () => (model && subject ? [{ key: TARGET_KEY, model, adapter: adapter ?? undefined }] : []),
    [model, subject, adapter]
  );
  const deployment: DeploymentIndicatorState =
    useModelDeploymentStatuses(targets).get(TARGET_KEY) ??
    (isLoading ? { kind: 'loading' } : { kind: 'unknown' });

  return {
    model,
    adapter,
    isLoading,
    isError,
    isAdapterMissing,
    job: { ...job, workspace: subject?.workspace ?? workspace },
    fileset: parseFilesetRef(subject?.fileset, subject?.workspace ?? workspace),
    deployment,
  };
};
