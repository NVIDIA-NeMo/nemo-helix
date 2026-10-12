// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { getQueryKeyInfiniteModels } from '@nemo/common/src/api/models/useModels';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import { getModelsListModelsQueryKey } from '@nemo/sdk/generated/platform/models';
import {
  HelixJobStatus,
  type HelixJobListSortField,
  type HelixJobResponse,
  type HelixJobsListFilter,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useRef } from 'react';

/**
 * Job states in which a fine-tuning run has not yet registered its output model.
 *
 * The model entity is created by the `model_entity` step, which runs after training
 * finishes -- so for every status here the model the user is waiting for does not
 * exist in the catalogue yet. `cancelling`, `pausing` and `resuming` are included
 * because they are transitional: the run may still reach `completed`.
 */
export const IN_FLIGHT_CUSTOMIZATION_STATUSES: HelixJobStatus[] = [
  HelixJobStatus.created,
  HelixJobStatus.pending,
  HelixJobStatus.active,
  HelixJobStatus.cancelling,
  HelixJobStatus.paused,
  HelixJobStatus.pausing,
  HelixJobStatus.resuming,
];

/** How often in-flight runs are re-read, so a finished one becomes selectable on its own. */
export const PENDING_CUSTOMIZATION_POLL_INTERVAL_MS = 15_000;

/**
 * Most-recent in-flight runs to track. A placeholder row is only useful while the user
 * is waiting on it, and a workspace with more than this many concurrent runs is better
 * served by the Fine-tuning list itself than by a dropdown full of placeholders.
 */
const PAGE_SIZE = 50;

export interface PendingCustomization {
  /** Platform job name -- the route segment of the job detail page. */
  jobName: string;
  /** Name the run will register its output model under once training ends. */
  outputModelName: string;
  status: HelixJobStatus;
  /** Base model URN the run started from, when the spec records one. */
  baseModel?: string;
  createdAt?: string;
  workspace: string;
}

/**
 * Marks the synthetic model entities built by {@link toPendingModelEntity}. A real
 * entity's id is a UUID, so the prefix cannot collide with one.
 */
export const PENDING_MODEL_ID_PREFIX = 'pending-customization:';

export const isPendingModelEntity = (model: Pick<ModelEntity, 'id'>): boolean =>
  model.id.startsWith(PENDING_MODEL_ID_PREFIX);

/** The job name carried by a synthetic entity, for linking a placeholder back to its run. */
export const getPendingJobName = (model: Pick<ModelEntity, 'id'>): string | undefined =>
  isPendingModelEntity(model) ? model.id.slice(PENDING_MODEL_ID_PREFIX.length) : undefined;

/**
 * Presents an in-flight run as the model entity it is going to become, so a list or a
 * dropdown built on `ModelEntity` can show it without a parallel code path. Callers are
 * expected to render it as unavailable -- see {@link isPendingModelEntity}.
 */
export const toPendingModelEntity = (pending: PendingCustomization): ModelEntity => ({
  id: `${PENDING_MODEL_ID_PREFIX}${pending.jobName}`,
  name: pending.outputModelName,
  workspace: pending.workspace,
  created_at: pending.createdAt ?? '',
  updated_at: pending.createdAt ?? '',
  ...(pending.baseModel ? { base_model: pending.baseModel } : {}),
});

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length > 0 ? value : undefined;

/**
 * The base model a run started from. Each backend spells it differently -- automodel and
 * RL hold a URN string in `spec.model`, unsloth an object with a `name`.
 */
const readBaseModel = (spec: Record<string, unknown> | undefined): string | undefined => {
  if (!spec) return undefined;
  return asString(spec.model) ?? asString(asRecord(spec.model)?.name);
};

const toPendingCustomization = (job: HelixJobResponse): PendingCustomization | null => {
  const spec = asRecord(job.spec);
  const outputModelName = asString(asRecord(spec?.output)?.name);
  // A run that declares no output name has no model for the user to wait for.
  if (!outputModelName) return null;
  return {
    jobName: job.name,
    outputModelName,
    status: job.status,
    baseModel: readBaseModel(spec),
    createdAt: job.created_at,
    workspace: job.workspace,
  };
};

export interface UsePendingCustomizationsOptions {
  enabled?: boolean;
}

export interface UsePendingCustomizationsResult {
  pending: PendingCustomization[];
  isLoading: boolean;
}

/**
 * Fine-tuning runs that are under way and have not produced their model entity yet.
 *
 * A run registers its output only after training finishes, so between submission and
 * completion the model the user is waiting for exists nowhere -- not in the Fine-tuning
 * list, not in a model picker. This hook supplies the missing representation, and polls
 * so that a run leaving the in-flight set invalidates the model lists: the real entity
 * then appears without the user reloading the page.
 */
export const usePendingCustomizations = (
  workspace: string | null | undefined,
  { enabled = true }: UsePendingCustomizationsOptions = {}
): UsePendingCustomizationsResult => {
  const queryClient = useQueryClient();
  const isEnabled = enabled && Boolean(workspace);

  const { data, isLoading } = useJobsListJobs(
    workspace ?? '',
    {
      page: 1,
      page_size: PAGE_SIZE,
      sort: '-created_at' as HelixJobListSortField,
      filter: withOperators<HelixJobsListFilter>({
        source: JOB_SOURCE.CUSTOMIZATION,
        status: IN_FLIGHT_CUSTOMIZATION_STATUSES,
      }),
    },
    {
      query: {
        enabled: isEnabled,
        staleTime: 0,
        refetchInterval: (query) =>
          query.state.status === 'error' ? false : PENDING_CUSTOMIZATION_POLL_INTERVAL_MS,
      },
    }
  );

  const pending = useMemo(
    () => (data?.data ?? []).map(toPendingCustomization).filter((p) => p !== null),
    [data?.data]
  );

  // A run dropping out of the in-flight set is the moment its model entity appears. The
  // model lists were fetched before it existed, so refresh them rather than leaving the
  // user to reload the page.
  const previousJobNames = useRef<string[] | null>(null);
  useEffect(() => {
    if (!isEnabled || !data) return;
    const jobNames = pending.map((p) => p.jobName);
    const previous = previousJobNames.current;
    previousJobNames.current = jobNames;
    if (!previous) return;
    const finished = previous.filter((name) => !jobNames.includes(name));
    if (finished.length === 0) return;
    // Both key families: the paged list behind the Fine-tuning table and the infinite
    // one behind every model dropdown.
    void queryClient.invalidateQueries({
      queryKey: getModelsListModelsQueryKey(workspace ?? ''),
    });
    void queryClient.invalidateQueries({ queryKey: getQueryKeyInfiniteModels(workspace ?? '') });
  }, [data, isEnabled, pending, queryClient, workspace]);

  return { pending, isLoading: isEnabled && isLoading };
};
