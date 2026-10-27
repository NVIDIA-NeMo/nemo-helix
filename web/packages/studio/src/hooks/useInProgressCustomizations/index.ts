// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { getQueryKeyInfiniteModels } from '@nemo/common/src/api/models/useModels';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import { getModelsListModelsQueryKey } from '@nemo/sdk/generated/platform/models';
import {
  FinetuningType,
  HelixJobStatus,
  type HelixJobListSortField,
  type HelixJobResponse,
  type HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useRef } from 'react';

/**
 * Job states in which a run has not yet produced its model.
 *
 * A customization job registers its output in `model-entity-creation`, the last of its
 * four steps — after training *and* after the checkpoint upload. For every status here
 * the thing the user is waiting for exists nowhere in the model catalogue, which is
 * precisely why these runs need a surface of their own.
 *
 * The transitional states are included because such a run may still reach `completed`.
 */
export const IN_PROGRESS_JOB_STATUSES: HelixJobStatus[] = [
  HelixJobStatus.created,
  HelixJobStatus.queued,
  HelixJobStatus.pending,
  HelixJobStatus.active,
  HelixJobStatus.cancelling,
  HelixJobStatus.paused,
  HelixJobStatus.pausing,
  HelixJobStatus.resuming,
];

/** How often in-flight runs are re-read, so a finished one leaves the table on its own. */
export const IN_PROGRESS_POLL_INTERVAL_MS = 15_000;

/** Runs are few; one page is ample, and the table is not paginated. */
const PAGE_SIZE = 50;

export interface InProgressCustomization {
  /** Platform job name — the route segment of the job detail page. */
  jobName: string;
  /** Name the run will register its output under, model or adapter. */
  outputName: string;
  status: HelixJobStatus;
  /** Base model URN the run started from, when the spec records one. */
  baseModel?: string;
  /** e.g. `lora`, `all_weights`. Absent for specs that do not declare one. */
  finetuningType?: FinetuningType;
  createdAt?: string;
}

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length > 0 ? value : undefined;

/**
 * The base model a run started from. Backends spell it differently: automodel and RL hold
 * a URN string in `spec.model`, unsloth an object with a `name`.
 */
const readBaseModel = (spec: Record<string, unknown> | undefined): string | undefined =>
  spec ? (asString(spec.model) ?? asString(asRecord(spec.model)?.name)) : undefined;

/** The spec is untyped here, so only a value the UI can format is accepted. */
const asFinetuningType = (value: unknown): FinetuningType | undefined =>
  Object.values(FinetuningType).find((type) => type === value);

const toInProgressCustomization = (job: HelixJobResponse): InProgressCustomization | null => {
  const spec = asRecord(job.spec);
  const outputName = asString(asRecord(spec?.output)?.name);
  // A run that declares no output has nothing for the user to wait for.
  if (!outputName) return null;
  return {
    jobName: job.name,
    outputName,
    status: job.status,
    baseModel: readBaseModel(spec),
    finetuningType: asFinetuningType(asRecord(spec?.training)?.finetuning_type),
    createdAt: job.created_at,
  };
};

export interface UseInProgressCustomizationsResult {
  runs: InProgressCustomization[];
  isLoading: boolean;
  isError: boolean;
}

/**
 * Fine-tuning runs that are under way and have not registered their output yet.
 *
 * Polls, and refreshes the model lists when a run drops out of the in-flight set — that
 * is the moment its model or adapter appears, and the lists were fetched before it
 * existed. Without this the finished model stays invisible until a page reload.
 */
export const useInProgressCustomizations = (
  workspace: string
): UseInProgressCustomizationsResult => {
  const queryClient = useQueryClient();
  const enabled = Boolean(workspace);

  const { data, isLoading, isError } = useJobsListJobs(
    workspace,
    {
      page: 1,
      page_size: PAGE_SIZE,
      sort: '-created_at' as HelixJobListSortField,
      filter: withOperators<HelixJobsListFilter>({
        source: JOB_SOURCE.CUSTOMIZATION,
        // The `$in` operator (see `filterOperators.ts`), not `status: [...]`: an array
        // serializes to repeated `filter[status]=` params, which the API does not read as
        // set membership -- it matches no jobs at all, with no error to say so.
        status: { $in: IN_PROGRESS_JOB_STATUSES },
      }),
    },
    {
      query: {
        enabled,
        staleTime: 0,
        refetchInterval: (query) =>
          query.state.status === 'error' ? false : IN_PROGRESS_POLL_INTERVAL_MS,
      },
    }
  );

  const runs = useMemo(
    () => (data?.data ?? []).map(toInProgressCustomization).filter((run) => run !== null),
    [data?.data]
  );

  const previousJobNames = useRef<{ workspace: string; jobNames: string[] } | null>(null);
  useEffect(() => {
    if (!enabled || !data) return;
    const jobNames = runs.map((run) => run.jobName);
    const previous = previousJobNames.current;
    previousJobNames.current = { workspace, jobNames };
    // Keyed by workspace: comparing a new workspace's first result against the previous
    // one's names would read every job that did not carry over as finished.
    if (!previous || previous.workspace !== workspace) return;
    if (previous.jobNames.every((name) => jobNames.includes(name))) return;
    // Both key families: the paged list behind the table and the infinite one behind
    // every model dropdown.
    void queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) });
    void queryClient.invalidateQueries({ queryKey: getQueryKeyInfiniteModels(workspace) });
  }, [data, enabled, queryClient, runs, workspace]);

  return { runs, isLoading: enabled && isLoading, isError };
};
