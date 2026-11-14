// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EntitiesListEntitiesParams, RailsConfig } from '@nemo/sdk/generated/platform/schema';
import {
  createGuardrailCheck,
  type CreateGuardrailCheckInput,
  getGuardrailChecksQueryKey,
  guardrailChecksForConfigFilter,
  listGuardrailChecks,
  runGuardrailChecks,
  updateGuardrailCheck,
  type UpdateGuardrailCheckPatch,
} from '@studio/api/guardrail-checks/guardrailChecks';
import { invalidateGuardrailChecksCaches } from '@studio/api/guardrail-checks/invalidateGuardrailChecksCaches';
import type { GuardrailCheckEntity, GuardrailChecksPage } from '@studio/api/guardrail-checks/types';
import {
  queryOptions,
  useMutation,
  type UseMutationOptions,
  useQuery,
  type UseQueryOptions,
} from '@tanstack/react-query';

type QueryOpts<TData> = Omit<UseQueryOptions<TData, Error>, 'queryKey' | 'queryFn'>;

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** React Query options for a (optionally filtered/paginated) list of guardrail checks. */
const getGuardrailChecksQueryOptions = (
  workspace: string,
  params?: EntitiesListEntitiesParams,
  options?: QueryOpts<GuardrailChecksPage>
) =>
  queryOptions<GuardrailChecksPage, Error>({
    queryKey: getGuardrailChecksQueryKey(workspace, params),
    queryFn: ({ signal }) => listGuardrailChecks(workspace, params, signal),
    ...options,
  });

/** Query hook for the guardrail checks in a workspace. */
const useGuardrailChecks = (
  workspace: string,
  params?: EntitiesListEntitiesParams,
  options?: QueryOpts<GuardrailChecksPage>
) => useQuery(getGuardrailChecksQueryOptions(workspace, params, options));

/** Convenience hook for the checks belonging to a single guardrail config. */
export const useGuardrailChecksForConfig = (
  workspace: string,
  configId: string,
  params?: Omit<EntitiesListEntitiesParams, 'filter'>,
  options?: QueryOpts<GuardrailChecksPage>
) =>
  useGuardrailChecks(
    workspace,
    { ...params, filter: guardrailChecksForConfigFilter(configId) },
    options
  );

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

type UseCreateGuardrailCheckOptions = Omit<
  UseMutationOptions<
    GuardrailCheckEntity,
    Error,
    { workspace: string; input: CreateGuardrailCheckInput }
  >,
  'mutationFn'
>;

/** Mutation hook to create a guardrail check; refreshes check caches on success. */
export const useCreateGuardrailCheck = (options?: UseCreateGuardrailCheckOptions) =>
  useMutation({
    ...options,
    mutationFn: ({ workspace, input }) => createGuardrailCheck(workspace, input),
    onSuccess: (...args) => {
      const [entity] = args;
      invalidateGuardrailChecksCaches(entity.workspace, entity.name);
      options?.onSuccess?.(...args);
    },
  });

type UseUpdateGuardrailCheckOptions = Omit<
  UseMutationOptions<
    GuardrailCheckEntity,
    Error,
    { workspace: string; name: string; patch: UpdateGuardrailCheckPatch }
  >,
  'mutationFn'
>;

/** Mutation hook to update a guardrail check; refreshes check caches on success. */
export const useUpdateGuardrailCheck = (options?: UseUpdateGuardrailCheckOptions) =>
  useMutation({
    ...options,
    mutationFn: ({ workspace, name, patch }) => updateGuardrailCheck(workspace, name, patch),
    onSuccess: (...args) => {
      const [entity] = args;
      invalidateGuardrailChecksCaches(entity.workspace, entity.name);
      options?.onSuccess?.(...args);
    },
  });

type UseRunGuardrailChecksOptions = Omit<
  UseMutationOptions<
    Awaited<ReturnType<typeof runGuardrailChecks>>,
    Error,
    { workspace: string; checks: GuardrailCheckEntity[]; draftConfig?: RailsConfig }
  >,
  'mutationFn'
>;

/** Mutation hook to run a batch of guardrail checks. */
export const useRunGuardrailChecks = (options?: UseRunGuardrailChecksOptions) =>
  useMutation({
    ...options,
    mutationFn: ({ workspace, checks, draftConfig }) =>
      runGuardrailChecks(workspace, checks, draftConfig),
    onSuccess: (...args) => {
      const [, variables] = args;
      invalidateGuardrailChecksCaches(variables.workspace);
      options?.onSuccess?.(...args);
    },
  });
