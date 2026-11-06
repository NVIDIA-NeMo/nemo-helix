// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EntitiesListEntitiesParams, RailsConfig } from '@nemo/sdk/generated/platform/schema';
import {
  createGuardrailCheck,
  type CreateGuardrailCheckInput,
  deleteGuardrailCheck,
  getGuardrailCheck,
  getGuardrailCheckQueryKey,
  getGuardrailChecksQueryKey,
  guardrailChecksForConfigFilter,
  listGuardrailChecks,
  runGuardrailCheck,
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
export const getGuardrailChecksQueryOptions = (
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
export const useGuardrailChecks = (
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

/** React Query options for a single guardrail check, scoped by its parent config. */
export const getGuardrailCheckQueryOptions = (
  workspace: string,
  name: string,
  parent: string,
  options?: QueryOpts<GuardrailCheckEntity>
) =>
  queryOptions<GuardrailCheckEntity, Error>({
    queryKey: getGuardrailCheckQueryKey(workspace, name, parent),
    queryFn: ({ signal }) => getGuardrailCheck(workspace, name, parent, signal),
    ...options,
  });

/**
 * Fetch a single guardrail check by name. Checks are hard children of a config,
 * so `parent` (the parent config's entity id) is required to resolve the lookup.
 */
export const useGuardrailCheck = (
  workspace: string,
  name: string,
  parent: string,
  options?: QueryOpts<GuardrailCheckEntity>
) => useQuery(getGuardrailCheckQueryOptions(workspace, name, parent, options));

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export type UseCreateGuardrailCheckOptions = Omit<
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

export type UseUpdateGuardrailCheckOptions = Omit<
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

export type UseDeleteGuardrailCheckOptions = Omit<
  UseMutationOptions<void, Error, GuardrailCheckEntity>,
  'mutationFn'
>;

/**
 * Mutation hook to delete a guardrail check; refreshes check caches on success.
 * The loaded entity retains its immutable ID and observed version.
 */
export const useDeleteGuardrailCheck = (options?: UseDeleteGuardrailCheckOptions) =>
  useMutation({
    ...options,
    mutationFn: deleteGuardrailCheck,
    onSuccess: (...args) => {
      const [, variables] = args;
      invalidateGuardrailChecksCaches(variables.workspace, variables.name);
      options?.onSuccess?.(...args);
    },
  });

export type UseRunGuardrailCheckOptions = Omit<
  UseMutationOptions<
    Awaited<ReturnType<typeof runGuardrailCheck>>,
    Error,
    { workspace: string; check: GuardrailCheckEntity; draftConfig?: RailsConfig }
  >,
  'mutationFn'
>;

/** Mutation hook to run a single guardrail check and persist the run. */
export const useRunGuardrailCheck = (options?: UseRunGuardrailCheckOptions) =>
  useMutation({
    ...options,
    mutationFn: ({ workspace, check, draftConfig }) =>
      runGuardrailCheck(workspace, check, draftConfig),
    onSuccess: (...args) => {
      const [, variables] = args;
      invalidateGuardrailChecksCaches(variables.workspace, variables.check.name);
      options?.onSuccess?.(...args);
    },
  });

export type UseRunGuardrailChecksOptions = Omit<
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
