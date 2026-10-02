// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { toInferenceModelEntityId } from '@nemo/common/src/utils/models';
import {
  getModelsGetLatestDeploymentQueryKey,
  modelsGetLatestDeployment,
} from '@nemo/sdk/generated/platform/model-deployments';
import {
  getModelsGetProviderQueryKey,
  modelsGetProvider,
} from '@nemo/sdk/generated/platform/model-providers';
import type {
  Adapter,
  ModelDeployment,
  ModelDeploymentStatus,
  ModelEntity,
  ModelProvider,
} from '@nemo/sdk/generated/platform/schema';
import { useQueries, type UseQueryResult } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export type DeploymentIndicatorState =
  /** Still resolving providers or the deployment. */
  | { kind: 'loading' }
  /**
   * A provider lists this exact id in its `served_models`, so inference would
   * route. `hasDeployment` says whether the serving provider names a backing
   * deployment: false means an external provider (such as `default/build`) that
   * is reachable but has no deployment to report on, whereas true with no
   * `status` means a deployment exists but could not be read.
   */
  | {
      kind: 'served';
      hasDeployment: boolean;
      status?: ModelDeploymentStatus;
      statusMessage?: string;
      providerRef?: string;
    }
  /** Adapter only: the base model is served, but this adapter is not loaded. */
  | { kind: 'adapter-not-loaded' }
  /** Nothing serves this id, and (for an adapter) nothing serves its base either. */
  | { kind: 'not-deployed' }
  /**
   * A provider could not be fetched, so whether this is served is genuinely
   * unknown. Distinct from `not-deployed`, which asserts nothing serves it.
   */
  | { kind: 'unknown' };

const PROVIDER_STALE_TIME = 5 * 60 * 1000;

/** One table row to resolve: a model, or one of its adapters. */
export interface DeploymentStatusTarget {
  /** Row id, used as the key of the returned map. */
  key: string;
  /** The model that owns the row. For an adapter row, the *parent* model. */
  model: ModelEntity;
  /** Set when the row is an adapter subrow. */
  adapter?: Adapter;
}

/**
 * Resolve deployment status for every row of a table in one pass.
 *
 * Resolved for the whole table rather than per row because `rowActions` is a
 * plain callback, not a component, and may not call hooks. Doing it here gives
 * the status badge and the Deploy action a single answer rather than two that
 * can disagree.
 *
 * Providers are fetched once per provider, not once per row, so an expanded
 * model and all of its adapters cost a single request between them.
 *
 * @param targets - The rows to resolve. Pass a stable array; it is only read.
 * @returns Status by row key. A missing key means the row is still resolving.
 */
export function useModelDeploymentStatuses(
  targets: DeploymentStatusTarget[]
): Map<string, DeploymentIndicatorState> {
  const providerRefs = useMemo(() => {
    const refs = new Set<string>();
    for (const { model } of targets) {
      for (const ref of model.model_providers ?? []) refs.add(ref);
    }
    return [...refs].sort();
  }, [targets]);

  // `combine` rather than a useMemo over the results: useQueries hands back a new
  // array on every render, so a memo keyed on it would never hold and this hook's
  // returned Map would change identity every render -- which in turn defeats the
  // useCallback around the table's makeColumns, since it depends on that Map.
  const combineProviders = useCallback(
    (results: UseQueryResult<ModelProvider>[]) => {
      const map = new Map<
        string,
        { provider?: ModelProvider; isLoading: boolean; isError: boolean }
      >();
      providerRefs.forEach((ref, index) => {
        const query = results[index];
        map.set(ref, {
          provider: query?.data,
          isLoading: Boolean(query?.isLoading),
          isError: Boolean(query?.isError),
        });
      });
      return map;
    },
    [providerRefs]
  );

  const providersByRef = useQueries({
    queries: providerRefs.map((ref) => {
      const parts = getPartsFromReference(ref);
      return {
        queryKey: getModelsGetProviderQueryKey(parts.workspace, parts.name),
        queryFn: () => modelsGetProvider(parts.workspace, parts.name),
        retry: false,
        staleTime: PROVIDER_STALE_TIME,
      };
    }),
    combine: combineProviders,
  });

  /** For each target, which provider serves it, and which serves its base. */
  const matches = useMemo(
    () =>
      targets.map((target) => {
        const refs = target.model.model_providers ?? [];
        const targetId = toInferenceModelEntityId(target.model, target.adapter);
        const baseId = toInferenceModelEntityId(target.model);

        let servesTarget: ModelProvider | undefined;
        let servesBase = false;
        let isLoading = false;
        let isError = false;

        for (const ref of refs) {
          const entry = providersByRef.get(ref);
          if (!entry) continue;
          if (entry.isLoading) isLoading = true;
          if (entry.isError) isError = true;
          const served = entry.provider?.served_models ?? [];
          if (!servesTarget && served.some((sm) => sm.model_entity_id === targetId)) {
            servesTarget = entry.provider;
          }
          if (!servesBase && served.some((sm) => sm.model_entity_id === baseId)) {
            servesBase = true;
          }
        }

        return { target, servesTarget, servesBase, isLoading, isError: isError && !servesTarget };
      }),
    [targets, providersByRef]
  );

  const deploymentRefs = useMemo(() => {
    const refs = new Set<string>();
    for (const { servesTarget } of matches) {
      if (servesTarget?.model_deployment_id) refs.add(servesTarget.model_deployment_id);
    }
    return [...refs].sort();
  }, [matches]);

  const combineDeployments = useCallback(
    (results: UseQueryResult<ModelDeployment>[]) => {
      const map = new Map<
        string,
        { status?: ModelDeploymentStatus; statusMessage?: string; isLoading: boolean }
      >();
      deploymentRefs.forEach((ref, index) => {
        const query = results[index];
        map.set(ref, {
          status: query?.data?.status,
          statusMessage: query?.data?.status_message,
          isLoading: Boolean(query?.isLoading),
        });
      });
      return map;
    },
    [deploymentRefs]
  );

  const deploymentsByRef = useQueries({
    queries: deploymentRefs.map((ref) => {
      const parts = getPartsFromReference(ref);
      return {
        queryKey: getModelsGetLatestDeploymentQueryKey(parts.workspace, parts.name),
        queryFn: () => modelsGetLatestDeployment(parts.workspace, parts.name),
        retry: false,
      };
    }),
    combine: combineDeployments,
  });

  return useMemo(() => {
    const states = new Map<string, DeploymentIndicatorState>();

    for (const { target, servesTarget, servesBase, isLoading, isError } of matches) {
      const hasProviders = (target.model.model_providers ?? []).length > 0;

      if (hasProviders && isLoading) {
        states.set(target.key, { kind: 'loading' });
        continue;
      }

      if (servesTarget) {
        const deploymentRef = servesTarget.model_deployment_id;
        const deployment = deploymentRef ? deploymentsByRef.get(deploymentRef) : undefined;
        if (deploymentRef && deployment?.isLoading) {
          states.set(target.key, { kind: 'loading' });
          continue;
        }
        states.set(target.key, {
          kind: 'served',
          hasDeployment: Boolean(deploymentRef),
          status: deployment?.status,
          statusMessage: deployment?.statusMessage,
          providerRef: `${servesTarget.workspace}/${servesTarget.name}`,
        });
        continue;
      }

      // A provider we could not read might be the one serving this id, so nothing
      // below may assert otherwise — including the base-model match, since with
      // several providers one can serve the base while the unread one serves the
      // adapter.
      if (isError) {
        states.set(target.key, { kind: 'unknown' });
        continue;
      }

      if (target.adapter && servesBase) {
        states.set(target.key, { kind: 'adapter-not-loaded' });
        continue;
      }

      states.set(target.key, { kind: 'not-deployed' });
    }

    return states;
  }, [matches, deploymentsByRef]);
}
