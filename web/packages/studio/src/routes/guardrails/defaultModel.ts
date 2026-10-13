// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { ResourceRef } from '@nemo/common/src/types';
import { hasModelProvider } from '@nemo/common/src/utils/models';
import { DEFAULT_BUILD_MODEL_NAME } from '@studio/constants/constants';

/**
 * Strip the workspace prefix and version suffix so a preferred name matches across
 * workspaces — the URN's prefix varies per user, and `DEFAULT_BUILD_MODEL_NAME` is written
 * without a version.
 */
const bareName = (value: string): string => (value.split('/').pop() ?? value).split('@')[0];

/**
 * A model to seed a new guardrail config's `main` entry with, as a URN.
 *
 * Prefers {@link DEFAULT_BUILD_MODEL_NAME} when the workspace serves it, otherwise the first
 * available model. Returns null when nothing qualifies — the Configuration tab then shows
 * an empty required field, which is better than seeding a name that fails at run time.
 *
 * Models with no `model_providers` are skipped: no provider means no deployment reached
 * READY, so seeding one hands the user a config whose only symptom is a failed test run.
 */
export const resolveDefaultGuardrailModel = (
  groups: ModelWorkspaceGroup[],
  preferred: string = DEFAULT_BUILD_MODEL_NAME
): string | null => {
  const usable = groups
    .flatMap((group) => group.models)
    .filter(hasModelProvider)
    .map((entity) => ({ entity, urn: getURNFromNamedEntityRef(entity) }))
    .filter((candidate): candidate is { entity: (typeof candidate)['entity']; urn: ResourceRef } =>
      Boolean(candidate.urn)
    );

  const wanted = bareName(preferred);
  const match = usable.find(
    ({ entity, urn }) =>
      urn === preferred || entity.name === preferred || bareName(entity.name ?? '') === wanted
  );

  return (match ?? usable[0])?.urn ?? null;
};
