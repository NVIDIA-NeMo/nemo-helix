// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { resourceRefSchema, type ResourceRef } from '@nemo/common/src/types';

export interface NamedEntity {
  workspace?: string;
  /** @deprecated Use workspace instead */
  namespace?: string;
  name?: string;
}

interface GetEntityFullNameOpts {
  encode?: boolean; // Encode as URI component for route params
}

/**
 * Given an entity type that has keys for workspace/namespace and name, get the unique identifier
 * as a concatenated string joined by '/'. e.g. "workspace/name"
 * @returns {String} - Joined string of workspace and name
 */
export const getEntityReference = (
  entityRef?: NamedEntityRef,
  opts?: GetEntityFullNameOpts
): string => {
  if (!entityRef) return '';

  let base: string;
  if (typeof entityRef === 'string') {
    base = entityRef;
  } else {
    // Prefer workspace, fall back to namespace for backward compatibility
    const workspace = entityRef.workspace ?? entityRef.namespace ?? DEFAULT_WORKSPACE;
    const name = entityRef.name ?? '';
    base = `${workspace}/${name}`;
  }

  return opts?.encode ? encodeURIComponent(base) : base;
};

export interface NamedEntityParts {
  workspace: string;
  /** @deprecated Use workspace instead */
  namespace: string;
  name: string;
}

export const getPartsFromReference = (id: string): NamedEntityParts => {
  let usedId = id;
  if (id !== decodeURIComponent(id)) {
    usedId = decodeURIComponent(id);
  }
  const [workspace, name] = usedId.split('/');
  const ws = workspace ?? '';
  return { workspace: ws, namespace: ws, name: name ?? '' };
};

export type NamedEntityRef = string | NamedEntity;

export const getPartsFromNamedEntityRef = (ref: NamedEntityRef): NamedEntityParts => {
  if (typeof ref === 'string') {
    return getPartsFromReference(ref);
  }
  const ws = ref.workspace ?? ref.namespace ?? '';
  return {
    workspace: ws,
    namespace: ws,
    name: ref.name ?? '',
  };
};

export const getURNFromNamedEntityRef = (
  ref?: NamedEntityRef,
  opts?: GetEntityFullNameOpts
): ResourceRef | undefined => {
  if (!ref) return undefined;
  const result = getEntityReference(ref, opts);
  const parsed = resourceRefSchema.safeParse(result);
  return parsed.success ? parsed.data : undefined;
};
