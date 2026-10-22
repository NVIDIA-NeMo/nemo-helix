// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { nameTakenMessage } from '@studio/components/form/entityNameCheck';

export const filesetNameTakenMessage = (name: string, workspace: string): string =>
  nameTakenMessage('fileset', name, workspace);

export const retrieveFilesetIfPresent = async (
  workspace: string,
  name: string
): Promise<FilesetOutput | null> => {
  try {
    return await filesRetrieveFileset(workspace, name);
  } catch (e) {
    if (isNotFoundError(e)) return null;
    throw e;
  }
};

/** How many numbered copies (`name-2` … `name-20`) a free-name lookup tries. */
export const FILESET_NAME_SUFFIX_LIMIT = 20;

/** First free `name-2`, `name-3`, …, or null when every one up to `limit` is taken. */
const findFreeSuffixedName = async (
  workspace: string,
  base: string,
  limit: number
): Promise<string | null> => {
  for (let suffix = 2; suffix <= limit; suffix += 1) {
    const candidate = `${base}-${suffix}`;
    if (!(await retrieveFilesetIfPresent(workspace, candidate))) return candidate;
  }
  return null;
};

/** First free `name-2`, `name-3`, …. Falls back to the last candidate rather than throwing,
 *  so the create that follows reports the collision. */
export const findAvailableFilesetName = async (
  workspace: string,
  base: string,
  limit = FILESET_NAME_SUFFIX_LIMIT
): Promise<string> => (await findFreeSuffixedName(workspace, base, limit)) ?? `${base}-${limit}`;

/** `base` when it is free in `workspace`, otherwise the first free suffixed name, or null when
 *  every suffix is taken too. */
export const findFreeFilesetName = async (
  workspace: string,
  base: string
): Promise<string | null> =>
  (await retrieveFilesetIfPresent(workspace, base))
    ? findFreeSuffixedName(workspace, base, FILESET_NAME_SUFFIX_LIMIT)
    : base;
