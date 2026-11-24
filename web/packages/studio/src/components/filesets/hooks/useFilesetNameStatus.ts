// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  getFilesListFilesetsQueryKey,
  useFilesListFilesets,
} from '@nemo/sdk/generated/platform/files';
import type { FilesListFilesetsParams } from '@nemo/sdk/generated/platform/schema';
import { nameCheckStatus, type NameCheckStatus } from '@studio/components/form/entityNameCheck';
import type { QueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { type Control, type FieldValues, type Path, useWatch } from 'react-hook-form';
import { useDebounce } from 'use-debounce';

const nameCheckParams = (name: string): FilesListFilesetsParams => ({
  page_size: 1,
  filter: { name },
});

/** The sanitized name when the typed input is salvageable, otherwise ''. A salvageable input
 *  has at least one letter or number, producing a valid entity name after sanitization.
 *  An unsalvageable input already shows a format error, so there is nothing to ask the server. */
export const strictFilesetNameCandidate = (name: string | undefined): string => {
  const trimmed = name?.trim() ?? '';
  return sanitizeEntityName(trimmed) !== undefined ? toValidEntityName(trimmed, '') : '';
};

/** Drops the cached check for `name`, so a name the server just rejected as taken is
 *  checked again rather than still reading as free. */
export const invalidateFilesetNameCheck = (
  queryClient: QueryClient,
  workspace: string,
  name: string
): Promise<void> =>
  queryClient.invalidateQueries({
    queryKey: getFilesListFilesetsQueryKey(workspace, nameCheckParams(name)),
  });

/** Whether the fileset name `candidate` is free in `workspace`; pass '' to skip the check.
 *  `ownName` is a fileset this form already created, so finding it is not a conflict. */
export function useFilesetNameStatus(
  workspace: string,
  candidate: string,
  { ownName }: { ownName?: string } = {}
): NameCheckStatus {
  const [debouncedName] = useDebounce(candidate, DEFAULT_DEBOUNCE_MS);
  const conflictQuery = useFilesListFilesets(workspace, nameCheckParams(debouncedName), {
    query: {
      enabled: Boolean(workspace && debouncedName) && debouncedName !== ownName,
      // A conflict blocks submit, so a cached answer must not outlive the fileset it saw, and
      // a failed check should say so now rather than after the default retry backoff.
      staleTime: 0,
      retry: false,
    },
  });

  if (ownName && candidate === ownName) return undefined;
  return nameCheckStatus(candidate, debouncedName, conflictQuery);
}

interface UseWatchedFilesetNameStatusOptions<T extends FieldValues> {
  control: Control<T>;
  name: Path<T>;
  workspace: string;
  ownName?: string;
  onStatusChange: (status: NameCheckStatus) => void;
}

/** Checks a form's fileset name field as it changes. Watching it here, not at the form root,
 *  re-renders only the caller per keystroke; the form re-renders only when the status changes. */
export function useWatchedFilesetNameStatus<T extends FieldValues>({
  control,
  name,
  workspace,
  ownName,
  onStatusChange,
}: UseWatchedFilesetNameStatusOptions<T>): { candidate: string; status: NameCheckStatus } {
  const value: unknown = useWatch({ control, name });
  const candidate = strictFilesetNameCandidate(typeof value === 'string' ? value : undefined);
  const status = useFilesetNameStatus(workspace, candidate, { ownName });

  useEffect(() => {
    onStatusChange(status);
  }, [status, onStatusChange]);

  return { candidate, status };
}
