// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Submit orchestration for the Create Dataset panel (spec 11.3, 13): create the fileset, upload the
// staged files for the Upload tab, and surface failures without losing the user's input. A request
// is never cancelled: if the panel closes while it runs, it finishes and reports by toast.

import { getErrorMessage, isConflictError } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
  useFilesCreateFileset,
} from '@nemo/sdk/generated/platform/files';
import type { CreateFilesetRequest, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import type { FilesetEntry } from '@studio/api/files/types';
import { uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';
import { CreatedDatasetToast } from '@studio/components/CreateDatasetPanel/CreatedDatasetToast';
import { invalidateFilesetNameCheck } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { type QueryClient, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';

export type SubmitOutcome =
  /** Created while the panel was open: the caller reports it and closes. */
  | { status: 'created'; fileset: FilesetOutput }
  /** Created after the panel closed: already announced by toast, nothing left to do. */
  | { status: 'created-after-close' }
  /** The server already has this name; the caller flags the name field. */
  | { status: 'conflict' }
  /** Failed and already reported by toast. */
  | { status: 'failed' };

export interface UseCreateDatasetResult {
  /** `uploads` is the Upload tab's staged files; omit it for an external source. */
  submit: (
    request: CreateFilesetRequest,
    uploads?: readonly FilesetEntry[]
  ) => Promise<SubmitOutcome>;
  isPending: boolean;
  /** An Upload fileset this panel created whose files then failed to upload, if any. */
  createdName: string | undefined;
}

const failureMessage = (err: unknown, fallback: string) =>
  err instanceof Error ? getErrorMessage(err, fallback) : fallback;

/** Drops the caches a new fileset and its files make stale. Run once its uploads have settled. */
function resetFilesetCaches(queryClient: QueryClient, fileset: FilesetOutput) {
  const { workspace, name } = fileset;
  void queryClient.resetQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) });
  void queryClient.resetQueries({ queryKey: getFilesRetrieveFilesetQueryKey(workspace, name) });
  void queryClient.resetQueries({ queryKey: getFilesListFilesetFilesQueryKey(workspace, name) });
}

export function useCreateDataset(workspace: string, open: boolean): UseCreateDatasetResult {
  const toast = useToast();
  const queryClient = useQueryClient();
  const { mutateAsync: createFileset } = useFilesCreateFileset();
  const [isPending, setIsPending] = useState(false);
  // An Upload fileset whose files then failed to upload. It exists server-side, so a retry resumes
  // the upload rather than creating it again (which would 409), and the name check must not
  // count it against itself.
  const [createdFileset, setCreatedFileset] = useState<FilesetOutput>();
  // Counts closes. A request compares it with the value it started under to learn the panel closed
  // (and maybe reopened) meanwhile, so nothing it finishes can leak into a later session.
  const closes = useRef(0);

  useEffect(() => {
    // Runs on mount too: a panel that starts closed just counts one close, which is harmless.
    if (!open) {
      closes.current += 1;
      setIsPending(false);
      setCreatedFileset(undefined);
    }
  }, [open]);

  const submit = useCallback<UseCreateDatasetResult['submit']>(
    async (request, uploads) => {
      const startedAt = closes.current;
      const isClosed = () => closes.current !== startedAt;
      setIsPending(true);
      try {
        let fileset = uploads && createdFileset?.name === request.name ? createdFileset : undefined;
        if (!fileset) {
          try {
            fileset = await createFileset({ workspace, data: request });
          } catch (err) {
            if (isConflictError(err)) {
              // Lost a race with another create: have the name check ask the server again.
              void invalidateFilesetNameCheck(queryClient, workspace, request.name);
              if (!isClosed()) return { status: 'conflict' };
            }
            toast.error(failureMessage(err, 'Failed to create dataset'));
            return { status: 'failed' };
          }
          if (uploads && !isClosed()) setCreatedFileset(fileset);
        }

        if (uploads) {
          try {
            await uploadFilesetEntries(workspace, fileset.name, uploads);
          } catch (err) {
            resetFilesetCaches(queryClient, fileset);
            toast.error(failureMessage(err, 'Failed to upload files'));
            return { status: 'failed' };
          }
        }
        resetFilesetCaches(queryClient, fileset);

        if (isClosed()) {
          toast.success(<CreatedDatasetToast workspace={workspace} name={fileset.name} />);
          return { status: 'created-after-close' };
        }
        setCreatedFileset(undefined);
        return { status: 'created', fileset };
      } finally {
        if (closes.current === startedAt) setIsPending(false);
      }
    },
    [createFileset, createdFileset, queryClient, toast, workspace]
  );

  return { submit, isPending, createdName: createdFileset?.name };
}
