// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Submit orchestration for the Create Dataset panel (spec 11.3, 13): create the fileset, upload the
// staged files for the Upload tab, and surface failures without losing the user's input. A request
// is never cancelled: if the panel closes while it runs, it finishes and reports by toast. An
// upload that fails rolls the new fileset back (deletes it), so a failure leaves nothing behind
// and a retry starts clean; if the rollback itself fails, the toast says the dataset may remain.

import { getErrorMessage, isConflictError } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  filesDeleteFileset,
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
}

const failureMessage = (err: unknown, fallback: string) =>
  err instanceof Error ? getErrorMessage(err, fallback) : fallback;

/** Drops the caches a new fileset and its files make stale. Run once its uploads (and any rollback) have settled. */
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
  // Counts closes. A request compares it with the value it started under to learn the panel closed
  // (and maybe reopened) meanwhile, so nothing it finishes can leak into a later session.
  const closes = useRef(0);

  useEffect(() => {
    // Runs on mount too: a panel that starts closed just counts one close, which is harmless.
    if (!open) {
      closes.current += 1;
      setIsPending(false);
    }
  }, [open]);

  const submit = useCallback<UseCreateDatasetResult['submit']>(
    async (request, uploads) => {
      const startedAt = closes.current;
      const isClosed = () => closes.current !== startedAt;
      setIsPending(true);
      try {
        let fileset: FilesetOutput;
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

        if (uploads) {
          try {
            await uploadFilesetEntries(workspace, fileset.name, uploads);
          } catch (err) {
            // Roll back so the failed attempt leaves no half-filled dataset (and the name stays free).
            const rolledBack = await filesDeleteFileset(workspace, fileset.name).then(
              () => true,
              () => false
            );
            resetFilesetCaches(queryClient, fileset);
            toast.error(
              rolledBack
                ? failureMessage(err, 'Failed to upload files')
                : `Upload failed and the dataset ${fileset.name} may still exist. Delete it manually.`
            );
            return { status: 'failed' };
          }
        }
        resetFilesetCaches(queryClient, fileset);

        if (isClosed()) {
          toast.success(<CreatedDatasetToast workspace={workspace} name={fileset.name} />);
          return { status: 'created-after-close' };
        }
        return { status: 'created', fileset };
      } finally {
        if (closes.current === startedAt) setIsPending(false);
      }
    },
    [createFileset, queryClient, toast, workspace]
  );

  return { submit, isPending };
}
