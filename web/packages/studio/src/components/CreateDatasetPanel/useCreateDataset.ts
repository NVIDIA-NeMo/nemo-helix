// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Submit orchestration for the Create Dataset panel (spec 11.3, 13): create the fileset, upload the
// staged files for the Upload tab, and surface failures without losing the user's input.

import { getErrorMessage, isConflictError } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getFilesListFilesetsQueryKey,
  useFilesCreateFileset,
} from '@nemo/sdk/generated/platform/files';
import type { CreateFilesetRequest, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import type { FilesetEntry } from '@studio/api/files/types';
import { uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';
import { invalidateFilesetNameCheck } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';

export type SubmitOutcome =
  | { status: 'created'; fileset: FilesetOutput }
  /** The server already has this name; the caller flags the name field. */
  | { status: 'conflict' }
  /** Failed and already reported by toast. */
  | { status: 'failed' };

const failureMessage = (err: unknown, fallback: string) =>
  err instanceof Error ? getErrorMessage(err, fallback) : fallback;

export function useCreateDataset(workspace: string) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const { mutateAsync: createFileset } = useFilesCreateFileset();
  const [isPending, setIsPending] = useState(false);
  // An Upload fileset whose files then failed to upload. It exists server-side, so a retry resumes
  // the upload rather than creating it again (which would 409), and the name check must not
  // count it against itself.
  const [createdFileset, setCreatedFileset] = useState<FilesetOutput>();

  const forgetCreated = useCallback(() => setCreatedFileset(undefined), []);

  /** `uploads` is the Upload tab's staged files; omit it for an external source. */
  const submit = useCallback(
    async (
      request: CreateFilesetRequest,
      uploads?: readonly FilesetEntry[]
    ): Promise<SubmitOutcome> => {
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
              return { status: 'conflict' };
            }
            toast.error(failureMessage(err, 'Failed to create dataset'));
            return { status: 'failed' };
          }
          void queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) });
          if (uploads) setCreatedFileset(fileset);
        }

        if (uploads) {
          try {
            await uploadFilesetEntries(workspace, fileset.name, uploads);
          } catch (err) {
            toast.error(failureMessage(err, 'Failed to upload files'));
            return { status: 'failed' };
          }
        }
        return { status: 'created', fileset };
      } finally {
        setIsPending(false);
      }
    },
    [createFileset, createdFileset, queryClient, toast, workspace]
  );

  return { submit, isPending, createdName: createdFileset?.name, forgetCreated };
}
