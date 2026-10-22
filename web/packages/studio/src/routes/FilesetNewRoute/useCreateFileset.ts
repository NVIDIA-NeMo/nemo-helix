// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getErrorMessage as getApiErrorMessage,
  isConflictError,
} from '@nemo/common/src/api/common/utils';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  filesUploadFile,
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
  useFilesCreateFileset,
} from '@nemo/sdk/generated/platform/files';
import {
  FilesetOutput,
  FilesetPurpose,
  CreateFilesetRequest,
} from '@nemo/sdk/generated/platform/schema';
import { filesetNameTakenMessage } from '@studio/components/filesets/filesetNames';
import { invalidateFilesetNameCheck } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
import { DATASET_TYPE_SAMPLE } from '@studio/routes/FilesetNewRoute/constants';
import {
  getExternalStorageCreateErrorMessage,
  toFileList,
} from '@studio/routes/FilesetNewRoute/helpers';
import {
  DatasetCreateFilesetFormSchema,
  DatasetFormFields,
  DatasetType,
} from '@studio/routes/FilesetNewRoute/types';
import { getFilesetDetailRoute, getFilesetDetailsRoute } from '@studio/routes/utils';
import { storageConfigFromUrl } from '@studio/util/storageConfigFromUrl';
import { QueryObserverResult, useQueryClient } from '@tanstack/react-query';
import { MutableRefObject, useCallback } from 'react';
import type { UseFormGetValues, UseFormSetError } from 'react-hook-form';
import { useNavigate } from 'react-router';

interface UseCreateFilesetParams {
  workspace: string;
  activeTab: DatasetType;
  storageTab: 'local' | 'external';
  hasValidationErrors: boolean;
  /** The name's uniqueness check found a fileset already using it. */
  nameTaken: boolean;
  getValues: UseFormGetValues<DatasetFormFields>;
  setError: UseFormSetError<DatasetFormFields>;
  sampleFilesRef: MutableRefObject<Promise<QueryObserverResult<File[], Error>> | null>;
  setIsSubmitPending: (value: boolean) => void;
  /** A fileset this form created whose upload then failed. */
  createdFileset?: FilesetOutput;
  /** Called with the fileset once it exists, before its files upload. */
  onCreated: (fileset: FilesetOutput) => void;
}

export function useCreateFileset({
  workspace,
  activeTab,
  storageTab,
  hasValidationErrors,
  nameTaken,
  getValues,
  setError,
  sampleFilesRef,
  setIsSubmitPending,
  createdFileset,
  onCreated,
}: UseCreateFilesetParams): (data: DatasetFormFields) => Promise<void> {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();

  const { mutateAsync: createFileset } = useFilesCreateFileset({
    mutation: {
      onSuccess: (fileset) => {
        queryClient.resetQueries({ queryKey: getFilesListFilesetsQueryKey(fileset.workspace) });
        queryClient.resetQueries({
          queryKey: getFilesRetrieveFilesetQueryKey(fileset.workspace, fileset.name),
        });
        queryClient.resetQueries({
          queryKey: getFilesListFilesetFilesQueryKey(fileset.workspace, fileset.name),
        });
      },
    },
  });

  /** Step: create fileset. Present failures to user in caller. */
  const createFilesetStep = useCallback(
    async (workspace: string, data: CreateFilesetRequest): Promise<FilesetOutput> => {
      const fileset: FilesetOutput = await createFileset({
        workspace,
        data: {
          name: data.name,
          description: data.description ?? '',
          project: data.project,
          storage: data.storage ?? undefined,
          purpose: data.purpose ?? FilesetPurpose.generic,
          metadata: data.metadata,
          custom_fields: data.custom_fields,
          cache: data.cache,
        },
      });
      return fileset;
    },
    [createFileset]
  );

  /** Step: upload files to dataset. Present failures to user in caller. */
  const uploadFilesToDatasetStep = useCallback(
    async (workspace: string, fileset: FilesetOutput, files: File[]): Promise<void> => {
      await Promise.all(
        files.map(async (file) => {
          const blob = new Blob([await file.arrayBuffer()], {
            type: file.type || 'application/octet-stream',
          });
          return filesUploadFile(workspace, fileset.name, file.name, blob);
        })
      );
    },
    []
  );

  const onSubmit = useCallback(
    async (data: DatasetFormFields) => {
      const { success, error } = DatasetCreateFilesetFormSchema.safeParse(data);
      if (!success) {
        toast.error(error.message);
        return;
      }

      if (hasValidationErrors) {
        toast.error('Fix dataset validation errors before creating this fileset.');
        return;
      }

      if (nameTaken) return;

      setIsSubmitPending(true);

      // Step 1 (sample/local only): fetch sample files or read local file selection
      let files: File[] = [];
      if (activeTab === DATASET_TYPE_SAMPLE) {
        if (!sampleFilesRef.current) {
          toast.error('No sample files could be loaded.');
          setIsSubmitPending(false);
          return;
        }
        const result = (await sampleFilesRef.current) as QueryObserverResult<File[], Error>;
        if (result.isError || result.error) {
          toast.error(getApiErrorMessage(result.error as Error, 'Failed to load sample files'));
          setIsSubmitPending(false);
          return;
        }
        if (!result.data?.length) {
          toast.error('No sample files could be loaded.');
          setIsSubmitPending(false);
          return;
        }
        files = result.data;
      } else if (storageTab === 'local') {
        files = toFileList(getValues('files'));
      }

      // Step 2: create fileset. Sample tab always produces a dataset-purpose fileset
      // (preconfigured samples are training/eval data by definition); Custom tab
      // uses whatever the user picked in the Purpose selector.
      const effectivePurpose =
        activeTab === DATASET_TYPE_SAMPLE ? FilesetPurpose.dataset : data.purpose;
      let createPayload: CreateFilesetRequest = {
        name: data.name,
        description: data.description ?? '',
        project: data.project,
        storage: data.storage ?? undefined,
        purpose: effectivePurpose,
        metadata: data.metadata,
        custom_fields: data.custom_fields,
        cache: data.cache,
      };
      const url = getValues('url');
      const secretRef = getValues('secretKey')?.trim() || undefined;
      if (storageTab === 'external' && url?.trim()) {
        try {
          createPayload = {
            ...createPayload,
            storage: storageConfigFromUrl({
              url: url.trim(),
              secretKey: secretRef,
            }),
          };
        } catch (e) {
          toast.error(
            e instanceof Error
              ? e.message
              : 'Invalid external storage URL or credential. For NGC, select a secret with your API key.'
          );
          setIsSubmitPending(false);
          return;
        }
      }

      // A retry after a failed upload resumes the upload: creating the fileset again would 409
      // on the one this form already made, and its files could never be uploaded.
      let fileset: FilesetOutput;
      if (createdFileset?.name === createPayload.name) {
        fileset = createdFileset;
      } else {
        try {
          fileset = await createFilesetStep(workspace, createPayload);
        } catch (err) {
          setIsSubmitPending(false);
          // The name check can lose a race with another create. The sample tab has no name
          // field to put the error on, so it toasts; the recheck then moves it to a free name.
          if (isConflictError(err)) {
            void invalidateFilesetNameCheck(queryClient, workspace, createPayload.name);
            const message = filesetNameTakenMessage(createPayload.name, workspace);
            if (activeTab === DATASET_TYPE_SAMPLE) toast.error(message);
            else setError('name', { message });
            return;
          }
          const message =
            storageTab === 'external' && url?.trim()
              ? getExternalStorageCreateErrorMessage(err, url.trim())
              : getApiErrorMessage(err as Error, 'Failed to create fileset');
          toast.error(message);
          return;
        }
        onCreated(fileset);
      }

      // Step 3: upload files to dataset
      if (files.length) {
        try {
          await uploadFilesToDatasetStep(workspace, fileset, files);
        } catch (err) {
          toast.error(getApiErrorMessage(err as Error, 'Failed to upload files'));
          setIsSubmitPending(false);
          return;
        }
      }

      setIsSubmitPending(false);
      if (
        FILESET_DETAILS_ENABLED &&
        (fileset.purpose === FilesetPurpose.dataset || fileset.purpose === FilesetPurpose.model)
      ) {
        navigate(getFilesetDetailRoute(workspace, fileset.name));
        return;
      }
      navigate(getFilesetDetailsRoute(workspace, getEntityReference(fileset), undefined, true));
    },
    [
      activeTab,
      createFilesetStep,
      createdFileset,
      getValues,
      hasValidationErrors,
      nameTaken,
      navigate,
      onCreated,
      queryClient,
      sampleFilesRef,
      setError,
      setIsSubmitPending,
      storageTab,
      toast,
      uploadFilesToDatasetStep,
      workspace,
    ]
  );

  return onSubmit;
}
