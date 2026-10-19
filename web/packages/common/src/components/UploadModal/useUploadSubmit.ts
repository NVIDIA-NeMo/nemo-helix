// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import type { SubmitUploadType } from '@nemo/common/src/components/UploadModal/types';
import { validateUploadForm } from '@nemo/common/src/components/UploadModal/validation';
import { filesCreateFileset, filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { useCallback, useRef } from 'react';

/**
 * Constructs the Fileset file URL.
 * Format: ``fileset://{workspace}/{name}/{filepath}``
 */
const getFilesetFileUrl = (fileset: FilesetOutput, filepath: string): string =>
  `fileset://${fileset.workspace}/${fileset.name}/${filepath}`;

interface UseUploadSubmitParams {
  workspace: string;
  includeDataset: boolean;
  includeTabs: boolean;
  /** Async-capable: ``submit`` awaits before returning so callers can finish
   *  blob-URL bookkeeping / form commits before the modal closes. */
  onSubmit: (data: SubmitUploadType) => void | Promise<void>;
}

/** Encapsulates the upload-and-commit flow used by both the modal and the
 *  inline variant of the picker: validate selection, create/lookup the
 *  fileset, upload any new files, then call ``onSubmit`` with the resulting
 *  dataset reference.
 *
 *  ``submit`` is stable across renders — internal state (``dataset``,
 *  ``selectedFiles``, ``activeTab``) is read lazily through a ref at call
 *  time, so consumers can put ``submit`` in effect dependency arrays without
 *  causing extra reads. ``isSubmitting`` is returned eagerly for components
 *  that bind a loading state to it. */
export const useUploadSubmit = ({
  workspace,
  includeDataset,
  includeTabs,
  onSubmit,
}: UseUploadSubmitParams) => {
  const [state, dispatch] = useUploadModalContext();
  const stateRef = useRef(state);
  stateRef.current = state;

  const onSubmitRef = useRef(onSubmit);
  onSubmitRef.current = onSubmit;

  const submit = useCallback(async (): Promise<boolean> => {
    const { dataset, selectedFiles, activeTab } = stateRef.current;
    const onSubmitFn = onSubmitRef.current;

    const errors = validateUploadForm({
      selectedFiles,
      selectedDataset: dataset,
      isDatasetRequired: includeDataset || (includeTabs && activeTab === 'dataset'),
    });
    if (Object.keys(errors).length > 0) {
      dispatch({ type: 'SET_ERRORS', payload: errors });
      return false;
    }

    dispatch({ type: 'SET_SUBMITTING', payload: true });
    try {
      if (!includeDataset && includeTabs && activeTab === 'file') {
        await onSubmitFn({
          type: 'file',
          files: selectedFiles.map((file) => file.file as File),
        });
        return true;
      }

      const newFiles = selectedFiles.filter((file) => file.type === 'new').map((file) => file.file);

      if (dataset && dataset.type === 'new' && newFiles.length > 0) {
        const newDataset = await filesCreateFileset(workspace, {
          name: dataset.name,
          description: undefined,
          project: workspace,
          purpose: 'dataset',
        });
        await Promise.all(
          newFiles.map((file) =>
            filesUploadFile(newDataset.workspace, newDataset.name, file.name, file)
          )
        );
        await onSubmitFn({
          type: 'dataset',
          dataset: newDataset,
          path: newFiles[0].name,
          url: getFilesetFileUrl(newDataset, newFiles[0].name),
        });
      } else if (dataset && dataset.type === 'existing') {
        if (newFiles.length > 0) {
          await Promise.all(
            newFiles.map((file) =>
              filesUploadFile(dataset.dataset.workspace, dataset.dataset.name, file.name, file)
            )
          );
          await onSubmitFn({
            type: 'dataset',
            dataset: dataset.dataset,
            path: newFiles[0].name,
            url: getFilesetFileUrl(dataset.dataset, newFiles[0].name),
          });
        } else {
          const existingFile = selectedFiles[0].file as FilesetFileOutput;
          await onSubmitFn({
            type: 'dataset',
            dataset: dataset.dataset,
            path: existingFile.path,
            url: getFilesetFileUrl(dataset.dataset, existingFile.path),
          });
        }
      }
      return true;
    } catch (error) {
      console.error('UploadModal: Error during submission', error);
      dispatch({
        type: 'SET_ERRORS',
        payload: { file: 'Error during submission. Please try again.' },
      });
      return false;
    } finally {
      dispatch({ type: 'SET_SUBMITTING', payload: false });
    }
  }, [workspace, includeDataset, includeTabs, dispatch]);

  return { submit, isSubmitting: state.isSubmitting };
};
