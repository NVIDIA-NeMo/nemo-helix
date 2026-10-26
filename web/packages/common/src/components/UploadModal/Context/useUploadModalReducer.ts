/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { UploadFile, UploadDataset } from '@nemo/common/src/components/UploadModal/types';
import {
  isFileTypeAllowed,
  sanitizeFilenameForDatasetName,
} from '@nemo/common/src/components/UploadModal/utils';
import type { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { useReducer } from 'react';

/**
 * Controls how files whose extension isn't in ``acceptableFileTypes`` appear
 * in the picker.
 *
 * - ``'show'`` (default): render every file; rely on parent-side validation
 *   after the user submits a selection. Backwards-compatible behaviour.
 * - ``'hide'``: filter the file list down to only allowed extensions.
 * - ``'disable'``: render every file but mark mismatches as disabled so the
 *   user can see what's there but can't select it.
 */
export type InvalidFileMode = 'show' | 'hide' | 'disable';

/**
 * State for the UploadModal component
 */
export type UploadModalState = {
  files: UploadFile[];
  dataset: UploadDataset | undefined;
  selectedFiles: UploadFile[];
  activeTab: 'dataset' | 'file';
  isSubmitting: boolean;
  isFetching: boolean;
  allowMultipleFileSelection: boolean;
  acceptableFileTypes: string[];
  acceptableFileSize: number;
  invalidFileMode: InvalidFileMode;
  /** When false, the dataset picker hides the "Create new dataset" option so
   *  consumers can restrict selection to existing filesets — e.g. the inline
   *  picker in ``autoCommit`` mode where uploading new files would race with
   *  the user editing the dataset name. Defaults to ``true``. */
  allowNewDataset: boolean;
  /** Fileset ``purpose`` the picker lists. Defaults to ``'dataset'``; ``null`` lists every purpose. */
  filesetPurpose?: FilesetPurpose | null;
  /** Label for the fileset picker. Defaults to ``'Dataset'``. */
  datasetLabel?: string;
  /** Auto-select the first root-level accepted file on fileset selection. */
  autoSelectFirstAcceptable?: boolean;
  showUpdatedAt?: boolean;
  errors: Record<string, string>;
};

/**
 * Actions that can be dispatched to update the UploadModal state
 */
export type UploadModalAction =
  | { type: 'RESET' }
  | { type: 'SET_FILES'; payload: UploadFile[] }
  | { type: 'UPDATE_DATASET'; payload: UploadDataset }
  | { type: 'SET_DATASET'; payload: UploadDataset | undefined }
  | { type: 'TOGGLE_FILE_SELECTION'; payload: UploadFile }
  | { type: 'SET_TAB'; payload: 'dataset' | 'file' }
  | { type: 'SET_SUBMITTING'; payload: boolean }
  | { type: 'SET_FETCHING'; payload: boolean }
  | { type: 'SET_ERRORS'; payload: Record<string, string> }
  | { type: 'CLEAR_ERRORS' }
  | { type: 'SET_ALLOW_MULTIPLE_FILE_SELECTION'; payload: boolean };

/**
 * Initial state for the UploadModal
 */
export const initialState: UploadModalState = {
  files: [],
  dataset: undefined,
  selectedFiles: [],
  activeTab: 'dataset',
  isSubmitting: false,
  isFetching: false,
  allowMultipleFileSelection: false,
  acceptableFileTypes: ['.json', '.jsonl'],
  acceptableFileSize: 50 * 1024 * 1024, // 50 MB
  invalidFileMode: 'show',
  allowNewDataset: true,
  errors: {},
};

/**
 * Reducer function for managing UploadModal state transitions
 */
export const uploadModalReducer = (
  state: UploadModalState,
  action: UploadModalAction
): UploadModalState => {
  switch (action.type) {
    case 'RESET':
      // Preserve configuration props that were passed as props to UploadModal
      return {
        ...initialState,
        acceptableFileTypes: state.acceptableFileTypes,
        acceptableFileSize: state.acceptableFileSize,
        allowMultipleFileSelection: state.allowMultipleFileSelection,
        invalidFileMode: state.invalidFileMode,
        allowNewDataset: state.allowNewDataset,
        showUpdatedAt: state.showUpdatedAt,
        filesetPurpose: state.filesetPurpose,
      };

    case 'UPDATE_DATASET':
      return {
        ...state,
        dataset: action.payload,
      };

    case 'SET_DATASET':
      // When selecting a dataset, reset the selected files and dataset files
      // and clear the errors
      return {
        ...state,
        dataset: action.payload,
        selectedFiles: [],
        files: [],
        errors: {},
      };

    case 'SET_FILES': {
      const newFiles = [...state.files, ...action.payload];
      // Auto-select the file if there's only one file and single file selection mode.
      // In 'disable' and 'hide' modes an unsupported file can't be picked (disabled or hidden),
      // so it must not be auto-selected either.
      const shouldAutoSelect =
        newFiles.length === 1 &&
        !state.allowMultipleFileSelection &&
        (state.invalidFileMode === 'show' ||
          isFileTypeAllowed(newFiles[0], state.acceptableFileTypes));

      // Auto-set dataset name from first file if it's a new dataset and name is empty
      let updatedDataset = state.dataset;
      if (
        state.files.length === 0 && // This is the first file being uploaded
        action.payload.length > 0 && // There are files being added
        state.dataset?.type === 'new' && // It's a new dataset
        !state.dataset.name // Name is not set
      ) {
        const firstFile = action.payload[0];
        if (firstFile.type === 'new') {
          // Sanitize filename to create a valid dataset name
          const sanitizedName = sanitizeFilenameForDatasetName(firstFile.file.name);
          updatedDataset = {
            ...state.dataset,
            name: sanitizedName,
          };
        }
      }

      return {
        ...state,
        files: newFiles,
        dataset: updatedDataset,
        selectedFiles: shouldAutoSelect ? [newFiles[0]] : state.selectedFiles,
        errors: {},
      };
    }

    case 'TOGGLE_FILE_SELECTION': {
      // Single file selection
      if (!state.allowMultipleFileSelection) {
        if (state.selectedFiles.some((file) => file.id === action.payload.id)) {
          return { ...state, selectedFiles: [], errors: {} };
        }
        return {
          ...state,
          selectedFiles: [action.payload],
          errors: {},
        };
      }
      // Multiple file selection
      const isAlreadySelected = state.selectedFiles.some((file) => file.id === action.payload.id);
      if (isAlreadySelected) {
        return {
          ...state,
          selectedFiles: state.selectedFiles.filter((file) => file.id !== action.payload.id),
          errors: {},
        };
      }
      return {
        ...state,
        selectedFiles: [...state.selectedFiles, action.payload],
        errors: {},
      };
    }

    case 'SET_TAB':
      return {
        ...state,
        activeTab: action.payload,
        files: [],
        selectedFiles: [],
        dataset: undefined,
        errors: {},
      };

    case 'SET_SUBMITTING':
      return {
        ...state,
        isSubmitting: action.payload,
      };

    case 'SET_FETCHING':
      return {
        ...state,
        isFetching: action.payload,
      };

    case 'SET_ERRORS':
      return {
        ...state,
        errors: action.payload,
      };

    case 'CLEAR_ERRORS':
      return {
        ...state,
        errors: {},
      };

    case 'SET_ALLOW_MULTIPLE_FILE_SELECTION':
      return {
        ...state,
        allowMultipleFileSelection: action.payload,
      };

    default:
      return state;
  }
};

/**
 * Initial state for the UploadModal reducer
 */
export const uploadModalInitialState: UploadModalState = initialState;

/**
 * Custom hook that provides the UploadModal reducer
 *
 * NOTE: This hook should really only be used by `UploadModalProvider`. It shouldn't be used in any
 * component.
 *
 * @param state an optional initial state
 * @returns A tuple containing the current state and dispatch function
 */
export const useUploadModalReducer = (state?: UploadModalState) =>
  useReducer(uploadModalReducer, state ?? uploadModalInitialState);
