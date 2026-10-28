// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FilesetFileOutput,
  FilesetOutput,
  FilesetPurpose,
} from '@nemo/sdk/generated/platform/schema';
import {
  ModalContent,
  ModalHeading,
  ModalMain,
  ModalRoot,
  ModalFooter,
} from '@nvidia/foundations-react-core';
import { ReactNode } from 'react';

/**
 * @deprecated Use FilesetFileOutput directly from '@nemo/sdk/generated/platform/schema'
 */
export type ListFileEntry = FilesetFileOutput;

/**
 * @deprecated Use FilesetOutput directly from '@nemo/sdk/generated/platform/schema'
 */
export type Dataset = FilesetOutput;

export type SubmitFileType = {
  type: 'file';
  files?: File[];
};

export type SubmitDatasetType = {
  type: 'dataset';
  dataset: FilesetOutput;
  path: string;
  url: string;
};

export type SubmitUploadType = SubmitFileType | SubmitDatasetType;

export interface UploadModalProps {
  workspace: string;
  open: boolean;
  onSubmit: (data: SubmitUploadType) => void;
  onClose: () => void;
  title?: ReactNode;
  submitButtonText?: string;
  cancelButtonText?: string;
  /** If true, shows dataset selection UI. If false, only shows file upload. */
  includeDataset?: boolean;
  /** If true, renders both dataset and file upload options in separate tabs. */
  includeTabs?: boolean;
  className?: string;
  /** If true, allows multiple file selection. */
  allowMultipleFileSelection?: boolean;
  /** Accepted file types for the upload. */
  acceptableFileTypes?: string[];
  /** Maximum file size for the upload. */
  acceptableFileSize?: number;
  /** How to render existing files whose extension isn't in
   *  ``acceptableFileTypes``. ``'show'`` (default) renders everything;
   *  ``'hide'`` filters them out; ``'disable'`` shows them but blocks
   *  selection. */
  invalidFileMode?: 'show' | 'hide' | 'disable';
  /** When false, the dataset picker hides the "Create new dataset" option.
   *  Defaults to ``true`` (legacy behaviour). */
  allowNewDataset?: boolean;
  /** Fileset ``purpose`` the picker lists. Defaults to ``'dataset'``. */
  filesetPurpose?: FilesetPurpose;
  /** Label for the fileset picker. Defaults to ``'Dataset'``. */
  datasetLabel?: string;
  /** Auto-select the first root-level accepted file on fileset selection. */
  autoSelectFirstAcceptable?: boolean;
  /** Show each fileset's updated-at date in the picker options (opt-in). */
  showUpdatedAt?: boolean;
  attributes?: {
    ModalRoot?: React.ComponentProps<typeof ModalRoot>;
    ModalContent?: React.ComponentProps<typeof ModalContent>;
    ModalHeading?: React.ComponentProps<typeof ModalHeading>;
    ModalMain?: React.ComponentProps<typeof ModalMain>;
    ModalFooter?: React.ComponentProps<typeof ModalFooter>;
  };
}

type ExistingFile = {
  type: 'existing';
  id: string;
  file: FilesetFileOutput;
};

type NewFile = {
  type: 'new';
  id: string;
  file: File;
};

export type NewDataset = {
  type: 'new';
  name: string;
};

export type ExistingDataset = {
  type: 'existing';
  dataset: FilesetOutput;
};

export type UploadFile = ExistingFile | NewFile;

export type UploadDataset = ExistingDataset | NewDataset;
