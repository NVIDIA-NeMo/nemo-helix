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
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { UploadModalProvider } from '@nemo/common/src/components/UploadModal/Context/UploadModalProvider';
import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { uploadModalInitialState } from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { UploadModalProps } from '@nemo/common/src/components/UploadModal/types';
import { UploadPickerBody } from '@nemo/common/src/components/UploadModal/UploadPickerBody';
import { useUploadSubmit } from '@nemo/common/src/components/UploadModal/useUploadSubmit';
import {
  Button,
  ModalContent,
  ModalDialog,
  ModalFooter,
  ModalHeading,
  ModalMain,
  ModalRoot,
} from '@nvidia/foundations-react-core';
import { FC, MouseEvent, useId, useMemo } from 'react';

const UploadModalContent: FC<UploadModalProps> = ({
  workspace,
  open,
  title = 'Select a File',
  submitButtonText = 'Add Selected File',
  cancelButtonText = 'Cancel',
  includeDataset = false,
  includeTabs = false,
  onSubmit,
  onClose,
  className,
  attributes,
}) => {
  const [, dispatch] = useUploadModalContext();
  const modalId = useId();
  const { submit, isSubmitting } = useUploadSubmit({
    workspace,
    includeDataset,
    includeTabs,
    onSubmit,
  });

  if (!workspace) {
    console.error('UploadModal: workspace is required');
    return null;
  }

  const handleUserClose = () => {
    dispatch({ type: 'RESET' });
    onClose();
  };

  const handleSubmit = async (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const ok = await submit();
    if (ok) handleUserClose();
  };

  return (
    <ModalRoot id={modalId} open={open} onOpenChange={handleUserClose} {...attributes?.ModalRoot}>
      <ModalDialog>
        <ModalContent className={`w-[560px] ${className || ''}`} {...attributes?.ModalContent}>
          <ModalHeading {...attributes?.ModalHeading}>{title}</ModalHeading>
          <ModalMain {...attributes?.ModalMain}>
            <UploadPickerBody
              workspace={workspace}
              includeDataset={includeDataset}
              includeTabs={includeTabs}
            />
          </ModalMain>
          <ModalFooter className="flex w-full justify-end gap-2" {...attributes?.ModalFooter}>
            <Button kind="tertiary" onClick={handleUserClose} type="button">
              {cancelButtonText}
            </Button>
            <LoadingButton
              type="button"
              color="brand"
              loading={isSubmitting}
              onClick={handleSubmit}
            >
              {submitButtonText}
            </LoadingButton>
          </ModalFooter>
        </ModalContent>
      </ModalDialog>
    </ModalRoot>
  );
};

/**
 * A generic upload modal that allows selecting a dataset and uploading a file.
 * The upload could be selected from an existing dataset or a new dataset, as well as just uploading a file.
 *
 * This component wraps the content with UploadModalProvider to provide context to all child components.
 */
export const UploadModal: FC<UploadModalProps> = ({
  allowMultipleFileSelection,
  acceptableFileTypes,
  acceptableFileSize,
  invalidFileMode,
  allowNewDataset,
  ...props
}) => {
  const initialState = useMemo(
    () => ({
      ...uploadModalInitialState,
      allowMultipleFileSelection:
        allowMultipleFileSelection ?? uploadModalInitialState.allowMultipleFileSelection,
      acceptableFileTypes: acceptableFileTypes ?? uploadModalInitialState.acceptableFileTypes,
      acceptableFileSize: acceptableFileSize ?? uploadModalInitialState.acceptableFileSize,
      invalidFileMode: invalidFileMode ?? uploadModalInitialState.invalidFileMode,
      allowNewDataset: allowNewDataset ?? uploadModalInitialState.allowNewDataset,
    }),
    [
      allowMultipleFileSelection,
      acceptableFileTypes,
      acceptableFileSize,
      invalidFileMode,
      allowNewDataset,
    ]
  );
  return (
    <UploadModalProvider initialState={initialState}>
      <UploadModalContent {...props} />
    </UploadModalProvider>
  );
};
