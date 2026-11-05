// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage, isConflictError } from '@nemo/common/src/api/common/utils';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { toValidFilesetName } from '@nemo/common/src/utils/filesetName';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { copyFilesetToLocal } from '@studio/api/files/copyFilesetToLocal';
import {
  filesetCreateFormSchema,
  type FilesetCreateFormData,
} from '@studio/components/FilesetCreateModal/constants';
import { FilesetNameField } from '@studio/components/filesets/FilesetNameField';
import { filesetNameTakenMessage } from '@studio/components/filesets/filesetNames';
import { invalidateFilesetNameCheck } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useState } from 'react';
import { useForm } from 'react-hook-form';

export interface CopyFilesetModalProps {
  source: FilesetOutput;
  onClose: () => void;
  onCopied?: (fileset: FilesetOutput) => void;
}

/** Copies a fileset — typically one backed by read-only external storage — into a new
 *  local fileset whose files the user can edit. */
export const CopyFilesetModal: FC<CopyFilesetModalProps> = ({ source, onClose, onCopied }) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [nameStatus, setNameStatus] = useState<NameCheckStatus>();
  const [progress, setProgress] = useState<{ copied: number; total: number }>();
  const [isPending, setIsPending] = useState(false);

  const {
    control,
    handleSubmit,
    setError,
    formState: { isValid },
  } = useForm<FilesetCreateFormData>({
    resolver: zodResolver(filesetCreateFormSchema),
    defaultValues: {
      name: toValidFilesetName(`${source.name}-copy`),
      description: source.description,
    },
    mode: 'onChange',
  });

  const onSubmit = async ({ name, description }: FilesetCreateFormData) => {
    setIsPending(true);
    try {
      const fileset = await copyFilesetToLocal({
        source,
        name,
        description,
        onProgress: (copied, total) => setProgress({ copied, total }),
      });
      void invalidateDatasetCaches(fileset.workspace, fileset.name, ['list']);
      toast.success(`Copied ${source.name} to ${fileset.name}.`);
      onClose();
      onCopied?.(fileset);
    } catch (error) {
      if (isConflictError(error)) {
        void invalidateFilesetNameCheck(queryClient, source.workspace, name);
        setError('name', { message: filesetNameTakenMessage(name, source.workspace) });
      } else {
        toast.error(getErrorMessage(error as Error, 'Failed to copy fileset'));
      }
    } finally {
      setIsPending(false);
      setProgress(undefined);
    }
  };

  return (
    <FormModal
      open
      onClose={onClose}
      title={`Copy Fileset: ${source.name}`}
      submitButtonText={isPending ? 'Copying…' : 'Copy'}
      loading={isPending}
      disabled={isPending}
      submitDisabled={!isValid || nameStatus === 'conflict'}
      onSubmit={handleSubmit(onSubmit)}
    >
      <Stack gap="density-lg">
        <Text kind="body/regular/sm">
          {source.storage.type === 'local'
            ? 'Creates a new local fileset with a copy of every file.'
            : 'Creates a local fileset with a copy of every file, so you can edit, upload, and delete files.'}{' '}
          Files are copied through your browser, so large filesets may take a while.
        </Text>
        <FilesetNameField
          control={control}
          name="name"
          workspace={source.workspace}
          onStatusChange={setNameStatus}
          disabled={isPending}
          autoFocus
        />
        <ControlledTextArea
          label="Description (optional)"
          disabled={isPending}
          useControllerProps={{ name: 'description', control }}
        />
        {progress && (
          <Text kind="body/regular/sm" aria-live="polite">
            Copied {progress.copied} of {progress.total} files…
          </Text>
        )}
      </Stack>
    </FormModal>
  );
};
