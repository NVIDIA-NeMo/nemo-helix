// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage as getApiErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { toValidFilesetName } from '@nemo/common/src/utils/filesetName';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getFilesListFilesetsQueryKey,
  useFilesCreateFileset,
} from '@nemo/sdk/generated/platform/files';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { SegmentedControl, Stack, Text } from '@nvidia/foundations-react-core';
import {
  filesetCreateFormSchema,
  type FilesetCreateFormData,
  PURPOSE_COPY,
  StorageMode,
  type SupportedPurpose,
} from '@studio/components/FilesetCreateModal/constants';
import { useRemoteRepoMetadata } from '@studio/hooks/useRemoteRepoMetadata';
import { FilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { getFilesetDetailRoute } from '@studio/routes/utils';
import {
  isHuggingFaceUrl,
  isNgcUrl,
  storageConfigFromUrl,
} from '@studio/util/storageConfigFromUrl';
import { useQueryClient } from '@tanstack/react-query';
import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

export interface FilesetCreateModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  purpose: SupportedPurpose;
}

/** "Lean" create modal for filesets. Heading and post-create navigation differ
 *  by purpose (dataset vs model). Generic purpose is not supported here -
 *  flag-off paths and the API itself still allow it.
 *
 *  Local mode: name + description only. User adds files later via the detail
 *  page. (No file upload here, by design.)
 *  External mode: name + description + URL + workspace secret. */
export const FilesetCreateModal: FC<FilesetCreateModalProps> = ({
  open,
  onClose,
  workspace,
  purpose,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [storageMode, setStorageMode] = useState<StorageMode>(StorageMode.Local);
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);

  const {
    control,
    handleSubmit,
    setValue,
    watch,
    reset,
    formState: { errors, isValid, dirtyFields },
  } = useForm<FilesetCreateFormData>({
    resolver: zodResolver(filesetCreateFormSchema),
    defaultValues: { name: '', description: '', url: '', secretKey: '' },
    mode: 'onChange',
  });

  const url = watch('url');
  const selectedSecretName = watch('secretKey');

  const secretKeyLabel = useMemo(() => {
    if (!url?.trim()) return 'Secret Key';
    try {
      const parsed = new URL(url);
      if (isHuggingFaceUrl(parsed)) return 'HuggingFace Token (optional)';
      if (isNgcUrl(parsed)) return 'NGC API Key';
    } catch {
      // invalid URL — fall through
    }
    return 'Secret Key';
  }, [url]);

  // External mode auto-fill: when the URL resolves to a recognised remote
  // repo, derive a name slug (HF + NGC) and a description (HF public only).
  // Only fills fields the user has not edited. Backend preview endpoint
  // (nhx-1tk) will eventually replace the client-side fetch.
  const isExternalMode = storageMode === StorageMode.External;
  const { data: remoteMetadata, isFetching: isRemoteFetching } = useRemoteRepoMetadata(
    url,
    isExternalMode
  );

  useEffect(() => {
    if (!isExternalMode || !remoteMetadata) return;
    if (!dirtyFields.name && remoteMetadata.slug) {
      setValue('name', toValidFilesetName(remoteMetadata.slug), {
        shouldValidate: true,
        shouldDirty: false,
      });
    }
    if (!dirtyFields.description && remoteMetadata.description) {
      setValue('description', remoteMetadata.description.slice(0, 255), {
        shouldValidate: false,
        shouldDirty: false,
      });
    }
  }, [isExternalMode, remoteMetadata, dirtyFields.name, dirtyFields.description, setValue]);

  const { mutateAsync: createFileset, isPending } = useFilesCreateFileset({
    mutation: {
      onSuccess: (fileset: FilesetOutput) => {
        // Invalidate the workspace-wide fileset list so the table picks up the
        // new row. Per-fileset caches are seeded directly in onSubmit so the
        // destination page renders without a loading spinner — see below.
        queryClient.invalidateQueries({
          queryKey: getFilesListFilesetsQueryKey(fileset.workspace),
        });
      },
    },
  });

  const handleClose = useCallback(() => {
    reset();
    setStorageMode(StorageMode.Local);
    onClose();
  }, [reset, onClose]);

  const handleSecretCreated = useCallback(
    (secretName: string) => {
      setValue('secretKey', secretName, { shouldValidate: true });
    },
    [setValue]
  );

  const onSubmit = useCallback(
    async (data: FilesetCreateFormData) => {
      const trimmedUrl = data.url?.trim();
      const isExternal = storageMode === StorageMode.External && !!trimmedUrl;

      let storage: ReturnType<typeof storageConfigFromUrl> | undefined;
      if (isExternal) {
        try {
          storage = storageConfigFromUrl({
            url: trimmedUrl,
            secretKey: data.secretKey?.trim() || undefined,
          });
        } catch (e) {
          toast.error(
            e instanceof Error
              ? e.message
              : 'Invalid external storage URL or credential. For NGC, select a secret with your API key.'
          );
          return;
        }
      }

      let fileset: FilesetOutput;
      try {
        fileset = await createFileset({
          workspace,
          data: {
            name: data.name,
            description: data.description ?? '',
            purpose,
            storage,
          },
        });
      } catch (err) {
        toast.error(getApiErrorMessage(err as Error, 'Failed to create fileset'));
        return;
      }

      // Post-create navigation, per ASTD-167:
      //   External -> Card tab (where the README renders)
      //   Local    -> Files tab (where the user uploads next)
      handleClose();
      navigate(
        getFilesetDetailRoute(fileset.workspace, fileset.name, {
          tab: isExternal ? FilesetDetailTab.Card : FilesetDetailTab.Files,
        })
      );
    },
    [storageMode, createFileset, workspace, purpose, toast, navigate, handleClose]
  );

  return (
    <>
      <FormModal
        open={open}
        onClose={handleClose}
        title={PURPOSE_COPY[purpose].title}
        submitButtonText={isPending ? 'Creating…' : PURPOSE_COPY[purpose].submit}
        loading={isPending}
        submitDisabled={!isValid}
        onSubmit={handleSubmit(
          onSubmit,
          handleFormErrorsGeneric({ title: 'Fileset Create Form Errors' })
        )}
      >
        <Stack gap="density-lg">
          <SegmentedControl
            size="tiny"
            className="w-full"
            value={storageMode}
            onValueChange={(value) => setStorageMode(value as StorageMode)}
            items={[
              { value: StorageMode.Local, children: 'Local' },
              { value: StorageMode.External, children: 'External' },
            ]}
          />

          {isExternalMode && (
            <>
              <ControlledTextInput
                label="URL"
                disabled={isPending}
                autoFocus
                useControllerProps={{ name: 'url', control }}
              />
              <SecretSearchableSelect
                workspace={workspace}
                queryEnabled={Boolean(workspace)}
                ensureOptionValue={selectedSecretName || undefined}
                useControllerProps={{ control, name: 'secretKey' }}
                onRequestNewSecret={() => setCreateSecretModalOpen(true)}
                triggerPlaceholder=""
                formFieldProps={{
                  slotLabel: secretKeyLabel,
                  slotInfo:
                    'Select a secret that stores the credential for this URL, or choose New Secret to create one.',
                  slotError: errors.secretKey?.message,
                }}
              />
            </>
          )}

          <ControlledTextInput
            label="Name"
            disabled={isPending || (isExternalMode && isRemoteFetching)}
            autoFocus={!isExternalMode}
            useControllerProps={{ name: 'name', control }}
          />
          <ControlledTextArea
            label="Description (optional)"
            disabled={isPending || (isExternalMode && isRemoteFetching)}
            useControllerProps={{ name: 'description', control }}
          />

          {!isExternalMode && (
            <Text kind="body/regular/sm">You can upload files after creating the fileset.</Text>
          )}
        </Stack>
      </FormModal>
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
        onSecretCreated={handleSecretCreated}
      />
    </>
  );
};
