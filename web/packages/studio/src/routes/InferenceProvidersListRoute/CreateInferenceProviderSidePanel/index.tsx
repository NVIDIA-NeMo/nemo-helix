/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { ENTITY_NAME_HELP, entityNameSchema } from '@nemo/common/src/utils/entityName';
import {
  getModelsListProvidersQueryKey,
  useModelsCreateProvider,
  useModelsListProviders,
} from '@nemo/sdk/generated/platform/model-providers';
import { Button, Flex, FormField, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { InferenceModelProviderSelect } from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel/InferenceModelProviderSelect';
import {
  PRESET_CREDENTIALS,
  type InferenceProviderPresetId,
} from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel/inferenceProviderPresets';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { useQueryClient } from '@tanstack/react-query';
import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

const PROVIDERS_PAGE_SIZE = 100;

const createProviderFormSchema = z.object({
  name: entityNameSchema('Name'),
  host_url: z.string().min(1, 'Host URL is required').url('Enter a valid URL').max(2048),
  api_key_secret_name: z.string().max(255).optional().or(z.literal('')),
});

type CreateProviderFormData = z.infer<typeof createProviderFormSchema>;

const defaultValues: CreateProviderFormData = {
  name: '',
  host_url: '',
  api_key_secret_name: '',
};

export interface CreateInferenceProviderSidePanelProps {
  workspace: string;
  open: boolean;
  onClose: () => void;
  defaultPreset?: InferenceProviderPresetId;
}

function normalizeHostUrl(url: string): string {
  const trimmed = url.trim().toLowerCase();
  if (!trimmed) return trimmed;
  try {
    const u = new URL(trimmed);
    const path = u.pathname.replace(/\/+$/, '');
    return `${u.origin}${path}`;
  } catch {
    return trimmed.replace(/\/+$/, '');
  }
}

export const CreateInferenceProviderSidePanel: FC<CreateInferenceProviderSidePanelProps> = ({
  workspace,
  open,
  onClose,
  defaultPreset,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [preset, setPreset] = useState<InferenceProviderPresetId>(defaultPreset ?? 'custom');
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);

  const { data: providersData, isLoading: isProvidersLoading } = useModelsListProviders(
    workspace,
    { page_size: PROVIDERS_PAGE_SIZE },
    { query: { enabled: open && !!workspace } }
  );
  const { existingNames, existingHostUrls } = useMemo(() => {
    const providers = providersData?.data ?? [];
    return {
      existingNames: new Set(providers.map((p) => p.name)),
      existingHostUrls: new Set(
        providers.map((p) => (p.host_url ? normalizeHostUrl(p.host_url) : '')).filter(Boolean)
      ),
    };
  }, [providersData?.data]);

  const isPresetDisabled = useCallback(
    (id: Exclude<InferenceProviderPresetId, 'custom'>) => {
      const c = PRESET_CREDENTIALS[id];
      return existingNames.has(c.name) || existingHostUrls.has(normalizeHostUrl(c.host_url));
    },
    [existingNames, existingHostUrls]
  );

  const {
    mutateAsync: createProvider,
    error: createError,
    isPending,
    reset: resetCreateMutation,
  } = useModelsCreateProvider({
    mutation: {
      onSuccess: () => {
        toast.success('Inference provider created successfully');
        queryClient.invalidateQueries({
          queryKey: getModelsListProvidersQueryKey(workspace),
        });
        resetAndClose();
      },
      onError: (error) => {
        toast.error(getErrorMessage(error, 'Failed to create inference provider'));
      },
    },
  });

  const formSchema = useMemo(
    () =>
      createProviderFormSchema.refine((data) => !existingNames.has(data.name), {
        path: ['name'],
        message: 'A provider with this name already exists.',
      }),
    [existingNames]
  );

  const {
    control,
    reset: resetForm,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(formSchema),
    defaultValues,
    disabled: isPending,
    mode: 'onChange',
  });

  const handlePresetChange = useCallback(
    (next: InferenceProviderPresetId) => {
      setPreset(next);
      if (next === 'custom') {
        setValue('name', '', { shouldValidate: true });
        setValue('host_url', '', { shouldValidate: true });
        return;
      }
      const c = PRESET_CREDENTIALS[next];
      setValue('name', c.name, { shouldValidate: true });
      setValue('host_url', c.host_url, { shouldValidate: true });
    },
    [setValue]
  );

  // Apply defaultPreset to form fields the first time the panel opens
  const hasAppliedDefaultRef = useRef(false);
  useEffect(() => {
    if (!open) {
      hasAppliedDefaultRef.current = false;
      return;
    }
    if (!hasAppliedDefaultRef.current && defaultPreset) {
      hasAppliedDefaultRef.current = true;
      handlePresetChange(defaultPreset);
    }
  }, [open, defaultPreset, handlePresetChange]);

  const reset = () => {
    resetCreateMutation();
    resetForm(defaultValues);
    setPreset('custom');
  };

  const resetAndClose = () => {
    reset();
    onClose();
  };

  useEffect(() => {
    if (!open) return;
    if (preset !== 'custom' && isPresetDisabled(preset)) {
      handlePresetChange('custom');
    }
  }, [open, preset, isPresetDisabled, handlePresetChange]);

  const onSubmit: SubmitHandler<CreateProviderFormData> = async (formData) => {
    try {
      await createProvider({
        workspace,
        data: {
          name: formData.name,
          host_url: formData.host_url,
          api_key_secret_name: formData.api_key_secret_name || undefined,
        },
      });
    } catch {
      // Error handling via mutation onError / createError
    }
  };

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen && !isPending) {
      resetAndClose();
    }
  };

  const showCustomFields = preset === 'custom';

  const panel = (
    <SidePanel
      open={open}
      onOpenChange={handleOpenChange}
      side="right"
      bordered
      modal
      className="max-w-[min(480px,100vw)] w-full [&_.nv-side-panel-main]:gap-density-xl"
      slotHeading="Add Inference Provider"
      renderContent={({ children }) => (
        <form className="flex min-h-0 flex-1 flex-col" onSubmit={handleSubmit(onSubmit)} noValidate>
          {children}
        </form>
      )}
      slotFooter={
        <Flex justify="end" gap="density-lg" className="w-full">
          <Button
            kind="tertiary"
            type="button"
            disabled={isPending}
            onClick={() => {
              if (!isPending) resetAndClose();
            }}
          >
            Cancel
          </Button>
          <LoadingButton
            type="submit"
            loading={isPending}
            disabled={isPending || isProvidersLoading}
          >
            Add Provider
          </LoadingButton>
        </Flex>
      }
    >
      <Stack gap="density-xl" className="min-h-0 flex-1">
        <Text kind="body/regular/md">Select a provider to add to your workspace.</Text>

        {createError?.message ? (
          <Text className="text-feedback-danger" kind="body/regular/sm">
            {createError.message}
          </Text>
        ) : null}

        <FormField name="model-provider" slotLabel="Model Provider">
          <InferenceModelProviderSelect
            value={preset}
            onValueChange={handlePresetChange}
            disabled={isPending}
            isPresetDisabled={isPresetDisabled}
          />
        </FormField>

        {showCustomFields ? (
          <>
            <ControlledTextInput
              useControllerProps={{ control, name: 'name' }}
              name="name"
              label="Name"
              formFieldProps={{
                slotInfo: ENTITY_NAME_HELP,
                slotError: errors.name?.message,
              }}
            />
            <ControlledTextInput
              useControllerProps={{ control, name: 'host_url' }}
              name="host_url"
              label="Host URL"
              formFieldProps={{
                slotInfo: 'The inference API base URL for this provider.',
                slotError: errors.host_url?.message,
              }}
            />
          </>
        ) : null}

        <SecretSearchableSelect
          workspace={workspace}
          queryEnabled={open && !!workspace}
          useControllerProps={{ control, name: 'api_key_secret_name' }}
          onRequestNewSecret={() => setCreateSecretModalOpen(true)}
          triggerPlaceholder=""
          formFieldProps={{
            slotLabel: 'API Key Secret',
            slotInfo: 'Name of a secret created in Secrets that holds the API key.',
            slotError: errors.api_key_secret_name?.message,
          }}
        />
      </Stack>
    </SidePanel>
  );

  return (
    <>
      {panel}
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
      />
    </>
  );
};
