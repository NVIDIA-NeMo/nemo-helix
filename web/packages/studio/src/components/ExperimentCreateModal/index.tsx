/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getListExperimentsQueryKey,
  useCreateExperiment,
} from '@nemo/sdk/generated/platform/experiments';
import { FormField, Stack, TextInput } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import {
  EXPERIMENT_SETTINGS_NAMES,
  experimentSettingsPayload,
} from '@studio/components/evaluation/shared/experimentSettings';
import { ExperimentSettingsFields } from '@studio/components/evaluation/shared/ExperimentSettingsFields';
import {
  experimentCreateDefaults,
  experimentCreateSchema,
  type ExperimentCreateFormFields,
} from '@studio/components/ExperimentCreateModal/constants';
import { AxiosError } from 'axios';
import { type FC } from 'react';
import { useController, useForm, type SubmitHandler } from 'react-hook-form';

export interface ExperimentCreateModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
}

export const ExperimentCreateModal: FC<ExperimentCreateModalProps> = ({
  open,
  onClose,
  workspace,
}) => {
  const {
    reset,
    handleSubmit,
    formState: { errors, isSubmitted, isSubmitting, touchedFields },
    setError,
    control,
  } = useForm<ExperimentCreateFormFields>({
    resolver: zodResolver(experimentCreateSchema),
    mode: 'onTouched',
    defaultValues: experimentCreateDefaults,
  });

  const {
    field: { value: nameValue, onChange: onNameChange, onBlur: onNameBlur },
  } = useController({ control, name: 'name' });

  const preview = toValidEntityName(nameValue, '');
  const nameError = touchedFields.name || isSubmitted ? errors.name?.message : undefined;

  const formDisabled = isSubmitting;

  const toast = useToast();

  const { mutateAsync: createExperiment } = useCreateExperiment({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getListExperimentsQueryKey(workspace) });
      },
    },
  });

  const resetAndClose = () => {
    reset(experimentCreateDefaults);
    onClose();
  };

  const onSubmit: SubmitHandler<ExperimentCreateFormFields> = async (data) => {
    try {
      await createExperiment({
        workspace,
        data: { name: data.name, ...experimentSettingsPayload(data) },
      });
      resetAndClose();
    } catch (error) {
      const errorDetail =
        error instanceof AxiosError && error.response?.data?.detail
          ? error.response.data.detail
          : undefined;

      if (
        error instanceof AxiosError &&
        error.response?.status === 409 &&
        typeof errorDetail === 'string'
      ) {
        setError('name', { message: errorDetail });
      } else {
        let errorMessage: string;
        if (Array.isArray(errorDetail) && errorDetail.length > 0 && errorDetail[0].msg) {
          errorMessage = errorDetail[0].msg;
        } else if (errorDetail && typeof errorDetail === 'string') {
          errorMessage = errorDetail;
        } else if (error instanceof Error) {
          errorMessage = error.message;
        } else {
          errorMessage = 'Unknown error';
        }

        toast.error(`Failed to create experiment: ${errorMessage}`);
      }
    }
  };

  return (
    <FormModal
      title="Create experiment"
      instruction="Group evaluations to allow easy comparison of top level and test cases"
      submitButtonText="Create"
      disabled={formDisabled}
      loading={isSubmitting}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Experiment Create Form Errors' })
      )}
      onClose={resetAndClose}
      open={open}
      className="w-[800px] min-h-[400px]"
    >
      <Stack gap="density-2xl" className="w-full min-w-0">
        <FormField
          slotLabel="Name"
          slotError={nameError}
          slotHelp={
            nameError ? undefined : preview ? (
              <>
                Your experiment will be created as <span className="text-primary">{preview}</span>
              </>
            ) : undefined
          }
          status={nameError ? 'error' : undefined}
        >
          <TextInput
            autoFocus
            disabled={formDisabled}
            status={nameError ? 'error' : undefined}
            value={nameValue}
            onChange={(e) => onNameChange((e.target as HTMLInputElement).value)}
            onBlur={onNameBlur}
          />
        </FormField>

        <ExperimentSettingsFields
          control={control}
          names={EXPERIMENT_SETTINGS_NAMES}
          disabled={formDisabled}
        />
      </Stack>
    </FormModal>
  );
};
