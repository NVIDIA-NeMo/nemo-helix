/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { EngineFields } from '@studio/routes/NewDeploymentRoute/EngineFields';
import { GPULoraFields } from '@studio/routes/NewDeploymentRoute/GPULoraFields';
import type { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { FC, useState } from 'react';
import { Control, FieldErrors } from 'react-hook-form';

export type HuggingFaceSourceFieldsProps = {
  workspace: string;
  queryEnabled: boolean;
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
};

export const HuggingFaceSourceFields: FC<HuggingFaceSourceFieldsProps> = ({
  workspace,
  queryEnabled,
  control,
  errors,
}) => {
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);

  return (
    <>
      <ControlledTextInput
        useControllerProps={{ control, name: 'repoId' }}
        name="repoId"
        label="Repo ID"
        formFieldProps={{
          slotInfo: 'Public or private model repo. Private repos need a token secret below.',
          slotError: errors.repoId?.message,
        }}
      />
      <SecretSearchableSelect
        workspace={workspace}
        triggerPlaceholder=""
        queryEnabled={queryEnabled}
        useControllerProps={{ control, name: 'hfTokenSecret' }}
        onRequestNewSecret={() => setCreateSecretModalOpen(true)}
        formFieldProps={{
          slotLabel: 'HuggingFace Secret',
          slotInfo: 'Required for private or gated models; stored as a workspace secret.',
          slotError: errors.hfTokenSecret?.message,
        }}
      />
      <EngineFields control={control} errors={errors} />
      <GPULoraFields control={control} errors={errors} />
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
      />
    </>
  );
};
