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
import { Flex } from '@nvidia/foundations-react-core';
import { GPULoraFields } from '@studio/routes/NewDeploymentRoute/GPULoraFields';
import type { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import type { FC } from 'react';
import type { Control, FieldErrors } from 'react-hook-form';

export type NgcSourceFieldsProps = {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
};

export const NgcSourceFields: FC<NgcSourceFieldsProps> = ({ control, errors }) => (
  <>
    <Flex gap="4" className="w-full">
      <div className="min-w-0 flex-1">
        <ControlledTextInput
          useControllerProps={{ control, name: 'imageName' }}
          name="imageName"
          label="Image Name"
          formFieldProps={{
            slotInfo: 'Example: nvcr.io/nim/meta/llama-3.2-1b-instruct',
            slotError: errors.imageName?.message,
          }}
        />
      </div>
      <div className="w-[140px] shrink-0">
        <ControlledTextInput
          useControllerProps={{ control, name: 'imageTag' }}
          name="imageTag"
          label="Image Tag"
          formFieldProps={{
            slotError: errors.imageTag?.message,
          }}
        />
      </div>
    </Flex>
    <GPULoraFields control={control} errors={errors} />
  </>
);
