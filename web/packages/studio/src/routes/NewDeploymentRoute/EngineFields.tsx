/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Engine } from '@nemo/sdk/generated/platform/schema';
import {
  engineRequiresImage,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import type { FC } from 'react';
import { useWatch, type Control, type FieldErrors } from 'react-hook-form';

const ENGINE_ITEMS = [
  { value: Engine.vllm, children: 'vLLM' },
  { value: Engine.nim, children: 'NIM' },
  { value: Engine.generic, children: 'Generic container' },
];

const ENGINE_INFO: Record<Engine, string> = {
  [Engine.vllm]:
    'Serves any supported architecture from a model-agnostic vLLM image. Leave the image blank to use the platform default.',
  [Engine.nim]:
    'Runs a prebuilt NIM container. The image must match the model architecture, so it is required here.',
  [Engine.generic]:
    'Runs your image as-is, with no inference-engine compilation. The image is required.',
};

export type EngineFieldsProps = {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
};

/**
 * Engine picker plus the image fields it governs.
 *
 * Shared by the HuggingFace and Workspace sources. The NGC source is a NIM
 * container by definition and collects its image in `NgcSourceFields`.
 */
export const EngineFields: FC<EngineFieldsProps> = ({ control, errors }) => {
  const engine = useWatch({ control, name: 'engine' });
  const imageRequired = engineRequiresImage(engine);

  return (
    <>
      <ControlledSelect
        useControllerProps={{ control, name: 'engine' }}
        items={ENGINE_ITEMS}
        formFieldProps={{
          slotLabel: 'Engine',
          slotInfo: ENGINE_INFO[engine],
          slotError: errors.engine?.message,
        }}
      />
      <ControlledTextInput
        useControllerProps={{ control, name: 'imageName' }}
        name="imageName"
        label={imageRequired ? 'Image name' : 'Image name (optional)'}
        formFieldProps={{
          slotInfo: imageRequired
            ? 'Fully qualified image repository, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct.'
            : 'Override the default vLLM image. Leave blank unless you need a specific build.',
          slotError: errors.imageName?.message,
        }}
      />
      <ControlledTextInput
        useControllerProps={{ control, name: 'imageTag' }}
        name="imageTag"
        label={imageRequired ? 'Image tag' : 'Image tag (optional)'}
        formFieldProps={{
          slotInfo: 'Pin a specific tag rather than relying on a floating one.',
          slotError: errors.imageTag?.message,
        }}
      />
    </>
  );
};
