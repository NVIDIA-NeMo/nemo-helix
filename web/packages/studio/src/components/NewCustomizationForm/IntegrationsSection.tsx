// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import type { MlflowIntegration, WandbIntegration } from '@nemo/sdk/generated/customizer/schema';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { ControlledJsonInput } from '@studio/components/NewCustomizationForm/ControlledJsonInput';
import { ControlledStringListInput } from '@studio/components/NewCustomizationForm/ControlledStringListInput';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationBackend } from '@studio/util/customizationBackend';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { useFormContext } from 'react-hook-form';

/**
 * Experiment tracking for any customization backend. All three share one `IntegrationsSpec`
 * in the API, so the path prefix is the only thing that varies.
 */
type IntegrationField =
  | `wandb.${keyof WandbIntegration & string}`
  | `mlflow.${keyof MlflowIntegration & string}`;

export const IntegrationsSection = ({ backend }: { backend: CustomizationBackend }) => {
  const { control, formState } = useFormContext<CustomizationFormFields>();
  const disabled = formState.isSubmitting;

  const field = <P extends IntegrationField>(path: P) => `${backend}.integrations.${path}` as const;

  return (
    <FormSection
      collapsible
      title="Experiment Tracking"
      description="Optional. Stream metrics to Weights & Biases or MLflow while the job runs."
    >
      <Stack gap="density-md">
        <Text kind="label/bold/sm">Weights &amp; Biases</Text>
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.project'), control }}
          formFieldProps={{ slotLabel: 'Project' }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.name'), control }}
          formFieldProps={{ slotLabel: 'Run Name' }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.entity'), control }}
          formFieldProps={{ slotLabel: 'Entity' }}
          disabled={disabled}
        />
        <ControlledStringListInput
          useControllerProps={{ name: field('wandb.tags'), control }}
          formFieldProps={{
            slotLabel: 'Tags',
            slotInfo: 'Comma separated labels attached to the W&B run.',
          }}
          placeholder="grpo, recipe-aligned"
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.notes'), control }}
          formFieldProps={{ slotLabel: 'Notes' }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.base_url'), control }}
          formFieldProps={{
            slotLabel: 'Base URL',
            slotInfo: 'Only needed for a self-hosted W&B instance.',
          }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('wandb.api_key_secret'), control }}
          formFieldProps={{
            slotLabel: 'API Key Secret',
            slotInfo:
              "Name of a stored secret holding the W&B API key, as 'secret-name' or 'workspace/secret-name'. The key itself is never entered here.",
          }}
          placeholder="wandb-api-key"
          disabled={disabled}
        />

        <Text kind="label/bold/sm">MLflow</Text>
        <ControlledTextInput
          useControllerProps={{ name: field('mlflow.experiment_name'), control }}
          formFieldProps={{ slotLabel: 'Experiment Name' }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('mlflow.name'), control }}
          formFieldProps={{ slotLabel: 'Run Name' }}
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('mlflow.description'), control }}
          formFieldProps={{ slotLabel: 'Description' }}
          disabled={disabled}
        />
        <ControlledJsonInput
          useControllerProps={{ name: field('mlflow.tags'), control }}
          formFieldProps={{
            slotLabel: 'Tags (JSON object)',
            slotInfo: 'Key-value pairs attached to the MLflow run.',
          }}
          placeholder='{"team": "nemo"}'
          disabled={disabled}
        />
        <ControlledTextInput
          useControllerProps={{ name: field('mlflow.tracking_uri'), control }}
          formFieldProps={{ slotLabel: 'Tracking URI' }}
          disabled={disabled}
        />
      </Stack>
    </FormSection>
  );
};
