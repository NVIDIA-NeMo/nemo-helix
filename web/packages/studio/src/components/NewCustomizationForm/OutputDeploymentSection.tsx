// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Stack, Switch, Text } from '@nvidia/foundations-react-core';
import { DeploymentFields } from '@studio/components/NewCustomizationForm/DeploymentFields';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import type { FC } from 'react';
import type { Control, FieldErrors } from 'react-hook-form';

export interface OutputDeploymentSectionProps {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  outputName: string;
  deployOutputModel: boolean;
  onDeployOutputModelChange: (deploy: boolean) => void;
}

/**
 * How the **model** this run produces will be served.
 *
 * The counterpart to `DeploymentSection`, which handles the adapter case. Rendered
 * when the run emits full weights — `all_weights` on any backend, automodel's
 * `lora_merged`, unsloth's merged save methods, and DPO, which has no
 * `finetuning_type` at all and is always full-weight.
 *
 * Simpler than the adapter flow in the one way that matters: there is nothing to
 * check. An adapter is served by a deployment of its *base*, which may already
 * exist in any of four states, so that section has to resolve the base's real
 * deployment before it can say anything useful. A full-weight run creates a new
 * Model Entity that by definition nothing is serving yet, so the only question is
 * whether to deploy it — which is also why `launch_model` guards its
 * `_has_active_deployment` check with `is_lora`.
 */
export const OutputDeploymentSection: FC<OutputDeploymentSectionProps> = ({
  control,
  errors,
  outputName,
  deployOutputModel,
  onDeployOutputModelChange,
}) => {
  // The name seeds the config and the deployment, and the config is created before
  // the job exists. Without it there is nothing to name either after.
  if (!outputName) {
    return (
      <FormSection title="Deployment">
        <Text kind="body/regular/sm" className="text-subtle">
          Name the output model to choose how it will be served.
        </Text>
      </FormSection>
    );
  }

  return (
    <FormSection
      title="Deployment"
      description={`This run produces ${outputName}, a new model with its own weights. Deploy it to make it servable.`}
    >
      <Stack gap="density-lg">
        <Switch
          checked={deployOutputModel}
          onCheckedChange={onDeployOutputModelChange}
          slotLabel="Deploy the model when training finishes"
        />

        {deployOutputModel ? (
          <Text kind="body/regular/sm" className="text-subtle">
            The deployment configuration is created now, so a bad engine or image is caught before
            the job starts. The job itself creates the deployment once training completes — no GPU
            is held while the run is in progress.
          </Text>
        ) : (
          <Banner kind="inline" status="warning">
            The job will run and {outputName} will be registered, but nothing will serve it until
            you deploy it. You can do that from the Deployments page at any time after the job
            finishes.
          </Banner>
        )}

        {/* Unlike the adapter flow, `loraEnabled` is a real choice here: this deployment
            serves a full-weight model, and enabling LoRA lets it also serve adapters
            trained against that model later. Nothing about this run depends on it. */}
        {deployOutputModel && <DeploymentFields control={control} errors={errors} />}
      </Stack>
    </FormSection>
  );
};
