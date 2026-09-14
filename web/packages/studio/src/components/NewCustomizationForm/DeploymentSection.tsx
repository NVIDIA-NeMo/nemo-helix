// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Stack, Switch, Text } from '@nvidia/foundations-react-core';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { BaseModelDeploymentReadiness } from '@studio/hooks/useBaseModelDeploymentReadiness';
import { AdvancedSettingsAccordion } from '@studio/routes/NewDeploymentRoute/AdvancedSettingsAccordion';
import { EngineFields } from '@studio/routes/NewDeploymentRoute/EngineFields';
import { GPULoraFields } from '@studio/routes/NewDeploymentRoute/GPULoraFields';
import type { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { useState, type FC } from 'react';
import type { Control, FieldErrors } from 'react-hook-form';

export interface DeploymentSectionProps {
  readiness: BaseModelDeploymentReadiness;
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  baseModelRef: string;
  deployBaseModel: boolean;
  onDeployBaseModelChange: (deploy: boolean) => void;
}

/**
 * How the adapter this run produces will be served.
 *
 * A LoRA adapter is not deployed on its own — it is served by a deployment of its
 * **base model** with `lora_enabled`. Rendered only when the run produces an
 * adapter; merged and full-weight runs emit a standalone model that needs its own
 * deployment, which is a different target and a different flow.
 */
export const DeploymentSection: FC<DeploymentSectionProps> = ({
  readiness,
  control,
  errors,
  baseModelRef,
  deployBaseModel,
  onDeployBaseModelChange,
}) => {
  const [advancedAccordion, setAdvancedAccordion] = useState<string>();

  if (!baseModelRef) {
    return (
      <FormSection title="Deployment">
        <Text kind="body/regular/sm" className="text-subtle">
          Select a base model to see how this adapter will be served.
        </Text>
      </FormSection>
    );
  }

  if (readiness.isLoading) {
    return (
      <FormSection title="Deployment">
        <Text kind="body/regular/sm" className="text-subtle">
          Checking whether {baseModelRef} is deployed…
        </Text>
      </FormSection>
    );
  }

  // Already serving adapters: there is no deployment to create, so there is
  // nothing to decide.
  if (readiness.state === 'serving-lora') {
    return (
      <FormSection title="Deployment">
        <Banner kind="inline" status="success">
          {baseModelRef} is served by{' '}
          <strong>{readiness.deploymentName ?? 'an active deployment'}</strong> with LoRA enabled.
          This adapter will be servable as soon as the job finishes — nothing to configure.
        </Banner>
      </FormSection>
    );
  }

  const description =
    readiness.state === 'serving-without-lora'
      ? `${baseModelRef} is deployed, but that deployment has LoRA disabled and will refuse to serve this adapter. Create a LoRA-enabled deployment alongside it.`
      : `${baseModelRef} is not deployed. A LoRA adapter is served by a deployment of its base model, so without one this run produces a checkpoint nothing can serve.`;

  return (
    <FormSection title="Deployment" description={description}>
      <Stack gap="density-lg">
        {readiness.state === 'unavailable' && (
          <Banner kind="inline" status="warning">
            The existing deployment for {baseModelRef} is in a failed state
            {readiness.status ? ` (${readiness.status})` : ''}. Creating a new one will not remove
            it.
          </Banner>
        )}

        <Switch
          checked={deployBaseModel}
          onCheckedChange={onDeployBaseModelChange}
          slotLabel="Deploy the base model when training finishes"
        />

        {deployBaseModel ? (
          <>
            <Text kind="body/regular/sm" className="text-subtle">
              The deployment configuration is created now, so a bad engine or image is caught before
              the job starts. The job itself creates the deployment once training completes — no GPU
              is held while the run is in progress.
            </Text>
            <EngineFields control={control} errors={errors} />
            {/* `loraEnabled` is pinned true by `baseDeploymentDefaults`: this deployment
                exists to serve the adapter this run produces, and a base deployed without
                LoRA support would refuse it. Not a decision the user should be offered. */}
            <GPULoraFields control={control} errors={errors} hideLoraToggle />
            <AdvancedSettingsAccordion
              control={control}
              errors={errors}
              advancedAccordion={advancedAccordion}
              onAdvancedAccordionChange={setAdvancedAccordion}
            />
          </>
        ) : (
          <Banner kind="inline" status="warning">
            The job will run, but the adapter it produces will not be servable until {baseModelRef}{' '}
            has a LoRA-enabled deployment. You can create one from the Deployments page at any time,
            including after the job finishes.
          </Banner>
        )}
      </Stack>
    </FormSection>
  );
};
