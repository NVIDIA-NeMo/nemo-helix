// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AdvancedSettingsAccordion } from '@studio/routes/NewDeploymentRoute/AdvancedSettingsAccordion';
import { EngineFields } from '@studio/routes/NewDeploymentRoute/EngineFields';
import { GPULoraFields } from '@studio/routes/NewDeploymentRoute/GPULoraFields';
import type { WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { useState, type FC } from 'react';
import type { Control, FieldErrors } from 'react-hook-form';

export interface DeploymentFieldsProps {
  control: Control<WizardFormValues>;
  errors: FieldErrors<WizardFormValues>;
  /**
   * Omit the LoRA switch because the caller has fixed `loraEnabled` itself.
   *
   * True only for the adapter flow, where a base deployed without LoRA support
   * refuses the adapter it exists to serve. A run that emits a standalone model
   * has no such invariant — serving adapters against it later is a real choice —
   * so that flow leaves the switch visible.
   */
  hideLoraToggle?: boolean;
}

/**
 * The deployment controls shared by both fine-tuning deployment sections.
 *
 * Extracted rather than duplicated because the two sections differ entirely in
 * their copy and their states but not at all in what they collect: the same
 * engine, GPU and advanced settings produce the same `ModelDeploymentConfig`
 * whichever model it points at. Keeping one copy means field order, the
 * accordion's local state, and any future addition land in both at once.
 */
export const DeploymentFields: FC<DeploymentFieldsProps> = ({
  control,
  errors,
  hideLoraToggle = false,
}) => {
  const [advancedAccordion, setAdvancedAccordion] = useState<string>();

  return (
    <>
      <EngineFields control={control} errors={errors} />
      <GPULoraFields control={control} errors={errors} hideLoraToggle={hideLoraToggle} />
      <AdvancedSettingsAccordion
        control={control}
        errors={errors}
        advancedAccordion={advancedAccordion}
        onAdvancedAccordionChange={setAdvancedAccordion}
      />
    </>
  );
};
