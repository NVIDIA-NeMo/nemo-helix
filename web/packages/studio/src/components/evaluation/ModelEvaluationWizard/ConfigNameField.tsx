// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import type { EvaluationFormValues } from '@studio/components/evaluation/EvalWizard/types';
import { useConfigNameStatus } from '@studio/components/evaluation/ModelEvaluationWizard/useConfigNameStatus';
import { nameFieldSlots } from '@studio/components/form/entityNameCheck';
import type { FC } from 'react';
import { useFormContext } from 'react-hook-form';

/**
 * What a newly authored configuration is saved as.
 *
 * The name becomes the fileset name verbatim, so it is checked for conflicts the
 * same way the agent flow checks an experiment name: sanitize, debounce, then
 * ask whether a fileset already answers to it. Whatever holds the name (a dataset
 * or an earlier config) is a fileset, so the conflict copy says that.
 */
export const ConfigNameField: FC = () => {
  const {
    control,
    formState: { errors },
  } = useFormContext<EvaluationFormValues>();
  const { preview, status } = useConfigNameStatus();

  return (
    <ControlledTextInput
      useControllerProps={{ name: 'name', control }}
      formFieldProps={{
        slotLabel: 'Name',
        ...nameFieldSlots({
          entity: 'configuration',
          preview,
          status,
          schemaError: errors.name?.message,
          describe: 'Saved as a fileset so later runs can reuse this configuration.',
        }),
        ...(status === 'conflict' && {
          slotError: `A fileset named ${preview} already exists in this workspace. Choose a different configuration name.`,
        }),
      }}
    />
  );
};
