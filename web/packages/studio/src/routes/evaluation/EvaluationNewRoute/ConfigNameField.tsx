// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  nameCheckStatus,
  nameFieldSlots,
} from '@studio/components/evaluation/shared/entityNameField';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';
import { useDebounce } from 'use-debounce';

/**
 * What a newly authored configuration is saved as.
 *
 * The name becomes the fileset name verbatim, so it is checked for conflicts the
 * same way the agent flow checks an experiment name: sanitize, debounce, then
 * ask whether a fileset already answers to it.
 */
export const ConfigNameField: FC = () => {
  const workspace = useWorkspaceFromPath();
  const {
    control,
    formState: { errors },
  } = useFormContext<EvaluationFormValues>();
  const name = useWatch({ control, name: 'name' });

  const preview = toValidEntityName(name ?? '', '');
  const [debouncedName] = useDebounce(preview, DEFAULT_DEBOUNCE_MS);
  const conflictQuery = useFilesListFilesets(
    workspace,
    { page_size: 1, filter: { name: debouncedName } },
    { query: { enabled: Boolean(debouncedName) } }
  );

  return (
    <ControlledTextInput
      useControllerProps={{ name: 'name', control }}
      formFieldProps={{
        slotLabel: 'Configuration Name',
        ...nameFieldSlots({
          entity: 'configuration',
          preview,
          status: nameCheckStatus(preview, debouncedName, conflictQuery),
          schemaError: errors.name?.message,
          describe: 'Saved as a fileset so later runs can reuse this configuration.',
        }),
      }}
    />
  );
};
