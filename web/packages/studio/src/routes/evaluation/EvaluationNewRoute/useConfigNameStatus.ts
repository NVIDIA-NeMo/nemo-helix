// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  nameCheckStatus,
  type NameCheckStatus,
} from '@studio/components/evaluation/shared/entityNameField';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useFormContext, useWatch } from 'react-hook-form';
import { useDebounce } from 'use-debounce';

/**
 * Whether the typed configuration name is free.
 *
 * Shared by the field that shows it and the step that blocks on it: the name
 * becomes the fileset name verbatim, and the check is a fetch rather than a
 * schema rule, so nothing in `formState` carries it.
 */
export function useConfigNameStatus(): { preview: string; status: NameCheckStatus } {
  const workspace = useWorkspaceFromPath();
  const { control } = useFormContext<EvaluationFormValues>();
  const name = useWatch({ control, name: 'name' });

  const preview = toValidEntityName(name ?? '', '');
  const [debouncedName] = useDebounce(preview, DEFAULT_DEBOUNCE_MS);
  const conflictQuery = useFilesListFilesets(
    workspace,
    { page_size: 1, filter: { name: debouncedName } },
    { query: { enabled: Boolean(debouncedName) } }
  );

  return { preview, status: nameCheckStatus(preview, debouncedName, conflictQuery) };
}
