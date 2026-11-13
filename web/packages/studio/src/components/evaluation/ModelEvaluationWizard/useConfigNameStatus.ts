// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import type { EvaluationFormValues } from '@studio/components/evaluation/EvalWizard/types';
import { useFilesetNameStatus } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useFormContext, useWatch } from 'react-hook-form';

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

  // Unlike the fileset create forms, this field sanitizes what is typed, so the check runs
  // on the sanitized preview rather than only on an already-valid name.
  const preview = toValidEntityName(name ?? '', '');
  return { preview, status: useFilesetNameStatus(workspace, preview) };
}
