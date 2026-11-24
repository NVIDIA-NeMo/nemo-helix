// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import type { EvaluationFormValues } from '@studio/components/evaluation/EvalWizard/types';
import {
  nameCheckParams,
  useFilesetNameStatus,
} from '@studio/components/filesets/hooks/useFilesetNameStatus';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useFormContext, useWatch } from 'react-hook-form';

const TAKEN_BY: Partial<Record<FilesetPurpose, string>> = {
  [FilesetPurpose.dataset]: 'a dataset',
  [FilesetPurpose.model]: 'a model',
};

/**
 * The error for a configuration name another fileset holds. Configurations and datasets share
 * the fileset namespace, so a dataset's name is taken even though no configuration uses it.
 * A generic fileset may or may not be a saved configuration, so it is only called a fileset.
 */
export const configNameTakenMessage = (
  purpose: FilesetPurpose | undefined,
  workspace: string
): string =>
  `This name is already used by ${(purpose && TAKEN_BY[purpose]) ?? 'a fileset'} in the ` +
  `${workspace} workspace. Choose a different configuration name.`;

/**
 * Whether the typed configuration name is free.
 *
 * Shared by the field that shows it and the step that blocks on it: the name
 * becomes the fileset name verbatim, and the check is a fetch rather than a
 * schema rule, so nothing in `formState` carries it.
 */
export function useConfigNameStatus(): {
  preview: string;
  status: NameCheckStatus;
  conflictMessage?: string;
} {
  const workspace = useWorkspaceFromPath();
  const { control } = useFormContext<EvaluationFormValues>();
  const name = useWatch({ control, name: 'name' });

  // Unlike the fileset create forms, this field sanitizes what is typed, so the check runs
  // on the sanitized preview rather than only on an already-valid name.
  const preview = toValidEntityName(name ?? '', '');
  const status = useFilesetNameStatus(workspace, preview);

  // Same query key as the check above, so a conflict reads the fileset it already found.
  const conflict = useFilesListFilesets(workspace, nameCheckParams(preview), {
    query: { enabled: status === 'conflict', staleTime: 0, retry: false },
  });

  return {
    preview,
    status,
    conflictMessage:
      status === 'conflict'
        ? configNameTakenMessage(conflict.data?.data?.[0]?.purpose, workspace)
        : undefined,
  };
}
