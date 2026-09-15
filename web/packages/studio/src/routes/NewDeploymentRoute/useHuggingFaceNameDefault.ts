// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { huggingFaceRepoIdToBaseName } from '@studio/routes/DeploymentsListRoute/huggingFaceDeploymentArtifacts';
import { SOURCE_HF, type WizardFormValues } from '@studio/routes/NewDeploymentRoute/schema';
import { useEffect, useState } from 'react';
import { useFormState, useWatch, type Control, type UseFormSetValue } from 'react-hook-form';

/**
 * Keeps the wizard's Name field in step with the Hugging Face Repo ID.
 *
 * Applies only to the HuggingFace source; NGC and Workspace keep the random
 * `generateDefaultName()` default. Derivation stops for good once the user edits
 * the name themselves, so a deliberate choice is never overwritten mid-typing.
 */
export function useHuggingFaceNameDefault(
  control: Control<WizardFormValues>,
  setValue: UseFormSetValue<WizardFormValues>
): void {
  const source = useWatch({ control, name: 'source' });
  const repoId = useWatch({ control, name: 'repoId' });
  const { dirtyFields, defaultValues } = useFormState({ control });

  // `dirtyFields.name` is value-based, not event-based: RHF drops the entry the
  // moment the field matches its default again. Reading it directly would hand
  // the name back to the repo ID as soon as a user undid their way back to the
  // generated default, so ownership is latched rather than read live.
  const [nameOwnedByUser, setNameOwnedByUser] = useState(false);

  // `reset()` installs fresh defaultValues, including a newly generated name.
  // That is the wizard starting over, and the only thing that returns ownership
  // of the name to this hook.
  const defaultName = defaultValues?.name;
  const [lastDefaultName, setLastDefaultName] = useState(defaultName);

  if (lastDefaultName !== defaultName) {
    setLastDefaultName(defaultName);
    setNameOwnedByUser(Boolean(dirtyFields.name));
  } else if (dirtyFields.name && !nameOwnedByUser) {
    setNameOwnedByUser(true);
  }

  useEffect(() => {
    if (source !== SOURCE_HF || nameOwnedByUser) return;

    // Only overwrite once the repo ID yields something; leaves the existing
    // default in place rather than blanking the field while the user is still
    // part-way through typing.
    const derived = huggingFaceRepoIdToBaseName(repoId ?? '');
    if (derived) {
      setValue('name', derived, { shouldDirty: false, shouldValidate: true });
    }
  }, [nameOwnedByUser, repoId, setValue, source]);
}
