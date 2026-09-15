// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, swallowConflict } from '@nemo/common/src/api/common/utils';
import { toError } from '@nemo/common/src/utils/logger';
import {
  getFilesListFilesetsQueryKey,
  useFilesCreateFileset,
  useFilesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import {
  getModelsListModelsQueryKey,
  useModelsCreateModel,
} from '@nemo/sdk/generated/platform/models';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import type { CustomizationTemplate } from '@studio/constants/customizationTemplates';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { fetchAndConvertDataset } from '@studio/util/huggingFaceDataset';
import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

interface UseTemplateSetupResult {
  /**
   * Registers the template's models and dataset in the workspace, then resolves to the
   * form values seeded from it — or null if setup failed, in which case {@link error}
   * says why.
   */
  run: (template: CustomizationTemplate) => Promise<CustomizationFormFields | null>;
  /** Non-empty while setup is running; also the label to show on the button. */
  statusLabel: string;
  error: string | null;
}

/**
 * Puts a template's prerequisites in place: base models registered as entities, and its
 * HuggingFace dataset fetched, converted and uploaded as a fileset. Creates are wrapped in
 * `swallowConflict`, so re-running a template already set up is a no-op.
 */
export const useTemplateSetup = (workspace: string): UseTemplateSetupResult => {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [statusLabel, setStatusLabel] = useState('');

  const { mutateAsync: createFileset } = useFilesCreateFileset();
  const { mutateAsync: uploadFile } = useFilesUploadFile();
  const { mutateAsync: createModel } = useModelsCreateModel();

  const run = async (template: CustomizationTemplate): Promise<CustomizationFormFields | null> => {
    setError(null);
    setStatusLabel('Setting up…');
    try {
      for (const model of template.models) {
        setStatusLabel(`Registering ${model.name}…`);
        await swallowConflict(
          createFileset({
            workspace,
            data: {
              name: model.name,
              purpose: FilesetPurpose.model,
              storage: {
                type: 'huggingface',
                repo_id: model.hfRepoId,
                repo_type: 'model',
                ...(model.requiresHfToken ? { token_secret: 'hf-token' } : {}),
              },
            },
          })
        );

        const modelEntity = {
          name: model.name,
          fileset: `${workspace}/${model.name}`,
          ...(model.trustRemoteCode !== undefined
            ? { trust_remote_code: model.trustRemoteCode }
            : {}),
        };
        // A conflict means something already holds this name. It is not necessarily ours —
        // a user who registered the same checkpoint themselves owns an entity we would be
        // repointing — so the existing one is left alone and reused as-is.
        await swallowConflict(createModel({ workspace, data: modelEntity }));
      }

      const datasetFiles = await fetchAndConvertDataset(
        queryClient,
        template.dataset,
        (fetched, total) => setStatusLabel(`Fetching dataset (${fetched}/${total})…`)
      );

      setStatusLabel('Uploading dataset…');
      await swallowConflict(
        createFileset({
          workspace,
          data: { name: template.dataset.name, purpose: FilesetPurpose.dataset },
        })
      );

      await uploadFile({
        workspace,
        name: template.dataset.name,
        path: 'training.jsonl',
        data: datasetFiles.training,
      });
      await uploadFile({
        workspace,
        name: template.dataset.name,
        path: 'validation.jsonl',
        data: datasetFiles.validation,
      });

      await Promise.all([
        queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) }),
        queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) }),
      ]);

      return template.buildFormSpec(workspace, `${workspace}/${template.dataset.name}`);
    } catch (e) {
      setError(getErrorMessage(toError(e), 'Failed to set up template'));
      return null;
    } finally {
      setStatusLabel('');
    }
  };

  return { run, statusLabel, error };
};
