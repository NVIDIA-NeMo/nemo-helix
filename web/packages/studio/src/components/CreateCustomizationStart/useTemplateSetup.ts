// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, swallowConflict } from '@nemo/common/src/api/common/utils';
import { toError } from '@nemo/common/src/utils/logger';
import {
  filesRetrieveFileset,
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

const toMegabytes = (bytes: number): string => (bytes / 1024 / 1024).toFixed(1);

/**
 * Puts a template's prerequisites in place: base models registered as entities, and its
 * HuggingFace dataset registered as an external fileset, then read back through the files
 * service, converted, and uploaded as a second fileset. Creates are wrapped in
 * `swallowConflict`, so re-running a template already set up is a no-op.
 *
 * The dataset rows come through the platform rather than from huggingface.co directly,
 * because the browser is not assumed to have egress to HuggingFace.
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

    // Download progress fires once per network chunk, but the label only changes every tenth
    // of a megabyte. Dropping the repeats keeps a multi-MB download from re-rendering the
    // recipe grid hundreds of times to paint the same string.
    let shownLabel = '';
    const setLabel = (next: string) => {
      if (next === shownLabel) return;
      shownLabel = next;
      setStatusLabel(next);
    };

    setLabel('Setting up…');
    try {
      for (const model of template.models) {
        setLabel(`Registering ${model.name}…`);
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

      const { dataset } = template;

      // Creating this fileset costs three sequential HuggingFace round trips server-side
      // (validate_storage, then resolve_config to pin the revision to a commit SHA), so it
      // gets its own label rather than sitting silently behind the previous one.
      setLabel('Connecting to Hugging Face…');
      const createdSource = await swallowConflict(
        createFileset({
          workspace,
          data: {
            name: dataset.sourceFilesetName,
            purpose: FilesetPurpose.dataset,
            description: `Raw Hugging Face source data for the Customizer quick-start recipes. Train on "${dataset.name}" instead, which holds the converted rows.`,
            storage: {
              type: 'huggingface',
              repo_id: dataset.hfRepoId,
              repo_type: 'dataset',
              ...(dataset.requiresHfToken ? { token_secret: 'hf-token' } : {}),
            },
          },
        })
      );

      // `swallowConflict` yields undefined exactly on a 409, which says only that the name is
      // taken — not that it is taken by our repo. Unlike the model filesets above (a mismatch
      // there fails loudly at training), a foreign dataset fileset fails silently: nothing
      // matches the file pattern and the user gets an error that never mentions ownership.
      // It is deliberately not repointed or deleted; it may be the user's own.
      if (!createdSource) {
        const existing = await filesRetrieveFileset(workspace, dataset.sourceFilesetName);
        const { storage } = existing;
        if (storage.type !== 'huggingface' || storage.repo_id !== dataset.hfRepoId) {
          const points =
            storage.type === 'huggingface' ? `"${storage.repo_id}"` : `${storage.type} storage`;
          throw new Error(
            `Fileset "${workspace}/${dataset.sourceFilesetName}" already exists and points at ${points}, not "${dataset.hfRepoId}". Rename or remove it, then try again.`
          );
        }
      }

      const datasetFiles = await fetchAndConvertDataset(
        queryClient,
        workspace,
        dataset,
        (phase, loadedBytes, totalBytes) => {
          if (phase === 'locating') {
            setLabel('Locating dataset file…');
          } else if (phase === 'downloading') {
            setLabel(
              totalBytes
                ? `Downloading dataset (${toMegabytes(loadedBytes ?? 0)}/${toMegabytes(totalBytes)} MB)…`
                : 'Downloading dataset…'
            );
          } else {
            setLabel('Preparing dataset…');
          }
        }
      );

      setLabel('Uploading dataset…');
      const createdTarget = await swallowConflict(
        createFileset({
          workspace,
          data: { name: dataset.name, purpose: FilesetPurpose.dataset },
        })
      );

      // Same reasoning as the source fileset, with a sharper failure mode. Only local
      // storage accepts writes — every external backend rejects upload server-side — so a
      // name already held by, say, a HuggingFace-backed fileset would surface as a raw
      // backend error from the uploads below rather than as the name collision it is.
      if (!createdTarget) {
        const existing = await filesRetrieveFileset(workspace, dataset.name);
        if (existing.storage.type !== 'local') {
          throw new Error(
            `Fileset "${workspace}/${dataset.name}" already exists on ${existing.storage.type} storage, which cannot be written to. Rename or remove it, then try again.`
          );
        }
      }

      await uploadFile({
        workspace,
        name: dataset.name,
        path: 'training.jsonl',
        data: datasetFiles.training,
      });
      await uploadFile({
        workspace,
        name: dataset.name,
        path: 'validation.jsonl',
        data: datasetFiles.validation,
      });

      await Promise.all([
        queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) }),
        queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) }),
      ]);

      return template.buildFormSpec(workspace, `${workspace}/${dataset.name}`);
    } catch (e) {
      setError(getErrorMessage(toError(e), 'Failed to set up template'));
      return null;
    } finally {
      setStatusLabel('');
    }
  };

  return { run, statusLabel, error };
};
