// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, swallowConflict } from '@nemo/common/src/api/common/utils';
import { toError } from '@nemo/common/src/utils/logger';
import {
  getFilesListFilesetsQueryKey,
  useFilesCreateFileset,
  useFilesDeleteFileset,
  useFilesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import {
  getModelsListModelsQueryKey,
  useModelsCreateModel,
} from '@nemo/sdk/generated/platform/models';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import {
  findAvailableFilesetName,
  preflightConvertedFileset,
  preflightSourceFileset,
  type TemplateFilesetConflict,
} from '@studio/components/CreateCustomizationStart/templatePreflight';
import { useDownloadFileHead } from '@studio/components/filesets/hooks/useDownloadFileHead';
import type { CustomizationTemplate } from '@studio/constants/customizationTemplates';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { fetchAndConvertDataset } from '@studio/util/huggingFaceDataset';
import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

/**
 * How the user chose to get past a {@link TemplateFilesetConflict}.
 *
 * The conflict travels with the choice rather than being read back off hook state: it is
 * what names the fileset to rename or delete, and reading it from state would make a
 * destructive action depend on a closure the caller cannot see.
 */
export interface ConflictResolution {
  /**
   * `rename` sets the colliding fileset aside and builds the template's data under a free
   * name; `replace` deletes the colliding fileset and rebuilds it from the recipe.
   */
  action: 'rename' | 'replace';
  conflict: TemplateFilesetConflict;
}

export interface TemplateSetupRun {
  values: CustomizationFormFields;
  /**
   * Fileset setup reused as-is instead of rebuilding, or null when it uploaded fresh data.
   * Reported to the user, since it decides what the job actually trains on.
   */
  reusedFilesetRef: string | null;
}

interface UseTemplateSetupResult {
  /**
   * Registers the template's models and dataset in the workspace, then resolves to the
   * form values seeded from it — or null if setup did not finish, in which case either
   * {@link error} or {@link conflict} says why.
   */
  run: (
    template: CustomizationTemplate,
    resolution?: ConflictResolution
  ) => Promise<TemplateSetupRun | null>;
  /** Non-empty while setup is running; also the label to show on the button. */
  statusLabel: string;
  error: string | null;
  /** Set when a fileset name is taken by something setup must not touch unasked. */
  conflict: TemplateFilesetConflict | null;
  clearConflict: () => void;
}

const toMegabytes = (bytes: number): string => (bytes / 1024 / 1024).toFixed(1);

/**
 * Puts a template's prerequisites in place: base models registered as entities, and its
 * HuggingFace dataset registered as an external fileset, then read back through the files
 * service, converted, and uploaded as a second fileset.
 *
 * Re-running a template that is already set up is the common case, not the exception —
 * recipes are shared and `default` is shared — so both filesets are checked up front. A
 * compatible one is reused and the multi-megabyte download is skipped entirely; an
 * incompatible one stops setup *before* the download, as a {@link conflict} the user can
 * act on rather than a dead-end error.
 *
 * The dataset rows come through the platform rather than from huggingface.co directly,
 * because the browser is not assumed to have egress to HuggingFace.
 */
export const useTemplateSetup = (workspace: string): UseTemplateSetupResult => {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<TemplateFilesetConflict | null>(null);
  const [statusLabel, setStatusLabel] = useState('');

  const { mutateAsync: createFileset } = useFilesCreateFileset();
  const { mutateAsync: uploadFile } = useFilesUploadFile();
  const { mutateAsync: deleteFileset } = useFilesDeleteFileset();
  const { mutateAsync: createModel } = useModelsCreateModel();
  const downloadFileHead = useDownloadFileHead();

  const run = async (
    template: CustomizationTemplate,
    resolution?: ConflictResolution
  ): Promise<TemplateSetupRun | null> => {
    setError(null);
    setConflict(null);

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
      let sourceName = dataset.sourceFilesetName;
      let convertedName = dataset.name;

      // "Replace" is the one path that destroys data, so it only ever touches the single
      // fileset the user was shown, by the name they were shown it under.
      if (resolution?.action === 'replace') {
        setLabel('Removing the old fileset…');
        await deleteFileset({ workspace, name: resolution.conflict.name });
      }

      setLabel('Checking the workspace…');

      if (resolution?.action === 'rename') {
        if (resolution.conflict.target === 'source') {
          sourceName = await findAvailableFilesetName(workspace, dataset.sourceFilesetName);
        } else {
          convertedName = await findAvailableFilesetName(workspace, dataset.name);
        }
      }

      // Checked before the source fileset is registered and long before any bytes move:
      // if the converted rows are already here, none of that work is needed, and if they
      // cannot be written, the user should not wait for a download to find out.
      const converted = await preflightConvertedFileset({
        workspace,
        name: convertedName,
        readHead: (path) =>
          downloadFileHead({ workspace, datasetName: convertedName, path, bytes: 8192 }),
      });
      if (converted.kind === 'conflict') {
        setConflict(converted.conflict);
        return null;
      }
      if (converted.kind === 'reusable') {
        return {
          values: template.buildFormSpec(workspace, converted.filesetRef),
          reusedFilesetRef: converted.filesetRef,
        };
      }

      const source = await preflightSourceFileset(workspace, sourceName, dataset.hfRepoId);
      if (source.kind === 'conflict') {
        setConflict(source.conflict);
        return null;
      }

      if (source.kind === 'absent') {
        // Creating this fileset costs three sequential HuggingFace round trips server-side
        // (validate_storage, then resolve_config to pin the revision to a commit SHA), so
        // it gets its own label rather than sitting silently behind the previous one.
        setLabel('Connecting to Hugging Face…');
        await swallowConflict(
          createFileset({
            workspace,
            data: {
              name: sourceName,
              purpose: FilesetPurpose.dataset,
              description: `Raw Hugging Face source data for the Customizer quick-start recipes. Train on "${convertedName}" instead, which holds the converted rows.`,
              storage: {
                type: 'huggingface',
                repo_id: dataset.hfRepoId,
                repo_type: 'dataset',
                ...(dataset.requiresHfToken ? { token_secret: 'hf-token' } : {}),
              },
            },
          })
        );
      }

      const datasetFiles = await fetchAndConvertDataset(
        queryClient,
        workspace,
        { ...dataset, sourceFilesetName: sourceName },
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
      await swallowConflict(
        createFileset({
          workspace,
          data: { name: convertedName, purpose: FilesetPurpose.dataset },
        })
      );

      await uploadFile({
        workspace,
        name: convertedName,
        path: 'training.jsonl',
        data: datasetFiles.training,
      });
      await uploadFile({
        workspace,
        name: convertedName,
        path: 'validation.jsonl',
        data: datasetFiles.validation,
      });

      await Promise.all([
        queryClient.invalidateQueries({ queryKey: getModelsListModelsQueryKey(workspace) }),
        queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) }),
      ]);

      return {
        values: template.buildFormSpec(workspace, `${workspace}/${convertedName}`),
        reusedFilesetRef: null,
      };
    } catch (e) {
      setError(getErrorMessage(toError(e), 'Failed to set up template'));
      return null;
    } finally {
      setStatusLabel('');
    }
  };

  return { run, statusLabel, error, conflict, clearConflict: () => setConflict(null) };
};
