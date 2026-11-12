// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CreateDatasetPanel } from '@studio/components/CreateDatasetPanel';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { FilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import { getDatasetDetailRoute } from '@studio/routes/utils';
import type { StorageBackend } from '@studio/util/storageBackend';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

// Not isRepositoryBackend: that only covers github/git, but HuggingFace and NGC datasets are external too.
const EXTERNAL_BACKENDS: ReadonlySet<StorageBackend | null | undefined> = new Set([
  'huggingface',
  'ngc',
  'github',
]);

/**
 * The dataset creation panel for the current workspace. After creating, opens the dataset:
 * the Card tab for external-source datasets, the Files tab otherwise.
 */
export const NewDatasetPanel: FC<{ open: boolean; onClose: () => void }> = ({ open, onClose }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  return (
    <CreateDatasetPanel
      open={open}
      workspace={workspace}
      onClose={onClose}
      onCreated={(fileset) => {
        onClose();
        navigate(
          getDatasetDetailRoute(fileset.workspace, fileset.name, {
            tab: EXTERNAL_BACKENDS.has(fileset.storage.type)
              ? FilesetDetailTab.Card
              : FilesetDetailTab.Files,
          })
        );
      }}
    />
  );
};
