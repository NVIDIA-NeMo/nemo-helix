// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { deleteExperiment } from '@nemo/sdk/generated/platform/experiments';
import {
  Button,
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
} from '@nvidia/foundations-react-core';
import { invalidateExperimentDeletion } from '@studio/api/evaluation/invalidateExperimentDeletion';
import { useQueryClient } from '@tanstack/react-query';
import { EllipsisVertical, Trash } from 'lucide-react';
import { useState } from 'react';

export const EXPERIMENT_DELETE_DESCRIPTION =
  'Evaluations belonging only to the selected experiments will also be deleted, including evaluations from other agents. Evaluations that also belong to an experiment outside the selection are preserved. This cannot be undone.';

interface ExperimentDeleteButtonProps {
  workspace: string;
  name: string;
  onDeleted?: () => void;
  asMenu?: boolean;
}

export const ExperimentDeleteButton = ({
  workspace,
  name,
  onDeleted,
  asMenu = false,
}: ExperimentDeleteButtonProps) => {
  const [open, setOpen] = useState(false);
  const client = useQueryClient();

  return (
    <div onClick={(event) => event.stopPropagation()}>
      {asMenu ? (
        <DropdownRoot>
          <DropdownTrigger asChild showChevron={false}>
            <Button kind="tertiary" aria-label={`More actions for experiment ${name}`}>
              <EllipsisVertical />
            </Button>
          </DropdownTrigger>
          <DropdownContent align="end">
            <DropdownItem onClick={() => setOpen(true)} danger>
              Delete
            </DropdownItem>
          </DropdownContent>
        </DropdownRoot>
      ) : (
        <Button
          kind="secondary"
          color="danger"
          aria-label={`Delete experiment ${name}`}
          onClick={() => setOpen(true)}
        >
          <Trash /> Delete
        </Button>
      )}
      <DeleteConfirmationModal
        open={open}
        onClose={() => setOpen(false)}
        title={`Delete experiment “${name}”?`}
        description={EXPERIMENT_DELETE_DESCRIPTION}
        successText="Experiment deleted."
        onDelete={async () => {
          await deleteExperiment(workspace, name);
          onDeleted?.();
          await invalidateExperimentDeletion(client, workspace);
          return true;
        }}
      />
    </div>
  );
};
