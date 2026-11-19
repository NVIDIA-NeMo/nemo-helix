// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { Trash } from '@nemo/icons/icons';
import {
  getCustomizationListJobTemplatesQueryKey,
  useCustomizationDeleteJobTemplate,
} from '@nemo/sdk/generated/customizer/customization-job-templates';
import type { CustomizationJobTemplate } from '@nemo/sdk/generated/customizer/schema';
import { Button } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { useState, type FC } from 'react';

interface Props {
  workspace: string;
  /** The saved template's name, which is also its key. */
  name: string;
  onDeleted: () => void;
  disabled?: boolean;
}

/** Deletes the saved template the form was started from, after a confirmation. */
export const DeleteSavedTemplate: FC<Props> = ({ workspace, name, onDeleted, disabled }) => {
  const [confirming, setConfirming] = useState(false);
  const queryClient = useQueryClient();
  const { mutateAsync: deleteTemplate } = useCustomizationDeleteJobTemplate();

  const remove = async () => {
    await deleteTemplate({ workspace, name });
    // Dropped from the cache before the refetch lands, so the start page never shows it.
    const queryKey = getCustomizationListJobTemplatesQueryKey(workspace);
    queryClient.setQueryData<CustomizationJobTemplate[]>(queryKey, (templates) =>
      templates?.filter((template) => (template.name ?? template.id) !== name)
    );
    void queryClient.invalidateQueries({ queryKey });
    onDeleted();
    return true;
  };

  return (
    <>
      <Button
        kind="tertiary"
        color="danger"
        disabled={disabled}
        onClick={() => setConfirming(true)}
      >
        <Trash size="16" aria-hidden />
        Delete template
      </Button>

      {confirming && (
        <DeleteConfirmationModal
          open
          entity="template"
          ids={[name]}
          description="The saved template is deleted, but jobs already started from it aren't affected."
          onDelete={remove}
          onClose={() => setConfirming(false)}
        />
      )}
    </>
  );
};
