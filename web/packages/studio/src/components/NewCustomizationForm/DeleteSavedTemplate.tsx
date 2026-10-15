// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import {
  getCustomizationListJobTemplatesQueryKey,
  useCustomizationDeleteJobTemplate,
} from '@nemo/sdk/generated/customizer/customization-job-templates';
import type { CustomizationJobTemplate } from '@nemo/sdk/generated/customizer/schema';
import { Banner, Button, Flex, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { Trash2 } from 'lucide-react';
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
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { mutateAsync: deleteTemplate, isPending } = useCustomizationDeleteJobTemplate();

  const remove = async () => {
    setError(null);
    try {
      await deleteTemplate({ workspace, name });
      // Dropped from the cache before the refetch lands, so the start page never shows it.
      const queryKey = getCustomizationListJobTemplatesQueryKey(workspace);
      queryClient.setQueryData<CustomizationJobTemplate[]>(queryKey, (templates) =>
        templates?.filter((template) => (template.name ?? template.id) !== name)
      );
      void queryClient.invalidateQueries({ queryKey });
      onDeleted();
      setConfirming(false);
    } catch (e) {
      // Keep the dialog open on failure rather than closing over it.
      setError(e instanceof Error ? e.message : 'Could not delete the template.');
    }
  };

  return (
    <>
      <Button
        kind="tertiary"
        color="danger"
        disabled={disabled}
        onClick={() => {
          setError(null);
          setConfirming(true);
        }}
      >
        <Trash2 size={16} aria-hidden />
        Delete template
      </Button>

      <Modal
        open={confirming}
        onOpenChange={(next) => !next && setConfirming(false)}
        slotHeading={`Delete ${name}?`}
      >
        <Stack gap="density-xl">
          {error ? (
            <Banner kind="inline" status="error">
              {error}
            </Banner>
          ) : null}

          <Text kind="body/regular/md">
            This removes the saved template. Jobs already started from it are not affected.
          </Text>

          <Flex justify="end" gap="density-md">
            <Button kind="tertiary" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
            <LoadingButton color="danger" loading={isPending} onClick={() => void remove()}>
              Delete
            </LoadingButton>
          </Flex>
        </Stack>
      </Modal>
    </>
  );
};
