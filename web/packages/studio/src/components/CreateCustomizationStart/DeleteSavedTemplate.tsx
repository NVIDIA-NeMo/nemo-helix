// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useCustomizationDeleteJobTemplate } from '@nemo/sdk/generated/customizer/customization-job-templates';
import type { CustomizationJobTemplate } from '@nemo/sdk/generated/customizer/schema';
import { Banner, Button, Flex, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import { Trash2 } from 'lucide-react';
import { useState, type FC } from 'react';

interface Props {
  workspace: string;
  template: CustomizationJobTemplate;
  /** Called once the delete lands, so the list can refetch. */
  onDeleted: () => void;
}

/**
 * Deletes the selected saved template, from the page footer rather than the card: an
 * interactive Card is a single click target and may not carry its own buttons.
 */
export const DeleteSavedTemplate: FC<Props> = ({ workspace, template, onDeleted }) => {
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { mutateAsync: deleteTemplate, isPending } = useCustomizationDeleteJobTemplate();

  const name = template.name ?? template.id;

  const remove = async () => {
    setError(null);
    try {
      await deleteTemplate({ workspace, name });
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
        onClick={() => {
          setError(null);
          setConfirming(true);
        }}
      >
        <Trash2 size={16} aria-hidden />
        Delete
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
