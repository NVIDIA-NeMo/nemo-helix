// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { getEntityNameError } from '@nemo/common/src/utils/entityName';
import {
  getCustomizationListJobTemplatesQueryKey,
  useCustomizationCreateJobTemplate,
} from '@nemo/sdk/generated/customizer/customization-job-templates';
import {
  Banner,
  Button,
  Flex,
  FormField,
  Modal,
  Stack,
  TextArea,
  TextInput,
} from '@nvidia/foundations-react-core';
import { getCustomizationBackend, type CustomizationJob } from '@studio/util/customizationBackend';
import { useQueryClient } from '@tanstack/react-query';
import { useState, type FC } from 'react';

interface Props {
  open: boolean;
  onClose: () => void;
  workspace: string;
  /** The job to save. Its spec is already the shape a template stores. */
  job: CustomizationJob;
  onSaved?: (name: string) => void;
}

export const SaveTemplateModal: FC<Props> = ({ open, onClose, workspace, job, onSaved }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  // Deferred to blur, so the message does not fire on the first keystroke.
  const [nameTouched, setNameTouched] = useState(false);
  const { mutateAsync: createTemplate, isPending } = useCustomizationCreateJobTemplate();
  const queryClient = useQueryClient();

  // A template is an entity, so its name has to satisfy the same rules as any other.
  const nameError = getEntityNameError(name.trim());

  const save = async () => {
    setError(null);
    setNameTouched(true);
    if (nameError) return;
    const backend = getCustomizationBackend(job.spec);
    if (!backend) {
      setError('This job has no recognisable training backend, so it cannot be saved.');
      return;
    }
    try {
      await createTemplate({
        workspace,
        data: {
          name: name.trim(),
          backend,
          config: job.spec as unknown as Record<string, unknown>,
          description,
        },
      });
      // Or the start page shows a stale list on the next visit.
      await queryClient.invalidateQueries({
        queryKey: getCustomizationListJobTemplatesQueryKey(workspace),
      });
      onSaved?.(name.trim());
      onClose();
    } catch (e) {
      // Stay open with the message: the name is what the backend rejects.
      setError(e instanceof Error ? e.message : 'Could not save the template.');
    }
  };

  return (
    <Modal open={open} onOpenChange={(next) => !next && onClose()} slotHeading="Save as template">
      <Stack gap="density-xl">
        {error ? (
          <Banner kind="inline" status="error">
            {error}
          </Banner>
        ) : null}

        <FormField
          slotLabel="Name"
          required
          slotError={nameTouched ? nameError : undefined}
          status={nameTouched && nameError ? 'error' : undefined}
        >
          <TextInput
            value={name}
            onValueChange={setName}
            onBlur={() => setNameTouched(true)}
            status={nameTouched && nameError ? 'error' : undefined}
          />
        </FormField>

        <FormField slotLabel="Description">
          <TextArea value={description} onValueChange={setDescription} rows={3} />
        </FormField>

        <Flex justify="end" gap="density-md">
          <Button kind="tertiary" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            color="brand"
            loading={isPending}
            disabled={nameError !== undefined}
            onClick={() => void save()}
          >
            Save
          </LoadingButton>
        </Flex>
      </Stack>
    </Modal>
  );
};
