// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledCombobox } from '@nemo/common/src/components/form/ControlledCombobox';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useEvalsListTasks } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import { useEvalsCreateTaskset } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import { Stack } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getTasksetDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const schema = z.object({
  name: z
    .string()
    .refine((value) => sanitizeEntityName(value) !== undefined, {
      message: 'Name must start with a letter and be at least 2 characters.',
    })
    .transform((value) => toValidEntityName(value, value)),
  description: z.string(),
  tasks: z.array(z.string()).min(1, 'Select at least one task.'),
});

type FormInput = z.input<typeof schema>;
type FormOutput = z.output<typeof schema>;

const DEFAULT_VALUES: FormInput = { name: '', description: '', tasks: [] };

interface Props {
  open: boolean;
  onClose: () => void;
}

export const TasksetFormModal: FC<Props> = ({ open, onClose }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: tasks, isPending: tasksPending } = useEvalsListTasks(
    workspace,
    { page_size: 1000, sort: 'name' },
    { query: { enabled: open } }
  );

  const {
    control,
    handleSubmit,
    reset,
    watch,
    formState: { errors },
  } = useForm<FormInput, unknown, FormOutput>({
    resolver: zodResolver(schema),
    defaultValues: DEFAULT_VALUES,
    mode: 'onTouched',
  });

  const preview = toValidEntityName(watch('name'), '');
  const {
    mutateAsync: createTaskset,
    isPending,
    error,
    reset: resetMutation,
  } = useEvalsCreateTaskset();

  const handleClose = () => {
    reset(DEFAULT_VALUES);
    resetMutation();
    onClose();
  };

  const onSubmit = async ({ name, description, tasks }: FormOutput) => {
    try {
      await createTaskset({
        workspace,
        name,
        data: { tasks, ...(description.trim() && { description: description.trim() }) },
      });
    } catch {
      // Surfaced through `errorText`; the modal stays open.
      return;
    }
    await queryClient.invalidateQueries({
      queryKey: [`/apis/evals/v2/workspaces/${workspace}/tasksets`],
    });
    handleClose();
    navigate(getTasksetDetailRoute(workspace, name));
  };

  return (
    <FormModal
      open={open}
      title="Create Taskset"
      submitButtonText="Create"
      disabled={isPending}
      loading={isPending}
      errorText={error ? getErrorMessage(error) : null}
      onSubmit={handleSubmit(onSubmit)}
      onClose={handleClose}
    >
      <Stack gap="density-lg">
        <ControlledTextInput
          label="Name"
          disabled={isPending}
          useControllerProps={{ name: 'name', control }}
          formFieldProps={{
            slotHelp:
              !errors.name && preview ? (
                <>
                  Your taskset will be created as <span className="text-primary">{preview}</span>
                </>
              ) : undefined,
            slotError: errors.name?.message,
            status: errors.name ? 'error' : undefined,
          }}
          attributes={{ Input: { autoComplete: 'off', spellCheck: false } }}
        />
        <ControlledTextArea
          label="Description (optional)"
          disabled={isPending}
          rows={2}
          useControllerProps={{ name: 'description', control }}
        />
        <ControlledCombobox
          kind="multiple"
          label="Tasks"
          disabled={isPending}
          loading={tasksPending}
          items={tasks?.data.map((task) => task.name) ?? []}
          placeholder="Select tasks..."
          emptyStateMessage={tasksPending ? 'Loading tasks...' : 'No tasks in this workspace.'}
          useControllerProps={{ name: 'tasks', control }}
          formFieldProps={{
            slotHelp: 'Each member is pinned to its latest revision when the taskset is created.',
          }}
        />
      </Stack>
    </FormModal>
  );
};
