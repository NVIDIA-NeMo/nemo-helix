// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
import { ControlledCombobox } from '@nemo/common/src/components/form/ControlledCombobox';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  evalsListTasks,
  getEvalsListTasksQueryKey,
} from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import {
  useEvalsCreateTaskset,
  useEvalsReplaceTaskset,
} from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Taskset, TasksetInput } from '@nemo/sdk/generated/evals/schema';
import { Button, Stack } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { parseTaskRef } from '@studio/routes/evaluation/taskHelpers';
import { invalidateTasksets } from '@studio/routes/evaluation/tasksets/helpers';
import { getTasksetDetailRoute } from '@studio/routes/utils';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

// A files-backed taskset may have no members, so only enforce one when there's no `files_ref`.
const getSchema = (minTasks: number) =>
  z.object({
    name: z
      .string()
      .refine((value) => sanitizeEntityName(value) !== undefined, {
        message: 'Name must start with a letter and be at least 2 characters.',
      })
      .transform((value) => toValidEntityName(value, value)),
    description: z.string(),
    tasks: z.array(z.string()).min(minTasks, 'Select at least one task.'),
  });

type FormInput = z.input<ReturnType<typeof getSchema>>;
type FormOutput = z.output<ReturnType<typeof getSchema>>;

interface Props {
  open: boolean;
  onClose: () => void;
  /** When set, the modal edits this taskset (publishing a new revision) instead of creating one. */
  taskset?: Taskset;
}

export const TasksetFormModal: FC<Props> = ({ open, onClose, taskset }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const defaultValues: FormInput = {
    name: taskset?.name ?? '',
    description: taskset?.description ?? '',
    tasks: taskset?.tasks ?? [],
  };

  // Drained so tasks past the first page stay selectable; keyed under the list key so task
  // writes invalidate it.
  const {
    data: taskNames,
    isPending: tasksPending,
    isError: tasksError,
    refetch: refetchTasks,
  } = useQuery({
    queryKey: [...getEvalsListTasksQueryKey(workspace), 'all-names'],
    queryFn: async () =>
      (
        await fetchAllPages((page, page_size) =>
          evalsListTasks(workspace, { page, page_size, sort: 'name' })
        )
      ).map((task) => task.name),
    enabled: open,
  });

  // Current members keep their stored (digest-pinned) refs so an edit doesn't silently move them
  // to latest; tasks added here go in bare and the server pins them to their latest revision.
  const members = taskset?.tasks ?? [];
  const memberKeys = new Set(
    members.map((ref) => {
      const { workspace: ws, name } = parseTaskRef(ref, workspace);
      return `${ws}/${name}`;
    })
  );
  const items = [
    ...members.map((ref) => ({ value: ref, children: parseTaskRef(ref, workspace).name })),
    ...(taskNames ?? []).filter((name) => !memberKeys.has(`${workspace}/${name}`)),
  ];

  const {
    control,
    handleSubmit,
    reset,
    watch,
    formState: { errors, isDirty, isSubmitting },
  } = useForm<FormInput, unknown, FormOutput>({
    resolver: zodResolver(getSchema(taskset?.files_ref ? 0 : 1)),
    defaultValues,
    mode: 'onTouched',
  });

  const preview = toValidEntityName(watch('name'), '');
  const create = useEvalsCreateTaskset();
  const replace = useEvalsReplaceTaskset();
  const { error } = taskset ? replace : create;

  const handleClose = () => {
    reset(defaultValues);
    create.reset();
    replace.reset();
    onClose();
  };

  const onSubmit = async ({ name, description, tasks }: FormOutput) => {
    // PUT replaces the whole record, so carry over the fields this form doesn't edit.
    const data: TasksetInput = {
      tasks,
      files_ref: taskset?.files_ref,
      metadata: taskset?.metadata,
      ...(description.trim() && { description: description.trim() }),
    };
    try {
      await (taskset ? replace : create).mutateAsync({ workspace, name, data });
    } catch {
      // Surfaced through `errorText`; the modal stays open.
      return;
    }
    await invalidateTasksets(queryClient, workspace);
    handleClose();
    if (!taskset) navigate(getTasksetDetailRoute(workspace, name));
  };

  return (
    <FormModal
      open={open}
      title={taskset ? `Edit Taskset: ${taskset.name}` : 'Create Taskset'}
      submitButtonText={taskset ? 'Save' : 'Create'}
      disabled={isSubmitting}
      loading={isSubmitting}
      submitDisabled={!!taskset && !isDirty}
      errorText={error ? getErrorMessage(error) : null}
      onSubmit={handleSubmit(onSubmit)}
      onClose={handleClose}
    >
      <Stack gap="density-lg">
        <ControlledTextInput
          label="Name"
          disabled={isSubmitting || !!taskset}
          useControllerProps={{ name: 'name', control }}
          formFieldProps={{
            slotHelp:
              !taskset && !errors.name && preview ? (
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
          disabled={isSubmitting}
          rows={2}
          useControllerProps={{ name: 'description', control }}
        />
        <ControlledCombobox
          kind="multiple"
          label="Tasks"
          disabled={isSubmitting}
          loading={tasksPending}
          items={items}
          placeholder="Select tasks..."
          emptyStateMessage={tasksPending ? 'Loading tasks...' : 'No tasks in this workspace.'}
          useControllerProps={{ name: 'tasks', control }}
          formFieldProps={{
            slotHelp: taskset
              ? 'Saving publishes a new revision. Current members stay on their pinned revision; added tasks are pinned to their latest.'
              : 'Each member is pinned to its latest revision when the taskset is created.',
            ...(tasksError && {
              status: 'error',
              slotError: (
                <>
                  Couldn&apos;t load tasks.{' '}
                  <Button type="button" kind="tertiary" size="small" onClick={() => refetchTasks()}>
                    Retry
                  </Button>
                </>
              ),
            }),
          }}
        />
      </Stack>
    </FormModal>
  );
};
