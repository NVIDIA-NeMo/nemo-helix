// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getListEvaluationsQueryKey,
  usePatchEvaluation,
} from '@nemo/sdk/generated/platform/evaluations';
import {
  getListExperimentsQueryKey,
  useCreateExperiment,
  useListExperiments,
} from '@nemo/sdk/generated/platform/experiments';
import {
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  TextArea,
  TextInput,
} from '@nvidia/foundations-react-core';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { DefaultSortControl } from '@studio/components/DefaultSortControl';
import { DEFAULT_SORT } from '@studio/components/DefaultSortControl/util';
import {
  experimentCreateSchema,
  type ExperimentCreateFormFields,
} from '@studio/components/ExperimentCreateModal/constants';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { Plus } from 'lucide-react';
import { type FC, type FormEvent, useEffect, useMemo, useState } from 'react';
import { useForm, type SubmitHandler } from 'react-hook-form';

// Sentinel Select value for the "create a new group" affordance. Reserved so it can't collide with a
// real group id (group ids are entity ids, never this literal).
const CREATE_NEW = '__create_new__';

export interface AddToGroupModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  /** Called after the selected evaluations are successfully added (e.g. to clear the row selection). */
  onSuccess?: () => void;
  workspace: string;
  /** The evaluations being curated into another group (the bulk row selection). */
  evaluations: EvaluationRow[];
}

/**
 * Adds one or more selected evaluations to another Experiment. Offers the workspace's groups in a
 * dropdown — excluding groups every selected evaluation already belongs to — plus a "Create new group"
 * option that reveals a name/description sub-form. On submit it either adds the evaluations to the
 * chosen group, or creates the group first and then adds them, then refreshes the board and toasts.
 *
 * Membership is owned by the evaluation side (there's no atomic "create group with evaluations"
 * endpoint), so the create path is two steps: create the group, then add each evaluation to it. A
 * created-but-partially-populated group is a valid state, so a failed add is surfaced as a warning
 * rather than rolled back.
 */
export const AddToGroupModal: FC<AddToGroupModalProps> = ({
  open,
  onClose,
  onSuccess,
  workspace,
  evaluations,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [selectedGroupId, setSelectedGroupId] = useState('');
  const isCreating = selectedGroupId === CREATE_NEW;
  // Default sort is a single `sort`-param string driven by a custom control (not a registered RHF
  // input), so it's managed here and merged into the create payload (mirrors ExperimentCreateModal).
  const [defaultSort, setDefaultSort] = useState<string>(DEFAULT_SORT);

  const {
    register,
    handleSubmit,
    reset,
    setValue,
    setError,
    formState: { errors, isValid },
  } = useForm<ExperimentCreateFormFields>({
    resolver: zodResolver(experimentCreateSchema),
    mode: 'onChange',
  });

  // Reset the selection and the create sub-form whenever the modal (re)opens so nothing carries over.
  useEffect(() => {
    if (open) {
      setSelectedGroupId('');
      setDefaultSort(DEFAULT_SORT);
      reset();
    }
  }, [open, reset]);

  // Only fetch groups while the modal is open. A single large page covers any realistic group count.
  const { data: groupsPage, isLoading } = useListExperiments(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: open && !!workspace } }
  );

  // Exclude a group only when EVERY selected evaluation already belongs to it (nothing to add there).
  const groups = useMemo(
    () =>
      (groupsPage?.data ?? []).filter(
        (group) => !evaluations.every((evaluation) => evaluation.experiment_ids.includes(group.id))
      ),
    [groupsPage, evaluations]
  );

  const groupNameById = useMemo(
    () => new Map(groups.map((group) => [group.id, group.name])),
    [groups]
  );

  const { mutateAsync: patchEvaluation, isPending: isAdding } = usePatchEvaluation();
  const { mutateAsync: createExperiment, isPending: isCreatingGroup } = useCreateExperiment();

  const busy = isAdding || isCreatingGroup;
  const count = evaluations.length;
  const countLabel = `${count} ${count === 1 ? 'evaluation' : 'evaluations'}`;

  // Add every selected evaluation to `groupId` by PATCHing its membership with the group merged in.
  // Membership is a replace, so send the full desired set (existing + new, de-duped); re-adding a group
  // the evaluation already belongs to is a no-op. Best-effort: returns how many failed rather than
  // throwing on the first error.
  const associateEvaluations = async (groupId: string): Promise<number> => {
    const results = await Promise.allSettled(
      evaluations.map((evaluation) =>
        patchEvaluation({
          workspace,
          name: evaluation.name,
          data: { experiment_ids: [...new Set([...evaluation.experiment_ids, groupId])] },
        })
      )
    );
    return results.filter((result) => result.status === 'rejected').length;
  };

  const refreshEvaluationBoards = () => {
    queryClient.invalidateQueries({ queryKey: getListEvaluationsQueryKey(workspace) });
  };

  // Shared finish for both paths once the target group's membership writes have settled.
  const finishAdds = (groupName: string, failed: number, createdVerb: string) => {
    if (failed === count) {
      // Nothing landed — keep the modal open so the user can retry.
      toast.error(`Failed to add ${countLabel} to "${groupName}".`);
      return;
    }
    refreshEvaluationBoards();
    if (failed > 0) {
      toast.warning(
        `${createdVerb} "${groupName}", but ${failed} of ${count} evaluations couldn't be added.`
      );
    } else {
      toast.success(`${createdVerb} "${groupName}" with ${countLabel}.`);
    }
    onSuccess?.();
    onClose();
  };

  const addToExistingGroup = async () => {
    const groupName = groupNameById.get(selectedGroupId);
    if (!groupName) return;
    finishAdds(groupName, await associateEvaluations(selectedGroupId), 'Added to');
  };

  const createGroupAndAdd: SubmitHandler<ExperimentCreateFormFields> = async (data) => {
    let created;
    try {
      created = await createExperiment({
        workspace,
        data: {
          name: data.name,
          description: data.description || undefined,
          default_sort: defaultSort,
        },
      });
    } catch (error) {
      // Creation failed (e.g. duplicate name) — surface inline on the name field where possible and
      // keep the modal open. No experiment was created, so there's nothing to add.
      const detail = error instanceof AxiosError ? error.response?.data?.detail : undefined;
      if (
        error instanceof AxiosError &&
        error.response?.status === 409 &&
        typeof detail === 'string'
      ) {
        setError('name', { message: detail });
        return;
      }
      const message =
        typeof detail === 'string'
          ? detail
          : error instanceof Error
            ? error.message
            : 'Unknown error';
      toast.error(`Failed to create experiment: ${message}`);
      return;
    }
    // Group now exists; adding evaluations is best-effort (a group with fewer evals is still valid).
    queryClient.invalidateQueries({ queryKey: getListExperimentsQueryKey(workspace) });
    finishAdds(created.name, await associateEvaluations(created.id), 'Created');
  };

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    // FormModal renders the <form>; it doesn't preventDefault. RHF's handleSubmit does it for us on the
    // create path; the add path has no RHF wrapper so we preventDefault ourselves.
    if (isCreating) {
      void handleSubmit(createGroupAndAdd)(e);
    } else {
      e.preventDefault();
      void addToExistingGroup();
    }
  };

  const submitButtonText = isCreating
    ? isCreatingGroup
      ? 'Creating…'
      : 'Create & add'
    : isAdding
      ? 'Adding…'
      : 'Add';

  return (
    <FormModal
      title="Add to experiment"
      instruction={`Add ${countLabel} to another experiment to compare across boards.`}
      submitButtonText={submitButtonText}
      disabled={busy}
      loading={busy}
      // Nothing chosen yet, or the create sub-form isn't valid (e.g. empty/invalid name).
      submitDisabled={!selectedGroupId || (isCreating && !isValid)}
      onSubmit={onSubmit}
      onClose={onClose}
      open={open}
      className="w-[560px]"
    >
      <Stack gap="density-2xl" className="w-full">
        <FormField slotLabel="Experiment">
          <SelectRoot
            value={selectedGroupId}
            onValueChange={setSelectedGroupId}
            disabled={busy || isLoading}
          >
            <SelectTrigger
              className="w-full"
              placeholder={isLoading ? 'Loading experiments…' : 'Select or create an experiment'}
              aria-label="Experiment"
              renderValue={(v) => {
                if (v === CREATE_NEW) return 'Create new experiment';
                return typeof v === 'string' && v ? (groupNameById.get(v) ?? undefined) : undefined;
              }}
            />
            <SelectContent className="w-(--radix-popper-anchor-width)">
              <SelectListbox>
                {groups.map((group) => (
                  <SelectItem key={group.id} value={group.id}>
                    {group.name}
                  </SelectItem>
                ))}
                {/* "Create new" sits at the bottom of the list, matching the workspace dropdown. */}
                <SelectItem value={CREATE_NEW}>
                  <Flex gap="density-sm" align="center">
                    <Plus className="size-4" />
                    Create new experiment
                  </Flex>
                </SelectItem>
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </FormField>

        {isCreating && (
          <>
            <FormField
              slotLabel="Name"
              slotError={errors.name?.message}
              status={errors.name && 'error'}
            >
              <TextInput
                autoFocus
                disabled={busy}
                status={errors.name && 'error'}
                {...register('name')}
                onChange={(e) =>
                  setValue('name', (e.target as HTMLInputElement).value.replace(/[\s-]+/g, '-'), {
                    shouldValidate: true,
                  })
                }
              />
            </FormField>
            <FormField
              slotLabel="Description (optional)"
              slotError={errors.description?.message}
              status={errors.description && 'error'}
            >
              <TextArea
                disabled={busy}
                status={errors.description && 'error'}
                {...register('description')}
              />
            </FormField>
            <DefaultSortControl value={defaultSort} onChange={setDefaultSort} disabled={busy} />
          </>
        )}
      </Stack>
    </FormModal>
  );
};
