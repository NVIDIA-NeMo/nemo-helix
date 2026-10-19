// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import {
  getGetExperimentQueryKey,
  getListExperimentsQueryKey,
  useUpdateExperiment,
} from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack, TextInput } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import { trendVisibilityStorageKey } from '@studio/components/charts/ExperimentTrendChart/visibility';
import {
  EXPERIMENT_SETTINGS_NAMES,
  type ExperimentSettingsValues,
  experimentSettingsFrom,
  experimentSettingsPayload,
} from '@studio/components/evaluation/shared/experimentSettings';
import { ExperimentSettingsFields } from '@studio/components/evaluation/shared/ExperimentSettingsFields';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { AxiosError } from 'axios';
import { type FC, type FormEvent, useEffect, useMemo } from 'react';
import { useForm } from 'react-hook-form';

export interface ExperimentEditModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  group: ExperimentResponse;
}

export const ExperimentEditModal: FC<ExperimentEditModalProps> = ({
  open,
  onClose,
  workspace,
  group,
}) => {
  const toast = useToast();

  const { control, handleSubmit, reset } = useForm<ExperimentSettingsValues>({
    defaultValues: experimentSettingsFrom(group),
  });

  // Reset form state whenever the modal (re)opens or points at a different group.
  useEffect(() => {
    if (open) reset(experimentSettingsFrom(group));
  }, [open, group, reset]);

  // Offer the group's discovered evaluators as first-class sort fields (only fetched while open).
  const { data: experimentsPage } = useListEvaluations(
    workspace,
    { filter: { experiment_id: group.id }, page_size: 100 },
    { query: { enabled: open && !!group.id } }
  );
  const evaluatorOptions = useMemo(
    () =>
      [
        ...new Set(
          (experimentsPage?.data ?? []).flatMap((e) => Object.keys(e.aggregate_scores ?? {}))
        ),
      ].sort(),
    [experimentsPage]
  );

  // `resolveTrendVisible` already retires a stored choice whose stamped flag no longer matches, which
  // covers the flag moving anywhere else — API, CLI, another tab. It cannot cover a round trip: turn
  // the flag off and back on and the stamp matches again, so a viewer who had hidden the chart would
  // watch the owner's edit do nothing. An explicit edit here is unambiguous, so drop the choice
  // outright. Deleting through the hook notifies the open page, so the chart follows without a reload.
  const [, , clearTrendChoice] = useLocalStorage(trendVisibilityStorageKey(group.id));

  const { mutateAsync: updateExperiment, isPending } = useUpdateExperiment({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetExperimentQueryKey(workspace, group.name),
        });
        queryClient.invalidateQueries({ queryKey: getListExperimentsQueryKey(workspace) });
      },
    },
  });

  const submit = async (values: ExperimentSettingsValues) => {
    try {
      await updateExperiment({
        workspace,
        name: group.name,
        data: {
          // Name is immutable for a group; send it unchanged so the update isn't treated as a rename.
          name: group.name,
          // The endpoint replaces the whole group, so resend the producer-owned fields this
          // form doesn't edit; omitting them clears the group's summary and insight link.
          insight_id: group.insight_id,
          summary: group.summary,
          metadata: group.metadata,
          ...experimentSettingsPayload(values),
        },
      });
      if (values.showEvaluationsOverTime !== (group.show_evaluations_over_time ?? false)) {
        clearTrendChoice();
      }
      onClose();
    } catch (error) {
      const detail = error instanceof AxiosError ? error.response?.data?.detail : undefined;
      const message =
        typeof detail === 'string'
          ? detail
          : error instanceof Error
            ? error.message
            : 'Unknown error';
      toast.error(`Failed to update experiment: ${message}`);
    }
  };

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault(); // FormModal doesn't preventDefault; without RHF's handler we must.
    void handleSubmit(submit)(e);
  };

  return (
    <FormModal
      title="Edit experiment"
      instruction="Update the experiment's description, default sort order, and presentation settings."
      submitButtonText={isPending ? 'Saving…' : 'Save'}
      disabled={isPending}
      loading={isPending}
      onSubmit={onSubmit}
      onClose={onClose}
      open={open}
      className="w-[800px] min-h-[400px]"
    >
      <Stack gap="density-2xl" className="w-full">
        <FormField slotLabel="Name">
          <TextInput value={group.name} disabled />
        </FormField>
        <ExperimentSettingsFields
          control={control}
          names={EXPERIMENT_SETTINGS_NAMES}
          disabled={isPending}
          evaluatorOptions={evaluatorOptions}
        />
      </Stack>
    </FormModal>
  );
};
