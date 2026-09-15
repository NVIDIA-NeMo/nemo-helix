// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Banner, Grid, Spinner, Text } from '@nvidia/foundations-react-core';
import {
  deleteCustomizationJobTemplate,
  getCustomizationJobTemplatesQueryKey,
  listCustomizationJobTemplates,
} from '@studio/api/customization-job-templates/customizationJobTemplates';
import type { CustomizationJobTemplatesPage } from '@studio/api/customization-job-templates/types';
import { SavedTemplateCard } from '@studio/components/CreateCustomizationStart/SavedTemplateCard';
import type { SavedTemplateGridProps } from '@studio/components/CreateCustomizationStart/types';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, type FC } from 'react';

export const SavedTemplateGrid: FC<SavedTemplateGridProps> = ({
  workspace,
  selectedTemplateId,
  onSelectTemplate,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [deletingName, setDeletingName] = useState<string | null>(null);

  const queryKey = getCustomizationJobTemplatesQueryKey(workspace);
  const {
    data: page,
    isPending,
    error,
  } = useQuery({
    queryKey,
    queryFn: ({ signal }) => listCustomizationJobTemplates(workspace, undefined, signal),
  });

  const { mutate: remove } = useMutation({
    mutationFn: (name: string) => deleteCustomizationJobTemplate(workspace, name),
    onMutate: (name: string) => setDeletingName(name),
    onSuccess: (_result, name) => {
      // Drop it from the cache rather than relying on the refetch: the start page reads
      // this same query to decide whether Continue is armed, and a refetch that fails
      // leaves the stale page in place — the grid would show an error while Continue
      // stayed enabled for a template that is gone.
      queryClient.setQueryData(queryKey, (previous?: CustomizationJobTemplatesPage) =>
        previous
          ? { ...previous, data: previous.data.filter((template) => template.name !== name) }
          : previous
      );
      void queryClient.invalidateQueries({ queryKey });
      toast.success(`Deleted template '${name}'`);
    },
    onError: (err: Error) => toast.error(getErrorMessage(err, 'Failed to delete template')),
    onSettled: () => setDeletingName(null),
  });

  if (isPending) return <Spinner size="small" aria-label="Loading your saved templates" />;

  if (error) {
    return (
      <Banner kind="inline" status="error">
        {getErrorMessage(error, 'Failed to load your saved templates.')}
      </Banner>
    );
  }

  const templates = page?.data ?? [];

  if (templates.length === 0) {
    return (
      <Text kind="body/regular/sm" className="text-secondary">
        You haven&apos;t saved any templates yet. Configure a fine-tuning job and choose “Save as
        Template” to reuse it later.
      </Text>
    );
  }

  return (
    <Grid cols={{ base: 1, md: 2, lg: 3 }} gap="density-md">
      {templates.map((template) => (
        <SavedTemplateCard
          key={template.id}
          template={template}
          selected={selectedTemplateId === template.name}
          onSelect={() => onSelectTemplate(template.name)}
          disabled={deletingName === template.name}
          isDeleting={deletingName === template.name}
          onDelete={() => remove(template.name)}
        />
      ))}
    </Grid>
  );
};
