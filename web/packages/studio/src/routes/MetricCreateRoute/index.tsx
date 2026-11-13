// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import {
  getEvalsListMetricsQueryKey,
  useEvalsCreateMetric,
} from '@nemo/sdk/generated/evals/evals-plugin-metrics-routes';
import {
  Banner,
  Button,
  Flex,
  FormField,
  PageHeader,
  Panel,
  RadioGroup,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { submitErrorMessage } from '@studio/components/evaluation/EvalWizard/entityNameField';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { buildMetric } from '@studio/routes/MetricCreateRoute/buildMetric';
import { MetricConfigFields } from '@studio/routes/MetricCreateRoute/MetricConfigFields';
import { TryMetricPanel } from '@studio/routes/MetricCreateRoute/TryMetricPanel';
import {
  METRIC_FORM_DEFAULTS,
  METRIC_TYPE_GROUPS,
  type MetricFormValues,
  metricSchema,
} from '@studio/routes/MetricCreateRoute/types';
import { getMetricDetailRoute, getMetricsRoute, getNewMetricRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';
import { FormProvider, useController, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

const MetricTypePicker: FC<{ form: ReturnType<typeof useForm<MetricFormValues>> }> = ({ form }) => {
  const { field } = useController({ control: form.control, name: 'type' });
  return (
    <Stack gap="density-lg">
      {METRIC_TYPE_GROUPS.map((group) => (
        <FormField key={group.heading} slotLabel={group.heading}>
          <RadioGroup
            kind="tile"
            orientation="vertical"
            name={`metric-type-${group.heading}`}
            value={group.options.some((option) => option.type === field.value) ? field.value : ''}
            onValueChange={(value) => field.onChange(value)}
            items={group.options.map((option) => ({
              value: option.type,
              children: (
                <Stack gap="density-xs">
                  <Text kind="label/bold/md">{option.label}</Text>
                  <Text kind="body/regular/sm" className="text-secondary">
                    {option.description}
                  </Text>
                </Stack>
              ),
            }))}
          />
        </FormField>
      ))}
    </Stack>
  );
};

export const MetricCreateRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { mutateAsync: createMetric, isPending, error } = useEvalsCreateMetric();
  const form = useForm<MetricFormValues>({
    defaultValues: METRIC_FORM_DEFAULTS,
    resolver: zodResolver(metricSchema),
  });

  useBreadcrumbs({
    items: [
      { href: getMetricsRoute(workspace), slotLabel: 'Metrics' },
      { href: getNewMetricRoute(workspace), slotLabel: 'Create Metric' },
    ],
  });

  const submit = form.handleSubmit(async (values) => {
    await createMetric({ workspace, name: values.name, data: buildMetric(values, workspace) });
    await queryClient.invalidateQueries({ queryKey: getEvalsListMetricsQueryKey(workspace) });
    navigate(getMetricDetailRoute(workspace, values.name));
  });

  return (
    <AccessibleTitle title="Create Metric">
      <FormProvider {...form}>
        <form noValidate aria-label="Create Metric" onSubmit={(event) => void submit(event)}>
          <Stack gap="density-2xl" padding="density-2xl">
            <PageHeader
              className="p-0"
              slotHeading={
                <>
                  Create Metric <FeatureFlagBadge flag="metricsEnabled" />
                </>
              }
              slotDescription="A reusable scorer. Any evaluation in this workspace can reference it by name."
            />
            {error ? (
              <Banner kind="inline" status="error">
                {submitErrorMessage(error)}
              </Banner>
            ) : null}
            <Flex gap="density-2xl" align="start" className="flex-col lg:flex-row">
              <Panel
                elevation="mid"
                className="w-full min-w-0 lg:flex-[3]"
                slotFooter={
                  <Flex justify="end" gap="density-md" className="w-full">
                    <Button kind="tertiary" onClick={() => navigate(getMetricsRoute(workspace))}>
                      Cancel
                    </Button>
                    <LoadingButton color="brand" type="submit" loading={isPending}>
                      Create
                    </LoadingButton>
                  </Flex>
                }
              >
                <Stack gap="density-2xl">
                  <Stack gap="density-lg">
                    <ControlledTextInput
                      useControllerProps={{ name: 'name', control: form.control }}
                      formFieldProps={{ slotLabel: 'Name', required: true }}
                      placeholder="answer-correctness"
                    />
                    <ControlledTextInput
                      useControllerProps={{ name: 'description', control: form.control }}
                      formFieldProps={{ slotLabel: 'Description' }}
                      placeholder="What this metric measures"
                    />
                  </Stack>
                  <Stack gap="density-md">
                    <Text kind="title/sm">How should responses be scored?</Text>
                    <MetricTypePicker form={form} />
                  </Stack>
                  <Stack gap="density-md">
                    <Text kind="title/sm">Configure</Text>
                    <MetricConfigFields />
                  </Stack>
                </Stack>
              </Panel>
              <aside className="w-full lg:sticky lg:top-0 lg:flex-[2]">
                <TryMetricPanel workspace={workspace} />
              </aside>
            </Flex>
          </Stack>
        </form>
      </FormProvider>
    </AccessibleTitle>
  );
};
