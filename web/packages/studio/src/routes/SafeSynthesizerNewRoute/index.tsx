// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { logger } from '@nemo/common/src/utils/logger';
import { useSafeSynthesizerCreateJob } from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { Banner, Button, Divider, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { SAFE_SYNTHESIZER_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { AdvancedParametersAccordion } from '@studio/routes/SafeSynthesizerNewRoute/components/AdvancedParametersAccordion';
import { Generation } from '@studio/routes/SafeSynthesizerNewRoute/components/Generation';
import { JobName } from '@studio/routes/SafeSynthesizerNewRoute/components/JobName';
import { PrivacyProtection } from '@studio/routes/SafeSynthesizerNewRoute/components/PrivacyProtection';
import { TrainingData } from '@studio/routes/SafeSynthesizerNewRoute/components/TrainingData';
import {
  safeSynthesizerJobRequestSchema,
  getSafeSynthesizerFormDefaults,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { getGenerateJobRoute, getSafeSynthesizerRoute } from '@studio/routes/utils';
import { FC, useState } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

export const SafeSynthesizerNewRoute: FC | null = SAFE_SYNTHESIZER_ENABLED
  ? () => {
      const workspace = useWorkspaceFromPath();
      const navigate = useNavigate();

      useBreadcrumbs({
        items: [
          {
            slotLabel: 'Safe Synthesizer',
            href: getSafeSynthesizerRoute(workspace),
          },
          {
            slotLabel: 'New Job',
          },
        ],
      });

      const [errorMessage, setErrorMessage] = useState<string | null>(null);

      const form = useForm({
        mode: 'onChange',
        resolver: zodResolver(safeSynthesizerJobRequestSchema),
        defaultValues: getSafeSynthesizerFormDefaults(),
        disabled: false,
      });
      const { getValues, handleSubmit } = form;

      const createJobMutation = useSafeSynthesizerCreateJob({
        mutation: {
          onSuccess: (data) => {
            setErrorMessage(null);
            // Navigate to the job details or list page
            if (data.name) {
              navigate(getGenerateJobRoute(workspace, data.name));
            } else {
              navigate(getSafeSynthesizerRoute(workspace));
            }
          },
          onError: (error) => {
            logger.error('Failed to create job', error);
            setErrorMessage(
              getErrorMessage(error, 'Failed to create job. Please check your input and try again.')
            );
          },
        },
      });

      const handleCancel = () => {
        navigate(getSafeSynthesizerRoute(workspace));
      };

      const submitForm = () => {
        setErrorMessage(null); // Clear any existing errors
        const formData = getValues();

        // Remove empty fields for group_training_examples_by and order_training_examples_by
        if (formData.spec.config.data) {
          if (!formData.spec.config.data.group_training_examples_by?.trim()) {
            delete formData.spec.config.data.group_training_examples_by;
          }
          if (!formData.spec.config.data.order_training_examples_by?.trim()) {
            delete formData.spec.config.data.order_training_examples_by;
          }
        }

        createJobMutation.mutate({
          workspace: workspace,
          data: formData,
        });
      };

      const handleSubmitError = (errors: unknown) => {
        logger.error(`Form validation errors: ${String(errors)}`);
        setErrorMessage('Please fix the form errors before submitting.');
      };

      return (
        <FormProvider {...form}>
          <form
            className="w-full"
            onSubmit={handleSubmit(submitForm, handleSubmitError)}
            noValidate
          >
            <Stack className="overflow-auto" gap="density-2xl" padding="density-2xl">
              {errorMessage && (
                <Banner kind="inline" status="error">
                  {errorMessage}
                </Banner>
              )}
              <Flex align="center" justify="center" className="w-full">
                <Panel
                  className="max-w-[600px] h-full overflow-auto"
                  elevation="high"
                  density="standard"
                  slotHeading="Generate Private Synthetic Data"
                  slotFooter={
                    <Flex gap="density-md" justify="end">
                      <Button
                        kind="tertiary"
                        onClick={handleCancel}
                        type="button"
                        disabled={createJobMutation.isPending}
                      >
                        Cancel
                      </Button>
                      <Button
                        kind="primary"
                        color="brand"
                        type="submit"
                        disabled={createJobMutation.isPending}
                      >
                        Continue
                      </Button>
                    </Flex>
                  }
                >
                  <Stack gap="density-2xl">
                    <Stack gap="density-2xl">
                      <Text kind="body/regular/md">
                        NVIDIA NeMo Safe Synthesizer enables you to create private versions of
                        sensitive tabular datasets. The resulting data is entirely synthetic, with
                        no one-to-one mapping to your original records. NeMo Safe Synthesizer is
                        purpose-built for privacy compliance and data protection while preserving
                        data utility for downstream AI tasks.
                      </Text>
                      <JobName />
                    </Stack>
                    <Divider orientation="horizontal" width="small" />
                    <Stack gap="density-2xl">
                      <Text kind="label/bold/lg">Training Data</Text>
                      <Text kind="body/regular/md">
                        Safe Synthesizer learns the patterns and correlations in your input dataset
                        to produce synthetic data with similar properties. NeMo Safe Synthesizer
                        supports numeric, categorical, text, and event-driven fields in tabular
                        data. Supported data types are JSONL, CSV, and Parquet.
                      </Text>
                      <TrainingData workspace={workspace} />
                    </Stack>
                    <Divider orientation="horizontal" width="small" />
                    <Stack gap="density-2xl">
                      <Text kind="label/bold/lg">Generation</Text>
                      <Text kind="body/regular/md">
                        Generate up to 130,000 synthetic records generated from your original
                        training data.
                      </Text>
                      <Generation />
                    </Stack>
                    <Divider orientation="horizontal" width="small" />
                    <Stack gap="density-2xl">
                      <Text kind="label/bold/lg">Privacy Protection</Text>
                      <Text kind="body/regular/md">
                        In addition to the inherent privacy of synthetic data, supplemental
                        protection can be added through these mechanisms to prevent adversarial
                        attacks and better meet your data sharing needs.
                      </Text>
                      <PrivacyProtection />
                    </Stack>
                    <Divider orientation="horizontal" width="small" />
                    <Stack gap="density-2xl">
                      <AdvancedParametersAccordion />
                    </Stack>
                  </Stack>
                </Panel>
              </Flex>
            </Stack>
          </form>
        </FormProvider>
      );
    }
  : null;
