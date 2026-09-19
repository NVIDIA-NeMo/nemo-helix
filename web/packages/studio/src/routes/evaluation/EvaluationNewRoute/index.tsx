// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { StepSection } from '@nemo/common/src/components/StepSection';
import {
  Button,
  Divider,
  Flex,
  PageHeader,
  Panel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { DatasetPanel } from '@studio/routes/evaluation/EvaluationNewRoute/DatasetPanel';
import { LiveTestPanel } from '@studio/routes/evaluation/EvaluationNewRoute/LiveTestPanel';
import { MetricPanel } from '@studio/routes/evaluation/EvaluationNewRoute/MetricPanel';
import {
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
  evaluationSchema,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useCreateEvaluation } from '@studio/routes/evaluation/EvaluationNewRoute/useCreateEvaluation';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { useWizardProgress } from '@studio/routes/evaluation/EvaluationNewRoute/useWizardProgress';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { FC } from 'react';
import { FormProvider, useForm, useFormContext } from 'react-hook-form';

/** The form body lives inside FormProvider so it can read the template bindings
 *  and wizard progress, both of which derive from form state. */
const EvaluationForm: FC = () => {
  const form = useFormContext<EvaluationFormValues>();
  const bindings = useDatasetBindings();
  const progress = useWizardProgress();
  const { createEvaluation, isPending } = useCreateEvaluation();

  /** The resolver decides validity; handleSubmit only runs on success. RHF's
   *  defaults give the UX the guidelines ask for: nothing red until submit, then
   *  each error retires as its field is fixed. */
  const handleSubmit = form.handleSubmit((values) => {
    void createEvaluation(values, bindings);
  });

  return (
    <form onSubmit={handleSubmit} noValidate>
      <Stack gap="density-xl">
        <Panel elevation="high" className="w-full">
          <Stack gap="density-2xl" padding="density-2xl">
            {/* Explicit bases rather than equal thirds: the middle column carries
                the metric cards and needs the room. Default shrink absorbs the
                gaps, so 30/40/30 does not overflow. The rules are borders, not
                Divider elements, which would take space of their own and skew the
                ratio. Each column pins its content to the top so expanding the row
                preview does not re-centre its neighbours. */}
            <Flex align="stretch" gap="density-xl" className="w-full">
              <Stack justify="start" gap="density-2xl" className="min-w-0 basis-[30%]">
                <StepSection step={1} title="Model to Evaluate" status={progress.model}>
                  <JudgeModelSelect<EvaluationFormValues>
                    formFieldName="model"
                    slotLabel="Model to Evaluate"
                    placeholder="Select a model"
                  />
                </StepSection>
                <StepSection
                  step={2}
                  title="Dataset"
                  status={progress.dataset}
                  lockedHint="Choose a model above to continue."
                >
                  <DatasetPanel />
                </StepSection>
              </Stack>
              <Stack
                justify="start"
                className="min-w-0 basis-[40%] border-l border-base pl-density-xl"
              >
                <StepSection
                  step={3}
                  title="Evaluation Metrics"
                  status={progress.metrics}
                  lockedHint="Choose a dataset file and map its Input and Ground Truth fields to continue."
                >
                  <MetricPanel />
                </StepSection>
              </Stack>
              <Stack
                justify="start"
                className="min-w-0 basis-[30%] border-l border-base pl-density-xl"
              >
                <Stack justify="start" gap="density-lg" className="min-w-0">
                  <Stack gap="density-sm">
                    <Flex align="center" gap="density-sm">
                      <Text kind="label/bold/xl">Live Test</Text>
                      <Text kind="body/regular/md" className="text-secondary">
                        Optional
                      </Text>
                    </Flex>
                    <Text kind="body/regular/md" className="text-secondary">
                      Test one single row from your dataset with this configuration to validate
                      metric scores and model performance.
                    </Text>
                  </Stack>
                  <LiveTestPanel />
                </Stack>
              </Stack>
            </Flex>
            <Divider />
            <Flex justify="end">
              <Button color="brand" type="submit" disabled={isPending}>
                Create Evaluation
              </Button>
            </Flex>
          </Stack>
        </Panel>
      </Stack>
    </form>
  );
};

export const EvaluationNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();

  useBreadcrumbs({
    items: [
      { slotLabel: 'Evaluations', href: getEvaluationResultsRoute(workspace) },
      { slotLabel: 'New Model Evaluation' },
    ],
  });

  const form = useForm<EvaluationFormValues>({
    defaultValues: EVALUATION_FORM_DEFAULTS,
    resolver: zodResolver(evaluationSchema),
  });

  return (
    <AccessibleTitle title="New Model Evaluation">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader className="p-0" slotHeading="New Model Evaluation" />
        <FormProvider {...form}>
          <EvaluationForm />
        </FormProvider>
      </Stack>
    </AccessibleTitle>
  );
};
