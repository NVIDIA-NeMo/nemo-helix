// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { Button, FormField, Stack, Stepper, Text, TextInput } from '@nvidia/foundations-react-core';
import { AdvancedAccordion } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/AdvancedAccordion';
import { BudgetSection } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/BudgetSection';
import { EvaluationSection } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/EvaluationSection';
import {
  optimizationFormSchema,
  type OptimizationFormOutput,
  type OptimizationFormValues,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { IntentSection } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/IntentSection';
import { buildOptimizationName } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationName';
import { optimizationTargets } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import { RunSummaryPanel } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/RunSummaryPanel';
import {
  budgetById,
  intentById,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { ChevronLeft } from 'lucide-react';
import { type FC, useEffect, useMemo, useRef, useState } from 'react';
import { FormProvider, useController, useForm } from 'react-hook-form';

const DEFAULT_INTENT = 'accuracy' as const;

export interface NewOptimizationFormProps {
  agentName?: string;
  evals: AgentEvaluationRow[];
  isEvalsPending: boolean;
  /** Returns to the studies table; also the target of the breadcrumb above the header. */
  onBack: () => void;
  /**
   * Starts the study from the validated answers.
   *
   * Left unset for now, which is what holds the run button closed: generating the optimize config,
   * staging the evaluation's rows, and creating the job are the submit path, and they land
   * separately. Wiring this up is the whole of that change at this call site.
   */
  onSubmit?: (values: OptimizationFormOutput) => Promise<void>;
}

/**
 * Configure a numeric HPO study for one agent.
 *
 * Renders in place of the studies table rather than in a modal: the form carries a run summary
 * beside it, which does not survive a dialog's width, and its own breadcrumb is what returns to
 * the list.
 *
 * The user answers three questions — what to tune for, what to score against, how many trials.
 * Nothing here asks for a config path or a fileset; those are derived on submit.
 */
export const NewOptimizationForm: FC<NewOptimizationFormProps> = ({
  agentName,
  evals,
  isEvalsPending,
  onBack,
  onSubmit,
}) => {
  const targets = useMemo(() => optimizationTargets(evals), [evals]);

  const methods = useForm<OptimizationFormValues, unknown, OptimizationFormOutput>({
    resolver: zodResolver(optimizationFormSchema),
    mode: 'onBlur',
    defaultValues: {
      name: agentName ? buildOptimizationName(agentName, DEFAULT_INTENT) : '',
      intent: DEFAULT_INTENT,
      budget: 'standard',
      experimentId: '',
      judgeModel: '',
      searchSpace: intentById(DEFAULT_INTENT).parameters,
    },
  });

  const {
    control,
    watch,
    setValue,
    handleSubmit,
    formState: { errors, touchedFields, isSubmitting },
  } = methods;

  const {
    field: { value: nameValue, onChange: onNameChange, onBlur: onNameBlur },
  } = useController({ control, name: 'name' });
  const [isNameGenerated, setIsNameGenerated] = useState(true);
  const generatedFor = useRef(`${agentName ?? ''}:${DEFAULT_INTENT}`);

  const preview = toValidEntityName(nameValue, '');

  const intentId = watch('intent');
  const intent = intentById(intentId);
  const budget = budgetById(watch('budget'));
  const searchSpace = watch('searchSpace');
  const experimentId = watch('experimentId');
  const judgeModel = watch('judgeModel');

  // Regenerate when the intent changes, since the intent is part of the name — and when the agent
  // finally resolves, which is what leaves the initial name empty.
  useEffect(() => {
    if (!isNameGenerated || !agentName) return;
    const key = `${agentName}:${intentId}`;
    if (generatedFor.current === key) return;
    generatedFor.current = key;
    setValue('name', buildOptimizationName(agentName, intentId), { shouldValidate: true });
  }, [agentName, intentId, isNameGenerated, setValue]);

  // The most recent evaluation is the one a user almost always means, so seed it — but only once
  // the list has actually arrived, which is after the form's defaults are set.
  useEffect(() => {
    if (!experimentId && targets[0]) {
      setValue('experimentId', targets[0].experimentId, { shouldValidate: true });
    }
  }, [experimentId, targets, setValue]);

  const target = targets.find((candidate) => candidate.experimentId === experimentId);

  const nameError = touchedFields.name ? errors.name?.message : undefined;

  const blockingReason = !agentName
    ? 'No agent selected.'
    : !experimentId
      ? 'Pick an evaluation to score trials against.'
      : !judgeModel
        ? 'Pick a judge model to score trials with.'
        : nameError
          ? 'Fix the name before running.'
          : errors.searchSpace
            ? 'Fix the search space before running.'
            : !onSubmit
              ? 'Running a study from here is not available yet.'
              : undefined;

  const submit = handleSubmit(async (values) => {
    if (!onSubmit) return;
    await onSubmit(values);
  });

  return (
    <FormProvider {...methods}>
      <Stack gap="density-xl" className="w-full">
        <Button kind="tertiary" className="w-fit px-0" onClick={onBack}>
          <ChevronLeft className="size-4" aria-hidden />
          Optimizations
        </Button>

        <Stack gap="density-sm">
          <Text kind="title/sm">New optimization</Text>
          <Text kind="body/regular/sm" color="secondary">
            Parameter sweep{agentName ? ` on ${agentName}` : ''}. Choose what to tune for, what to
            score it against, and how hard to look.
          </Text>
        </Stack>

        <div className="grid grid-cols-1 items-start gap-8 xl:grid-cols-[minmax(0,1fr)_380px]">
          <Stack gap="density-2xl">
            <FormField
              slotLabel="Name"
              slotError={nameError}
              slotHelp={
                nameError
                  ? undefined
                  : isNameGenerated
                    ? 'Named for you from the agent and what you are tuning for. Edit it if you want something else.'
                    : preview && (
                        <>
                          Your optimization will be created as{' '}
                          <span className="text-primary">{preview}</span>
                        </>
                      )
              }
              status={nameError ? 'error' : undefined}
            >
              <TextInput
                value={nameValue}
                disabled={isSubmitting}
                status={nameError ? 'error' : undefined}
                onChange={(event) => {
                  setIsNameGenerated(false);
                  onNameChange(event.currentTarget.value);
                }}
                onBlur={onNameBlur}
              />
            </FormField>

            <Stepper
              layout="vertical"
              aria-label="Optimization setup"
              activeStep={experimentId && judgeModel ? 3 : 1}
              items={[
                {
                  slotHeading: 'What are you tuning for?',
                  slotDescription: 'Pick one — it sets the objective and the parameters we sweep',
                  slotSuccessIndicator: 1,
                  slotContent: <IntentSection />,
                },
                {
                  slotHeading: 'Test it against',
                  slotDescription: 'Its latest evaluation supplies the dataset and metrics',
                  slotSuccessIndicator: 2,
                  slotContent: (
                    <EvaluationSection
                      targets={targets}
                      selected={target}
                      isLoading={isEvalsPending}
                    />
                  ),
                },
                {
                  slotHeading: 'How hard should we look?',
                  slotDescription: 'More trials, better odds of a win',
                  slotSuccessIndicator: 3,
                  slotContent: <BudgetSection />,
                },
              ]}
            />

            <AdvancedAccordion searchSpace={searchSpace} />
          </Stack>

          <RunSummaryPanel
            intent={intent}
            budget={budget}
            searchSpace={searchSpace}
            target={target}
            blockingReason={blockingReason}
            isSubmitting={isSubmitting}
            onRun={() => void submit()}
          />
        </div>
      </Stack>
    </FormProvider>
  );
};
