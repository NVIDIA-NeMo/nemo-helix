// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  Banner,
  Button,
  Card,
  Divider,
  Flex,
  Stack,
  Stepper,
  Text,
} from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { buildOptimizationName } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationName';
import {
  modelPairs,
  ROUTING_STRATEGIES,
  switchyardFormSchema,
  type SwitchyardFormOutput,
  type SwitchyardFormValues,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import { ModelOrderField } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/ModelOrderField';
import {
  DEFAULT_THRESHOLD,
  RoutingStrategiesSection,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/RoutingStrategiesSection';
import { ChevronLeft } from 'lucide-react';
import { type FC, useState } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

export interface SwitchyardOptimizationFormProps {
  agentName?: string;
  /** Returns to the strategy picker; the target of the back button above the header. */
  onBack: () => void;
  /**
   * Starts the study from the validated answers. A rejection is shown beside the run button, so
   * its message should be one a user can act on. Left unset, the run button stays closed.
   */
  onSubmit?: (values: SwitchyardFormOutput) => Promise<void>;
}

/** Drops the `workspace/` prefix from a model in the current workspace; the full name is the tooltip. */
const withoutWorkspace = (model: string, workspace: string | undefined): string =>
  workspace && model.startsWith(`${workspace}/`) ? model.slice(workspace.length + 1) : model;

const plural = (count: number, noun: string): string =>
  `${count} ${count === 1 ? noun : noun.endsWith('y') ? `${noun.slice(0, -1)}ies` : `${noun}s`}`;

/**
 * Configure a Switchyard model-routing study for one agent.
 *
 * Laid out like {@link NewOptimizationForm} — in place of the studies table, with a summary
 * beside it — and submits the fields of the plugin's `SwitchyardOptimizeSpec`: the models to route
 * between (most capable first), the routing strategies to try, and each strategy's threshold.
 */
export const SwitchyardOptimizationForm: FC<SwitchyardOptimizationFormProps> = ({
  agentName,
  onBack,
  onSubmit,
}) => {
  const workspace = useWorkspaceFromPath();

  const form = useForm<SwitchyardFormValues, unknown, SwitchyardFormOutput>({
    resolver: zodResolver(switchyardFormSchema),
    mode: 'onChange',
    defaultValues: {
      name: agentName ? buildOptimizationName(agentName, 'routing') : '',
      models: [],
      routingStrategies: ['random_routing'],
      strong_probability: DEFAULT_THRESHOLD,
      confidence_threshold: DEFAULT_THRESHOLD,
      base_threshold: DEFAULT_THRESHOLD,
      judgeModel: '',
    },
  });
  const {
    control,
    watch,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = form;

  const name = watch('name');
  const models = watch('models');
  const routingStrategies = watch('routingStrategies');
  const pairs = modelPairs(models);
  const virtualModels = pairs.length * routingStrategies.length;

  const thresholdError = ROUTING_STRATEGIES.some(
    (option) =>
      routingStrategies.includes(option.id) && errors[option.threshold.field] !== undefined
  );

  const blockingReason = !agentName
    ? 'No agent selected.'
    : models.length < 2
      ? 'Add at least two models to route between.'
      : routingStrategies.length === 0
        ? 'Pick at least one routing strategy.'
        : errors.name
          ? 'Fix the name before running.'
          : thresholdError
            ? 'Fix the thresholds before running.'
            : !onSubmit
              ? 'Running a study from here is not available yet.'
              : undefined;

  const [submitError, setSubmitError] = useState<string | undefined>();

  const submit = handleSubmit(async (values) => {
    if (!onSubmit) return;
    setSubmitError(undefined);
    try {
      await onSubmit(values);
    } catch (error) {
      setSubmitError(getErrorMessage(error as Error, 'Could not start the optimization.'));
    }
  });

  const preview = toValidEntityName(name, '');

  return (
    <FormProvider {...form}>
      <Stack gap="density-xl" className="w-full">
        <Button kind="tertiary" className="w-fit px-0" onClick={onBack}>
          <ChevronLeft className="size-4" aria-hidden />
          Back
        </Button>

        <Stack gap="density-sm">
          <Text kind="title/sm">New optimization</Text>
          <Text kind="body/regular/sm" color="secondary">
            Model routing{agentName ? ` on ${agentName}` : ''}. Choose the models to route between
            and how each request should pick one.
          </Text>
        </Stack>

        <div className="grid grid-cols-1 items-start gap-8 xl:grid-cols-[minmax(0,1fr)_380px]">
          <Stack gap="density-2xl">
            <ControlledTextInput
              useControllerProps={{ control, name: 'name' }}
              label="Name"
              disabled={isSubmitting}
              formFieldProps={{
                slotHelp: !errors.name && preview && (
                  <>
                    Your optimization will be created as{' '}
                    <span className="text-primary">{preview}</span>
                  </>
                ),
              }}
            />

            <Stepper
              layout="vertical"
              aria-label="Routing optimization setup"
              activeStep={models.length < 2 ? 0 : routingStrategies.length === 0 ? 1 : 2}
              items={[
                {
                  slotHeading: 'Which models can it use?',
                  slotDescription:
                    'Most capable first. Every model is paired with each model below it, and each pair routes between its two models',
                  slotSuccessIndicator: 1,
                  slotContent: <ModelOrderField workspace={workspace} disabled={isSubmitting} />,
                },
                {
                  slotHeading: 'How should each pair route?',
                  slotDescription:
                    'Pick one or more strategies to compare. The defaults are a sensible start',
                  slotSuccessIndicator: 2,
                  slotContent: (
                    <RoutingStrategiesSection workspace={workspace} disabled={isSubmitting} />
                  ),
                },
              ]}
            />
          </Stack>

          <Card className="h-fit min-w-0 self-start">
            <Stack gap="density-md" className="w-full min-w-0">
              <Text kind="label/bold/xs" color="secondary">
                THIS RUN WILL BUILD
              </Text>
              <Flex gap="density-xl" wrap="wrap">
                <Stack gap="density-xxs">
                  <Text kind="body/bold/lg">{virtualModels || '—'}</Text>
                  <Text kind="body/regular/xs" color="secondary">
                    routed models
                  </Text>
                </Stack>
                <Stack gap="density-xxs">
                  <Text kind="body/bold/lg">{pairs.length || '—'}</Text>
                  <Text kind="body/regular/xs" color="secondary">
                    model pairs
                  </Text>
                </Stack>
              </Flex>

              {pairs.length > 0 && (
                <ol
                  aria-label="Model pairs"
                  className="divide-base bg-surface-raised border-base min-w-0 divide-y overflow-hidden rounded-lg border"
                >
                  {pairs.map(({ capable, efficient }) => (
                    <li key={`${capable}|${efficient}`} className="px-3 py-2">
                      <Stack gap="density-xxs" className="min-w-0">
                        {[
                          { model: capable, role: 'capable' },
                          { model: efficient, role: 'efficient' },
                        ].map(({ model, role }) => (
                          <Flex key={role} align="center" gap="density-sm" className="min-w-0">
                            <Text
                              kind="body/regular/xs"
                              className="min-w-0 flex-1 truncate"
                              title={model}
                            >
                              {withoutWorkspace(model, workspace)}
                            </Text>
                            <Text kind="body/regular/xs" color="secondary" className="shrink-0">
                              {role}
                            </Text>
                          </Flex>
                        ))}
                      </Stack>
                    </li>
                  ))}
                </ol>
              )}

              <Text kind="body/regular/xs" color="secondary">
                {plural(pairs.length, 'pair')} × {plural(routingStrategies.length, 'strategy')}.
                Each routed model splits traffic between one pair, the first as its capable model.
                Each gets a copy of the agent config pointed at it — the stored agent is never
                modified.
              </Text>

              <Divider />

              {blockingReason && (
                <Banner kind="inline" status="warning">
                  {blockingReason}
                </Banner>
              )}

              {submitError && (
                <Banner kind="inline" status="error">
                  {submitError}
                </Banner>
              )}

              <LoadingButton
                color="brand"
                className="w-full"
                loading={isSubmitting}
                disabled={!!blockingReason || isSubmitting}
                onClick={() => void submit()}
              >
                Run optimization
              </LoadingButton>
            </Stack>
          </Card>
        </div>
      </Stack>
    </FormProvider>
  );
};
