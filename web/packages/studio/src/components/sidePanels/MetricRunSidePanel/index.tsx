// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useModelEntity } from '@nemo/common/src/api/models/useModelEntity';
import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  ChatCompletionInput,
  defaultChatCompletionMessageRow,
} from '@nemo/common/src/components/ChatCompletionInput';
import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import type { VariableDef } from '@nemo/common/src/components/form/VariableTextArea';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { useEvalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvaluateJobRequest, MetricInline, Model } from '@nemo/sdk/generated/evals/schema';
import {
  Button,
  Flex,
  FormField,
  SegmentedControl,
  SidePanel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { MetricItemWithId } from '@studio/components/dataViews/EvaluationMetricsDataView/types';
import { EvalCard } from '@studio/components/evaluation/EvalCard';
import { FileValidationPanel } from '@studio/components/sidePanels/MetricRunSidePanel/FileValidationPanel';
import { MetricRunExecutionParametersSection } from '@studio/components/sidePanels/MetricRunSidePanel/MetricRunExecutionParametersSection';
import type {
  MetricRunSidePanelFormData,
  MetricRunSidePanelProps,
} from '@studio/components/sidePanels/MetricRunSidePanel/types';
import {
  buildMetricRunChatPromptTemplate,
  buildMetricRunOnlineJobParams,
  getMetricRunValidationPromptTemplate,
  getModelSelectionFromSearchParam,
} from '@studio/components/sidePanels/MetricRunSidePanel/utils';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { DEFAULT_INFERENCE_PARAMS_FORM_VALUES } from '@studio/hooks/evaluation/useCreateConfigurationForm';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { getEvaluationResultDetailsRoute } from '@studio/routes/utils';
import { buildModelPayload } from '@studio/util/evaluations';
import { Plus } from 'lucide-react';
import { type FC, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Controller,
  FormProvider,
  useFieldArray,
  useForm,
  useWatch,
  type UseFormReturn,
} from 'react-hook-form';
import { useNavigate, useSearchParams } from 'react-router';

interface MetricRunFileValidationState {
  dataset: string | null;
  jobType: MetricRunSidePanelFormData['jobType'];
  promptTemplate: string;
}

const getPromptTemplateFromMetric = (metric: MetricItemWithId | null): unknown =>
  metric && 'prompt_template' in metric ? metric.prompt_template : undefined;

const getMetricRunFileValidationState = (
  form: UseFormReturn<MetricRunSidePanelFormData>,
  selectedMetric: MetricItemWithId | null
): MetricRunFileValidationState => {
  const { dataset, jobType, promptMessages } = form.getValues();

  return {
    dataset,
    jobType,
    promptTemplate: getMetricRunValidationPromptTemplate({
      metricPromptTemplate: getPromptTemplateFromMetric(selectedMetric),
      promptMessages: promptMessages ?? [],
    }),
  };
};

const getNextMetricRunFileValidationState = (
  previousState: MetricRunFileValidationState,
  nextState: MetricRunFileValidationState
): MetricRunFileValidationState =>
  previousState.dataset === nextState.dataset &&
  previousState.jobType === nextState.jobType &&
  previousState.promptTemplate === nextState.promptTemplate
    ? previousState
    : nextState;

interface MetricRunFileValidationPanelProps {
  form: UseFormReturn<MetricRunSidePanelFormData>;
  selectedMetric: MetricItemWithId | null;
  workspace: string;
  onVariablesChange: (variables: VariableDef[]) => void;
}

const MetricRunFileValidationPanel: FC<MetricRunFileValidationPanelProps> = ({
  form,
  selectedMetric,
  workspace,
  onVariablesChange,
}) => {
  const [validationState, setValidationState] = useState(() =>
    getMetricRunFileValidationState(form, selectedMetric)
  );

  useEffect(() => {
    setValidationState((previousState) =>
      getNextMetricRunFileValidationState(
        previousState,
        getMetricRunFileValidationState(form, selectedMetric)
      )
    );

    const subscription = form.watch((_value, { name }) => {
      if (
        name === undefined ||
        name === 'dataset' ||
        name === 'jobType' ||
        name.startsWith('promptMessages')
      ) {
        setValidationState((previousState) =>
          getNextMetricRunFileValidationState(
            previousState,
            getMetricRunFileValidationState(form, selectedMetric)
          )
        );
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [form, selectedMetric]);

  return (
    <FileValidationPanel
      dataset={validationState.dataset}
      jobType={validationState.jobType}
      promptTemplate={validationState.promptTemplate}
      workspace={workspace}
      onVariablesChange={onVariablesChange}
    />
  );
};

export const MetricRunSidePanel: FC<MetricRunSidePanelProps> = ({
  metric,
  open,
  onOpenChange,
  workspace,
  attributes,
}) => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const toast = useToast();
  const [datasetVariables, setDatasetVariables] = useState<VariableDef[]>([]);
  const [selectedJobType, setSelectedJobType] =
    useState<MetricRunSidePanelFormData['jobType']>('online');
  const [modelSelectOpen, setModelSelectOpen] = useState(false);
  const modelSearch = useModelSearch({ workspace, enabled: open && modelSelectOpen });
  const { mutateAsync: createEvaluateJob, isPending } = useEvalsCreateEvaluateJob();

  const modelSearchParam = searchParams.get(QUERY_PARAMETERS.model);
  const modelSelectionFromSearchParam = useMemo(
    () => getModelSelectionFromSearchParam(modelSearchParam, workspace),
    [modelSearchParam, workspace]
  );

  const defaultFormValues = useMemo<MetricRunSidePanelFormData>(
    () => ({
      jobType: 'online',
      dataset: null,
      model: modelSelectionFromSearchParam,
      inferenceParams: { ...DEFAULT_INFERENCE_PARAMS_FORM_VALUES },
      promptMessages: [defaultChatCompletionMessageRow()],
      metricName: null,
      ignore_request_failure: false,
    }),
    [modelSelectionFromSearchParam]
  );

  const form = useForm<MetricRunSidePanelFormData>({
    defaultValues: defaultFormValues,
  });
  const {
    fields: promptMessageFields,
    append: appendPromptMessage,
    insert: insertPromptMessage,
    move: movePromptMessage,
    remove: removePromptMessage,
  } = useFieldArray({ control: form.control, name: 'promptMessages' });

  const selectedModel = useWatch({ control: form.control, name: 'model' });
  const fetchedModel = useModelEntity(selectedModel?.model, {
    enabled: open && !!selectedModel?.adapter && !selectedModel?.entity,
  });
  const selectedModelEntity = selectedModel?.entity ?? fetchedModel;

  useEffect(() => {
    if (modelSearch.error) {
      form.setError('model', { message: modelSearch.error.message });
    }
  }, [form, modelSearch.error]);

  useEffect(() => {
    if (open) {
      form.reset(defaultFormValues);
      setSelectedJobType(defaultFormValues.jobType);
      setDatasetVariables([]);
    }
  }, [defaultFormValues, open, form]);

  const handleDatasetVariablesChange = useCallback((variables: VariableDef[]) => {
    setDatasetVariables((previousVariables) => {
      const unchanged =
        previousVariables.length === variables.length &&
        previousVariables.every(
          (previousVariable, index) =>
            previousVariable.name === variables[index]?.name &&
            previousVariable.description === variables[index]?.description
        );
      return unchanged ? previousVariables : variables;
    });
  }, []);

  useEffect(() => {
    const subscription = form.watch((_value, { name }) => {
      if (
        name?.startsWith('promptMessages') &&
        buildMetricRunChatPromptTemplate(form.getValues('promptMessages') ?? [])
      ) {
        form.clearErrors('promptMessages');
      }
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [form]);

  const handleSubmit = async (formData: MetricRunSidePanelFormData) => {
    if (!metric) {
      return;
    }

    if (!formData.dataset) {
      form.setError('dataset', { message: 'Please select a dataset' });
      return;
    }
    if (formData.jobType === 'online') {
      if (!formData.model) {
        form.setError('model', { message: 'Please select a model' });
        return;
      }
      if (!buildMetricRunChatPromptTemplate(formData.promptMessages)) {
        form.setError('promptMessages', { message: 'Please provide at least one prompt message' });
        return;
      }
    }

    try {
      let target: Model | undefined;
      if (formData.jobType === 'online') {
        const { model, adapter } = formData.model!;
        const modelValue = adapter ? `${model}::${adapter}` : model;
        const result = buildModelPayload(
          modelValue,
          selectedModelEntity ? [selectedModelEntity] : [],
          PLATFORM_BASE_URL
        );
        if (!result.ok) {
          toast.error(result.error);
          return;
        }
        const payload = result.payload;
        target =
          typeof payload === 'string'
            ? { url: payload, name: payload }
            : (payload as unknown as Model);
      }

      const onlineJobParams = buildMetricRunOnlineJobParams(formData);
      const promptTemplatePayload = buildMetricRunChatPromptTemplate(formData.promptMessages);

      const request: EvaluateJobRequest =
        formData.jobType === 'online'
          ? {
              spec: {
                metrics: [metric as unknown as MetricInline],
                dataset: formData.dataset,
                target,
                prompt_template: promptTemplatePayload ?? undefined,
                ...(onlineJobParams && { params: onlineJobParams }),
              },
            }
          : {
              spec: {
                metrics: [metric as unknown as MetricInline],
                dataset: formData.dataset,
              },
            };

      const job = await createEvaluateJob({ workspace, data: request });
      toast.success('Metric evaluation job created');
      onOpenChange(false);
      navigate(getEvaluationResultDetailsRoute(workspace, job.name), { flushSync: true });
    } catch (error) {
      const message = getErrorMessage(error as Error, 'Failed to create metric evaluation job');
      logger.error(`MetricRunSidePanel: ${message}`);
      toast.error(message);
    }
  };

  const metricType = metric && 'type' in metric ? (metric.type ?? null) : null;
  const metricDescription =
    metric && 'description' in metric
      ? (metric as { description?: string }).description
      : undefined;
  const promptMessagesError = form.formState.errors.promptMessages?.message;

  return (
    <SidePanel
      open={open}
      onOpenChange={onOpenChange}
      slotHeading="Run a Metric Job"
      modal
      bordered
      className="w-[560px] [&_.nv-side-panel-main]:p-0"
      {...attributes?.SidePanel}
      slotFooter={
        <Flex justify="end" gap="density-md" className="w-full">
          <Button kind="secondary" onClick={() => onOpenChange(false)} disabled={isPending}>
            Cancel
          </Button>
          <Button
            color="brand"
            disabled={isPending}
            onClick={form.handleSubmit(handleSubmit, (errors) => {
              logger.error(`Form validation errors: ${JSON.stringify(errors)}`);
              toast.error('Please fix the errors in the form before submitting.');
            })}
          >
            Continue
          </Button>
        </Flex>
      }
    >
      <FormProvider {...form}>
        <Stack className="overflow-auto h-full" padding="density-xl" gap="density-xl">
          <Text kind="body/regular/md">
            Create a Metric evaluation job to run this metric against your ground truth data.
          </Text>

          {metric && (
            <EvalCard name={metric.name ?? ''} description={metricDescription} type={metricType} />
          )}

          <Controller
            name="jobType"
            control={form.control}
            render={({ field }) => (
              <SegmentedControl
                value={field.value}
                onValueChange={(value) => {
                  const nextJobType = value as MetricRunSidePanelFormData['jobType'];
                  field.onChange(nextJobType);
                  setSelectedJobType(nextJobType);
                }}
                className="w-full"
                items={[
                  { value: 'online', children: 'Online (Model Evaluation)' },
                  { value: 'offline', children: 'Offline (Dataset Evaluation)' },
                ]}
              />
            )}
          />

          {selectedJobType === 'online' && (
            <>
              <Controller
                name="inferenceParams"
                control={form.control}
                render={({ field: inferenceParamsField }) => (
                  <Controller
                    name="model"
                    control={form.control}
                    render={({ field, fieldState }) => (
                      <FormField
                        slotLabel="Model"
                        status={fieldState.error ? 'error' : undefined}
                        slotError={fieldState.error?.message}
                      >
                        <ModelSelectV2
                          {...modelSearch}
                          value={field.value}
                          disabled={isPending}
                          onValueChange={field.onChange}
                          onOpenChange={setModelSelectOpen}
                          fullWidth
                          showParams
                          inferenceParams={inferenceParamsField.value}
                          onInferenceParamsChange={inferenceParamsField.onChange}
                        />
                      </FormField>
                    )}
                  />
                )}
              />

              <FormField
                slotLabel="Evaluation Model Prompt"
                status={promptMessagesError ? 'error' : undefined}
                slotError={promptMessagesError}
              >
                <Stack gap="density-md">
                  {promptMessageFields.map((field, index) => (
                    <ChatCompletionInput<MetricRunSidePanelFormData>
                      key={field.id}
                      control={form.control}
                      name={`promptMessages.${index}`}
                      disabled={isPending}
                      fieldArrayLength={promptMessageFields.length}
                      variables={datasetVariables}
                      contentPlaceholder="Add prompt content"
                      dataTestId={`metric-run-prompt-message-${index}`}
                      onMoveUp={index > 0 ? () => movePromptMessage(index, index - 1) : undefined}
                      onMoveDown={
                        index < promptMessageFields.length - 1
                          ? () => movePromptMessage(index, index + 1)
                          : undefined
                      }
                      onDuplicate={() => {
                        const message = form.getValues(`promptMessages.${index}`);
                        insertPromptMessage(index + 1, { ...message, expanded: true });
                      }}
                      onRemove={() => removePromptMessage(index)}
                      allowRemove={promptMessageFields.length > 1}
                      footer={({ insertVariable }) => (
                        <Flex>
                          <VariableButton
                            variables={datasetVariables}
                            disabled={isPending}
                            onSelect={(variable) => insertVariable(variable.name)}
                          />
                        </Flex>
                      )}
                    />
                  ))}
                  <Flex>
                    <Button
                      type="button"
                      kind="tertiary"
                      size="small"
                      disabled={isPending}
                      onClick={() => appendPromptMessage(defaultChatCompletionMessageRow())}
                    >
                      <Plus className="size-3.5" aria-hidden />
                      Add Message
                    </Button>
                  </Flex>
                </Stack>
              </FormField>
            </>
          )}

          <ControlledDatasetFileSelect
            label="Ground Truth Dataset"
            useControllerProps={{ name: 'dataset', control: form.control }}
            acceptedFileTypes={['.json', '.jsonl', '.csv', '.parquet']}
            setError={(error) => form.setError('dataset', error)}
            clearError={() => form.clearErrors('dataset')}
            workspace={workspace}
          />
          <MetricRunFileValidationPanel
            form={form}
            selectedMetric={metric}
            workspace={workspace}
            onVariablesChange={handleDatasetVariablesChange}
          />

          {selectedJobType === 'online' && <MetricRunExecutionParametersSection />}
        </Stack>
      </FormProvider>
    </SidePanel>
  );
};
