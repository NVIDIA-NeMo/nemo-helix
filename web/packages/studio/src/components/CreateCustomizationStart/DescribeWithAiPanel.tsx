// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import { getEntityReference, getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Banner, Flex, FormField, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import {
  type DraftInputs,
  estimateTrainingRows,
} from '@studio/components/CreateCustomizationStart/aiDraft';
import { DraftResult } from '@studio/components/CreateCustomizationStart/DraftResult';
import type { DescribeWithAiPanelProps } from '@studio/components/CreateCustomizationStart/types';
import {
  MAX_RETRIES,
  useDescribeWithAi,
} from '@studio/components/CreateCustomizationStart/useDescribeWithAi';
import { useCustomizationDatasetValidation } from '@studio/hooks/useCustomizationDatasetValidation';
import { useGymEnvironmentManifest } from '@studio/hooks/useGymEnvironmentManifest';
import { pickDefaultModelName } from '@studio/util/buildSuggestedModelOptions';
import { CUSTOMIZER_SCHEMA_LABELS } from '@studio/util/customizerSchema';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { type FC, type KeyboardEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useController } from 'react-hook-form';

const DRAFTING_MODEL_HELP =
  'Writes the draft and needs tool-calling support. This is not the model being fine-tuned.';

const GOAL_HELP =
  'Mention constraints that matter — cheap to serve, one GPU, a quick test run. Press ⌘/Ctrl + Enter to draft.';

const NO_TRAINING_FILES =
  'No training files were found in this dataset. Customizer needs at least one training file to start fine-tuning.';

const UNRECOGNIZED_FORMAT =
  "This dataset isn't in a format Customizer accepts. Training rows need messages (chat), prompt and completion, chosen and rejected (preference), or responses_create_params and agent_ref (NeMo Gym).";

const NO_ENVIRONMENT = '__none__';

const NO_ENVIRONMENT_OPTION = [{ value: NO_ENVIRONMENT, label: 'None' }];

export const DescribeWithAiPanel: FC<DescribeWithAiPanelProps> = ({ workspace, onDraft }) => {
  const draftingModelSearch = useModelSearch({ workspace });
  const [isBaseModelOpen, setIsBaseModelOpen] = useState(false);
  const baseModelSearch = useModelSearch({
    workspace,
    enabled: isBaseModelOpen,
    filter: { fileset: true },
  });

  const [baseModel, setBaseModel] = useState<ModelEntity | null>(null);
  const [datasetRef, setDatasetRef] = useState<string | null>(null);
  const [environmentRef, setEnvironmentRef] = useState<string | null>(null);

  // The full form's dataset check, without a training type so it detects any format.
  const dataset = useCustomizationDatasetValidation({ fileset: datasetRef ?? undefined });
  const isGymDataset = dataset.schema?.variant === 'grpo-gym';
  const datasetError =
    !datasetRef || dataset.isPending
      ? null
      : dataset.discoveryError
        ? `Couldn't read the dataset: ${getErrorMessage(dataset.discoveryError)}`
        : !dataset.hasTraining
          ? NO_TRAINING_FILES
          : !dataset.schema
            ? UNRECOGNIZED_FORMAT
            : null;
  const { trainingRowCount, rowCountIsEstimate } = estimateTrainingRows(
    dataset.training,
    dataset.encoding.ok
  );

  const pickedEnvironment = isGymDataset ? environmentRef : null;
  const environmentParts = pickedEnvironment ? getPartsFromReference(pickedEnvironment) : null;
  const environment = useGymEnvironmentManifest({
    workspace: environmentParts?.workspace ?? '',
    filesetName: environmentParts?.name ?? '',
  });
  const isEnvironmentLoading = !!pickedEnvironment && environment.isPending;
  const environmentError = pickedEnvironment ? environment.error : null;

  const inputs = useMemo<DraftInputs | null>(() => {
    if (!baseModel || !datasetRef || dataset.isPending || datasetError) return null;
    if (isEnvironmentLoading || environmentError) return null;
    return {
      model: baseModel,
      dataset: {
        fileset: datasetRef,
        schema: dataset.schema,
        hasValidation: dataset.hasValidation,
        shape: dataset.schemaShape,
        trainingRowCount,
        rowCountIsEstimate,
      },
      environment: pickedEnvironment
        ? { fileset: pickedEnvironment, manifest: environment.manifest }
        : null,
    };
  }, [
    baseModel,
    datasetRef,
    dataset.isPending,
    datasetError,
    dataset.schema,
    dataset.hasValidation,
    dataset.schemaShape,
    trainingRowCount,
    rowCountIsEstimate,
    isEnvironmentLoading,
    environmentError,
    pickedEnvironment,
    environment.manifest,
  ]);

  // A valid draft replaces the form until Edit; a failed one stays on the form, where
  // regenerating is one click away. Clearing the draft (a pick changed) returns to the form.
  const [showResult, setShowResult] = useState(false);
  const handleDraft = useCallback(
    (values: CustomizationFormFields | null) => {
      onDraft(values);
      setShowResult(values !== null);
    },
    [onDraft]
  );

  const { form, validation, requestError, isGenerating, retry, generate, clearDraft } =
    useDescribeWithAi(workspace, inputs, handleDraft);
  const isPreparing = dataset.isPending || isEnvironmentLoading;

  // Starts on Studio's suggested chat model, the same pick the Anonymizer and agent
  // creation default to, so the field rarely needs touching.
  const defaultDraftingModel = useMemo(() => {
    const name = pickDefaultModelName(draftingModelSearch.models);
    const model = draftingModelSearch.models.find((candidate) => candidate.name === name);
    return model ? getEntityReference(model) : undefined;
  }, [draftingModelSearch.models]);
  useEffect(() => {
    if (defaultDraftingModel && !form.getValues('model')) {
      form.setValue('model', defaultDraftingModel);
    }
  }, [form, defaultDraftingModel]);

  const { field: modelField, fieldState: modelState } = useController({
    control: form.control,
    name: 'model',
  });
  const { field: baseModelField, fieldState: baseModelState } = useController({
    control: form.control,
    name: 'baseModel',
  });

  // KUI puts onKeyDown on the textarea's wrapper; the key event bubbles up to it.
  const submitOnModEnter = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      event.currentTarget.closest('form')?.requestSubmit();
    }
  };

  const { schema } = dataset;
  const datasetSummary = schema
    ? `${CUSTOMIZER_SCHEMA_LABELS[schema.variant]} · ${rowCountIsEstimate ? '~' : ''}${trainingRowCount.toLocaleString()} examples`
    : undefined;

  if (showResult && validation?.status === 'valid') {
    return (
      <DraftResult
        summary={validation.summary}
        config={validation.config}
        onEdit={() => setShowResult(false)}
      />
    );
  }

  const status = isGenerating
    ? retry > 0
      ? `Fixing validation errors (retry ${retry} of ${MAX_RETRIES})…`
      : 'Drafting settings…'
    : dataset.isPending
      ? 'Validating dataset files…'
      : isEnvironmentLoading
        ? 'Reading the reward environment…'
        : null;

  const failure = requestError ?? (validation?.status === 'invalid' ? validation.errors : null);

  return (
    <form onSubmit={generate} noValidate>
      <Stack gap="density-xl" className="w-full">
        <Grid cols={{ base: 1, md: 2 }} gap="density-xl">
          <FormField
            slotLabel="Base model"
            slotError={baseModelState.error?.message}
            status={baseModelState.error ? 'error' : undefined}
            required
          >
            <ModelSelectV2
              {...baseModelSearch}
              value={baseModelField.value ? { model: baseModelField.value } : null}
              onValueChange={(selection) => {
                baseModelField.onChange(selection.model);
                setBaseModel(selection.entity ?? null);
                clearDraft();
              }}
              onOpenChange={(isOpen) => {
                setIsBaseModelOpen(isOpen);
                if (!isOpen) baseModelField.onBlur();
              }}
              disabled={isGenerating}
              placeholder="Choose the model to fine-tune"
              emptyMessage="No fine-tunable models found"
              hideAdapters
              fullWidth
              dropdownSide="bottom"
              aria-label="Base model"
            />
          </FormField>

          <Stack gap="density-md">
            <FilesetSearchableSelect
              workspace={workspace}
              purpose="dataset"
              useControllerProps={{ control: form.control, name: 'dataset' }}
              formFieldProps={{
                slotLabel: 'Training dataset',
                slotHelp: datasetSummary,
                required: true,
              }}
              triggerPlaceholder="Select a dataset"
              disabled={isGenerating}
              onChange={(reference) => {
                setDatasetRef(reference);
                clearDraft();
              }}
            />
            {datasetError ? (
              <Banner kind="inline" status="error">
                {datasetError}
              </Banner>
            ) : null}
          </Stack>

          {isGymDataset ? (
            <Stack gap="density-md">
              <FilesetSearchableSelect
                workspace={workspace}
                purpose="environment"
                useControllerProps={{ control: form.control, name: 'environment' }}
                formFieldProps={{
                  slotLabel: 'Reward environment',
                  slotHelp:
                    'Detected a NeMo Gym dataset. Optionally pick the environment that scores its rollouts — or pick it later in the form.',
                }}
                leadingOptions={NO_ENVIRONMENT_OPTION}
                triggerPlaceholder="None"
                disabled={isGenerating}
                onChange={(reference) => {
                  setEnvironmentRef(reference === NO_ENVIRONMENT ? null : reference);
                  clearDraft();
                }}
              />
              {environmentError ? (
                <Banner kind="inline" status="error">
                  {`Couldn't read the environment: ${getErrorMessage(environmentError)}`}
                </Banner>
              ) : null}
            </Stack>
          ) : null}
        </Grid>

        <ControlledTextArea
          useControllerProps={{ control: form.control, name: 'prompt' }}
          onChange={clearDraft}
          label="What's the goal of this fine-tune?"
          formFieldProps={{ required: true, slotHelp: GOAL_HELP }}
          rows={3}
          resizeable="auto"
          className="w-full"
          placeholder="For example: teach it a new task, change its tone, or distill a larger model into it."
          disabled={isGenerating}
          layout="vertical"
          onKeyDown={submitOnModEnter}
        />

        <Grid cols={{ base: 1, md: 2 }} gap="density-xl">
          <FormField
            slotLabel="Drafting model"
            slotHelp={DRAFTING_MODEL_HELP}
            slotError={modelState.error?.message}
            status={modelState.error ? 'error' : undefined}
            required
          >
            <ModelSelectV2
              {...draftingModelSearch}
              value={modelField.value ? { model: modelField.value } : null}
              onValueChange={(selection) => modelField.onChange(selection.model)}
              onOpenChange={(isOpen) => {
                if (!isOpen) modelField.onBlur();
              }}
              disabled={isGenerating}
              placeholder="Choose a model"
              fullWidth
              dropdownSide="bottom"
              aria-label="Drafting model"
            />
          </FormField>
        </Grid>

        {failure ? (
          <Banner kind="inline" status="error">
            {typeof failure === 'string' ? (
              failure
            ) : (
              <>
                The draft didn&apos;t pass the checks. Adjust the goal and draft again:
                <ul className="list-disc pl-density-lg">
                  {failure.map((error) => (
                    <li key={error}>
                      <Text kind="body/regular/sm">{error}</Text>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </Banner>
        ) : null}

        <Flex align="center" gap="density-md">
          <LoadingButton
            type="submit"
            kind="secondary"
            loading={isGenerating}
            disabled={isGenerating || isPreparing || !!datasetError || !!environmentError}
          >
            {validation || requestError ? 'Draft again' : 'Draft settings'}
          </LoadingButton>
          {status ? (
            <Text kind="body/regular/sm" className="text-secondary" role="status">
              {status}
            </Text>
          ) : null}
        </Flex>
      </Stack>
    </form>
  );
};
