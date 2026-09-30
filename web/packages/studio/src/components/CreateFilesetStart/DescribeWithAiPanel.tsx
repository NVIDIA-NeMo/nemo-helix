// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { Flex, FormField, Stack } from '@nvidia/foundations-react-core';
import { PROMPT_SUGGESTIONS } from '@studio/components/CreateFilesetStart/constants';
import { GeneratedConfigResult } from '@studio/components/CreateFilesetStart/GeneratedConfigResult';
import type { DescribeWithAiPanelProps } from '@studio/components/CreateFilesetStart/types';
import { useDescribeWithAi } from '@studio/components/CreateFilesetStart/useDescribeWithAi';
import { PromptSuggestionTags } from '@studio/components/PromptSuggestionTags';
import { providerForSelection } from '@studio/routes/DataDesignerJobBuildRoute/models';
import type { FC } from 'react';
import { useController } from 'react-hook-form';

const PROMPT_PLACEHOLDER =
  'Describe the rows you want: how many, what each column holds, and how the data should vary. Or start from an example below.';

const MODEL_HELP = 'Needs tool-calling support. This model will be used in LLM columns.';

/**
 * "Describe with AI" detail panel: pick the model that drafts the config, describe the fileset
 * in plain language, and see whether the draft is loadable. The generated config never reaches
 * the builder unvalidated — the parent gates Continue on `onValidConfig` returning a request.
 */
export const DescribeWithAiPanel: FC<DescribeWithAiPanelProps> = ({ workspace, onValidConfig }) => {
  const modelSearch = useModelSearch({ workspace });
  const { form, validation, requestError, rawOutput, pendingAction, generate, requestFix } =
    useDescribeWithAi(workspace, onValidConfig, modelSearch.groups);
  const isBusy = pendingAction !== null;

  const { field: modelField, fieldState: modelState } = useController({
    control: form.control,
    name: 'model',
  });
  const modelValue: ModelSelection | null = modelField.value ? { model: modelField.value } : null;

  const showSuggestions = form.watch('prompt').trim().length === 0 && !isBusy;

  return (
    <form onSubmit={generate} noValidate>
      <Flex gap="density-xl" className="w-full flex-wrap items-stretch">
        <Stack gap="density-md" className="min-w-[320px] flex-1">
          <FormField
            slotLabel="Model"
            slotHelp={MODEL_HELP}
            slotError={modelState.error?.message}
            status={modelState.error ? 'error' : undefined}
            required
          >
            <ModelSelectV2
              {...modelSearch}
              value={modelValue}
              onValueChange={(selection) => {
                modelField.onChange(selection.model);
                form.setValue('provider', providerForSelection(selection));
              }}
              onOpenChange={(isOpen) => {
                if (!isOpen) modelField.onBlur();
              }}
              disabled={isBusy}
              placeholder="Choose a model"
              fullWidth
              dropdownSide="bottom"
              aria-label="Model that drafts the config"
            />
          </FormField>

          <ControlledTextArea
            useControllerProps={{ control: form.control, name: 'prompt' }}
            label="What do you want to generate?"
            formFieldProps={{ required: true }}
            rows={8}
            className="w-full"
            placeholder={PROMPT_PLACEHOLDER}
            disabled={isBusy}
            layout="vertical"
            slotEnd={
              showSuggestions ? (
                <PromptSuggestionTags
                  suggestions={PROMPT_SUGGESTIONS}
                  onSelect={(prompt) =>
                    form.setValue('prompt', prompt, { shouldValidate: true, shouldDirty: true })
                  }
                />
              ) : undefined
            }
          />

          <Flex justify="start">
            <LoadingButton
              type="submit"
              kind="secondary"
              loading={pendingAction === 'generate'}
              disabled={isBusy}
            >
              {validation || requestError ? 'Regenerate' : 'Generate'}
            </LoadingButton>
          </Flex>
        </Stack>

        <Stack gap="density-md" className="min-w-[320px] flex-1">
          <GeneratedConfigResult
            validation={validation}
            requestError={requestError}
            rawOutput={rawOutput}
            isGenerating={pendingAction === 'generate'}
            isFixing={pendingAction === 'fix'}
            onFix={() => void requestFix()}
          />
        </Stack>
      </Flex>
    </form>
  );
};
