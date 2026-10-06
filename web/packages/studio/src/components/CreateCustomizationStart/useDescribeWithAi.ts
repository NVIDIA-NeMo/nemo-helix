// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { useChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { getErrorMessage } from '@nemo/common/src/utils/error';
import {
  type DraftInputs,
  type DraftValidation,
  validateDraft,
} from '@studio/components/CreateCustomizationStart/aiDraft';
import {
  buildDraftMessages,
  buildRetryMessages,
  draftCustomizationJobTool,
} from '@studio/components/CreateCustomizationStart/aiInstructions';
import { getWorkspaceAndModel } from '@studio/components/NewDataDesignerJobForm/utils';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import type { ChatCompletion } from 'openai/resources/index.mjs';
import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import { type UseFormReturn, useForm } from 'react-hook-form';
import { z } from 'zod';

export const ERROR_NO_TOOL_CALL =
  'The model replied without drafting a config. Try again, or pick a model with tool-calling support.';

/**
 * Rounds of handing validation errors back to the model before showing them, as the skill's
 * agent does when a submit is rejected.
 */
export const MAX_RETRIES = 2;

const describeWithAiFormSchema = z.object({
  /** URN of the chat model that drafts the config — not the model being fine-tuned. */
  model: z.string().min(1, 'Choose a model to draft the config.'),
  baseModel: z.string().min(1, 'Pick the model to fine-tune.'),
  dataset: z.string().min(1, 'Pick the training dataset.'),
  /** Only offered for NeMo Gym datasets; empty means pick it later in the form. */
  environment: z.string(),
  prompt: z.string().trim().min(1, 'Describe the goal of this fine-tune.'),
});

type DescribeWithAiFormValues = z.infer<typeof describeWithAiFormSchema>;

interface DescribeWithAiState {
  form: UseFormReturn<DescribeWithAiFormValues>;
  validation: DraftValidation | null;
  /** The request itself failed (network, auth, model error), as opposed to a bad draft. */
  requestError: string | null;
  isGenerating: boolean;
  /** Which automatic fix round is running: 0 for the first draft, up to MAX_RETRIES. */
  retry: number;
  generate: (event?: FormEvent) => Promise<void>;
  /** Drops the current draft, for when the picks it was built around change. */
  clearDraft: () => void;
}

/**
 * Drafts a starting config for the picked model and dataset. `onDraft` fires after every
 * run — with form values when the draft loads, null otherwise — so a failed regeneration
 * clears an earlier success.
 *
 * `inputs` is null until the picked model and dataset have both been read; the panel keeps
 * Generate disabled until then.
 */
export const useDescribeWithAi = (
  workspace: string,
  inputs: DraftInputs | null,
  onDraft: (values: CustomizationFormFields | null) => void
): DescribeWithAiState => {
  const form = useForm<DescribeWithAiFormValues>({
    resolver: zodResolver(describeWithAiFormSchema),
    defaultValues: { model: '', baseModel: '', dataset: '', environment: '', prompt: '' },
  });
  const [validation, setValidation] = useState<DraftValidation | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [retry, setRetry] = useState(0);

  const chatCompletion = useChatCompletion();

  // Leaving the AI option unmounts the panel while a run may still be in flight. Aborting it
  // keeps a late reply from publishing a draft the user can no longer see.
  const runRef = useRef<AbortController | null>(null);
  useEffect(() => () => runRef.current?.abort(), []);

  const runGeneration = useCallback(
    async ({ model, prompt }: DescribeWithAiFormValues) => {
      if (!inputs) return;
      runRef.current?.abort();
      const run = new AbortController();
      runRef.current = run;
      setIsGenerating(true);
      setRetry(0);
      setRequestError(null);
      onDraft(null);
      const { workspace: chatWorkspace, name } = getWorkspaceAndModel(model, workspace);
      let messages = buildDraftMessages(prompt, inputs);
      try {
        for (let attempt = 0; ; attempt += 1) {
          setRetry(attempt);
          const response = (await chatCompletion.mutateAsync({
            workspace: chatWorkspace,
            model: name,
            stream: false,
            messages,
            tools: [draftCustomizationJobTool],
            tool_choice: 'required',
            signal: run.signal,
          })) as ChatCompletion;
          if (run.signal.aborted) return;

          const toolCall = response.choices[0]?.message?.tool_calls?.[0];
          if (!toolCall || toolCall.type !== 'function') {
            setValidation({ status: 'invalid', errors: [ERROR_NO_TOOL_CALL] });
            return;
          }
          const result = validateDraft(toolCall.function.arguments, inputs);

          if (result.status === 'valid' || attempt >= MAX_RETRIES) {
            setValidation(result);
            onDraft(result.status === 'valid' ? result.values : null);
            return;
          }
          messages = buildRetryMessages(messages, toolCall.function.arguments, result.errors);
        }
      } catch (error) {
        if (run.signal.aborted) return;
        setRequestError(getErrorMessage(error, 'Generation failed.'));
        setValidation(null);
      } finally {
        setIsGenerating(false);
      }
    },
    [chatCompletion, inputs, onDraft, workspace]
  );

  const clearDraft = useCallback(() => {
    setValidation(null);
    setRequestError(null);
    onDraft(null);
  }, [onDraft]);

  return {
    form,
    validation,
    requestError,
    isGenerating,
    retry,
    generate: form.handleSubmit(runGeneration),
    clearDraft,
  };
};
