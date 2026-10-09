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

/** A retry would hit the same output limit, so this is shown rather than retried. */
export const ERROR_TRUNCATED =
  'The model ran out of output tokens before finishing the draft. Try again, or pick a drafting model with a larger output limit.';

export const ERROR_INPUTS_NOT_READY =
  "Studio hasn't finished reading the base model or dataset. Pick them again, then draft.";

const CONTEXT_HINT =
  "The drafting instructions may not fit in this model's context window. Pick a drafting model with a larger context.";

/** Providers word it differently: "maximum context length", "context_length_exceeded", "too long". */
const CONTEXT_ERROR = /context|too long|maximum.*tokens/i;

/** Low, so the structured job comes out consistent and fewer drafts need a retry. */
const TEMPERATURE = 0.2;

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
  /** Validates the form, then drafts from `inputs` — null until the picks have been read. */
  generate: (inputs: DraftInputs | null, event?: FormEvent) => Promise<void>;
  /** Drops the current draft, for when the picks it was built around change. */
  clearDraft: () => void;
}

/**
 * Drafts a starting config for the picked model and dataset. `onDraft` fires after every
 * run — with form values when the draft loads, null otherwise — so a failed regeneration
 * clears an earlier success.
 */
export const useDescribeWithAi = (
  workspace: string,
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
    async ({ model, prompt }: DescribeWithAiFormValues, inputs: DraftInputs | null) => {
      if (!inputs) {
        setRequestError(ERROR_INPUTS_NOT_READY);
        return;
      }
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
            temperature: TEMPERATURE,
            signal: run.signal,
          })) as ChatCompletion;
          if (run.signal.aborted) return;

          // Cut off mid-reply, the tool call's JSON is incomplete and would fail as "not
          // valid JSON" on every retry.
          if (response.choices[0]?.finish_reason === 'length') {
            setValidation({ status: 'invalid', errors: [ERROR_TRUNCATED] });
            return;
          }
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
        const message = getErrorMessage(error, 'Generation failed.');
        setRequestError(CONTEXT_ERROR.test(message) ? `${message} ${CONTEXT_HINT}` : message);
        setValidation(null);
      } finally {
        // A newer run aborts this one but is still in flight, so only the latest run settles.
        if (runRef.current === run) setIsGenerating(false);
      }
    },
    [chatCompletion, onDraft, workspace]
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
    generate: (inputs, event) =>
      form.handleSubmit((values) => runGeneration(values, inputs))(event),
    clearDraft,
  };
};
