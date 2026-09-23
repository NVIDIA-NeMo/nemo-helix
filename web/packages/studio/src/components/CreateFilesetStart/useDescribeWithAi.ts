// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { type ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { useChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { getErrorMessage } from '@nemo/common/src/utils/error';
import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { buildFixMessages } from '@studio/components/CreateFilesetStart/fixRequest';
import { DATA_DESIGNER_JOB_GENERATOR_SYSTEM_PROMPT } from '@studio/components/NewDataDesignerJobForm/constants';
import { generateDataDesignerJobRequestTool } from '@studio/components/NewDataDesignerJobForm/tools';
import {
  getWorkspaceAndModel,
  parseToolResponseToJobRequest,
  sanitizeJobRequestName,
} from '@studio/components/NewDataDesignerJobForm/utils';
import {
  type GeneratedConfigValidation,
  type GenerationModel,
  validateGeneratedJobRequest,
} from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import type { ChatCompletion, ChatCompletionMessageParam } from 'openai/resources/index.mjs';
import { type FormEvent, useCallback, useState } from 'react';
import { type UseFormReturn, useForm } from 'react-hook-form';
import { z } from 'zod';

export const ERROR_NO_TOOL_CALL =
  'The model replied without calling the job-config tool. Try again, or pick a model with tool-calling support.';
export const ERROR_PARSE_RESPONSE =
  "The model's response could not be read as a Data Designer job config.";

export const MODEL_REQUIRED_MESSAGE = 'Choose a model to draft the config.';
export const PROMPT_REQUIRED_MESSAGE = 'Describe the fileset you want.';

/**
 * The panel's inputs, owned by React Hook Form. Rules live here rather than on the individual
 * fields so a submit can never reach the model with an empty prompt, whatever renders the form.
 */
export const describeWithAiFormSchema = z.object({
  /** URN of the model that drafts the config (e.g. `workspace/model-name`). */
  model: z.string().min(1, MODEL_REQUIRED_MESSAGE),
  /**
   * Provider of the selected model, captured at selection time. Data Designer requires an
   * explicit provider on every model config, and reading it back from a filtered search list
   * later is unreliable — so the panel stores it alongside the model.
   */
  provider: z.string(),
  prompt: z.string().trim().min(1, PROMPT_REQUIRED_MESSAGE),
});

export type DescribeWithAiFormValues = z.infer<typeof describeWithAiFormSchema>;

export interface DescribeWithAiState {
  form: UseFormReturn<DescribeWithAiFormValues>;
  /** Result of the last generation, or null before the first run. */
  validation: GeneratedConfigValidation | null;
  /** Set when the request itself failed (network, auth, model error) rather than the output. */
  requestError: string | null;
  /**
   * The model's tool-call arguments from the last run, pretty-printed. Kept whether or not the
   * draft validated, so a rejected config can still be inspected.
   */
  rawOutput: string | null;
  /** Which request is in flight, or null when idle. */
  pendingAction: DescribeWithAiAction | null;
  /** Submit handler: validates the form, then runs one generation. */
  generate: (event?: FormEvent) => Promise<void>;
  /**
   * Sends the last draft back to the same model with the issues the builder found, and
   * replaces the result with whatever comes back. No-op when there is nothing to fix.
   */
  requestFix: () => Promise<void>;
}

/** Distinguishes a first draft from a repair run; both hit the same model and tool. */
export type DescribeWithAiAction = 'generate' | 'fix';

/** Pretty-print the tool-call arguments, falling back to the raw string if they aren't JSON. */
const formatRawOutput = (args: string): string => {
  try {
    return JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    return args;
  }
};

/**
 * Drives the "Describe with AI" panel: pick a model, describe the fileset, and have the model
 * draft a Data Designer job config via tool call. The draft is only accepted once it survives
 * {@link validateGeneratedJobRequest}, so the caller can gate Continue on `onValidConfig`
 * having produced a request.
 *
 * `onValidConfig` is called after every run — with the request when the draft is loadable, and
 * with null otherwise — so a failed regeneration clears a previously-valid result.
 */
export const useDescribeWithAi = (
  workspace: string,
  onValidConfig: (jobRequest: DataDesignerJobRequest | null) => void,
  modelGroups: ModelWorkspaceGroup[]
): DescribeWithAiState => {
  const form = useForm<DescribeWithAiFormValues>({
    resolver: zodResolver(describeWithAiFormSchema),
    defaultValues: { model: '', provider: '', prompt: '' },
  });
  const [validation, setValidation] = useState<GeneratedConfigValidation | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [rawOutput, setRawOutput] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<DescribeWithAiAction | null>(null);

  const chatCompletion = useChatCompletion();

  const settle = useCallback(
    (result: GeneratedConfigValidation) => {
      setValidation(result);
      onValidConfig(result.status === 'valid' ? result.jobRequest : null);
    },
    [onValidConfig]
  );

  /**
   * One round-trip to the model: send `messages`, then settle the panel on whatever comes back.
   * Shared by the first draft and the repair run so both are validated identically.
   */
  const runCompletion = useCallback(
    async (generationModel: GenerationModel, messages: ChatCompletionMessageParam[]) => {
      const model = generationModel.model;
      setRequestError(null);
      setRawOutput(null);
      onValidConfig(null);
      const { workspace: chatWorkspace, name: modelName } = getWorkspaceAndModel(model, workspace);

      try {
        const response = (await chatCompletion.mutateAsync({
          workspace: chatWorkspace,
          model: modelName,
          stream: false,
          messages,
          tools: [generateDataDesignerJobRequestTool],
          tool_choice: 'required',
        })) as ChatCompletion;

        const toolCall = response.choices[0]?.message?.tool_calls?.[0];
        if (!toolCall) {
          settle({ status: 'invalid', errors: [ERROR_NO_TOOL_CALL], warnings: [] });
          return;
        }

        setRawOutput(formatRawOutput(toolCall.function.arguments));

        const jobRequest = parseToolResponseToJobRequest(toolCall.function.arguments);
        if (!jobRequest?.spec?.config) {
          settle({ status: 'invalid', errors: [ERROR_PARSE_RESPONSE], warnings: [] });
          return;
        }

        settle(
          validateGeneratedJobRequest(
            sanitizeJobRequestName(jobRequest),
            modelGroups,
            generationModel
          )
        );
      } catch (error) {
        setRequestError(getErrorMessage(error, 'Generation failed.'));
        setValidation(null);
        onValidConfig(null);
      }
    },
    [chatCompletion, modelGroups, onValidConfig, settle, workspace]
  );

  const run = useCallback(
    async (
      action: DescribeWithAiAction,
      generationModel: GenerationModel,
      messages: ChatCompletionMessageParam[]
    ) => {
      setPendingAction(action);
      try {
        await runCompletion(generationModel, messages);
      } finally {
        setPendingAction(null);
      }
    },
    [runCompletion]
  );

  const runGeneration = useCallback(
    ({ model, provider, prompt }: DescribeWithAiFormValues) =>
      run('generate', { model, provider }, [
        { role: 'system', content: DATA_DESIGNER_JOB_GENERATOR_SYSTEM_PROMPT },
        { role: 'user', content: prompt },
      ]),
    [run]
  );

  const requestFix = useCallback(async () => {
    // Nothing to repair without a draft to send back, and nothing to repair it against.
    if (!rawOutput || !validation) return;
    const { model, provider, prompt } = form.getValues();
    if (!model) return;

    await run(
      'fix',
      { model, provider },
      buildFixMessages({
        prompt,
        config: rawOutput,
        errors: validation.status === 'invalid' ? validation.errors : [],
        warnings: validation.warnings,
      })
    );
  }, [form, rawOutput, run, validation]);

  return {
    form,
    validation,
    requestError,
    rawOutput,
    pendingAction,
    generate: form.handleSubmit(runGeneration),
    requestFix,
  };
};
