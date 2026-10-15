// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChatCompletionMessageRowValues } from '@nemo/common/src/components/ChatCompletionInput';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2';
import type { RunConfigOnlineModel } from '@nemo/sdk/generated/evals/schema';
import type { MetricRunSidePanelFormData } from '@studio/components/sidePanels/MetricRunSidePanel/types';
import { parseEvaluationModelValue } from '@studio/util/evaluations';

export interface MetricRunChatPromptTemplate {
  [key: string]: unknown;
  messages: Array<Pick<ChatCompletionMessageRowValues, 'role' | 'content'>>;
}

export const getModelSelectionFromSearchParam = (
  modelParam: string | null,
  workspace: string
): ModelSelection | null => {
  const trimmedModelParam = modelParam?.trim();
  if (!trimmedModelParam) return null;

  const { modelUrn, adapterName } = parseEvaluationModelValue(trimmedModelParam);
  if (!modelUrn) return null;

  return {
    model: modelUrn.includes('/') ? modelUrn : `${workspace}/${modelUrn}`,
    ...(adapterName ? { adapter: adapterName } : {}),
  };
};

export const getPromptTemplateTextForValidation = (
  messages: ChatCompletionMessageRowValues[]
): string => messages.map((message) => message.content).join('\n');

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const getMetricPromptTemplateTextForValidation = (promptTemplate: unknown): string => {
  if (typeof promptTemplate === 'string') return promptTemplate;
  if (!isRecord(promptTemplate)) return '';

  const messages = promptTemplate.messages;
  if (Array.isArray(messages)) {
    return messages
      .map((message) =>
        isRecord(message) && typeof message.content === 'string' ? message.content : ''
      )
      .filter(Boolean)
      .join('\n');
  }

  return Object.values(promptTemplate)
    .map((value) =>
      typeof value === 'string' ? value : getMetricPromptTemplateTextForValidation(value)
    )
    .filter(Boolean)
    .join('\n');
};

export const getMetricRunValidationPromptTemplate = ({
  metricPromptTemplate,
  promptMessages,
}: {
  metricPromptTemplate: unknown;
  promptMessages: ChatCompletionMessageRowValues[];
}): string =>
  [
    getMetricPromptTemplateTextForValidation(metricPromptTemplate),
    getPromptTemplateTextForValidation(promptMessages),
  ]
    .filter(Boolean)
    .join('\n');

export const buildMetricRunChatPromptTemplate = (
  messages: ChatCompletionMessageRowValues[]
): MetricRunChatPromptTemplate | null => {
  const normalizedMessages = messages
    .map(({ role, content }) => ({ role, content }))
    .filter((message) => message.content.trim().length > 0);

  return normalizedMessages.length > 0 ? { messages: normalizedMessages } : null;
};

export const buildMetricRunOnlineJobParams = (
  formData: Pick<MetricRunSidePanelFormData, 'inferenceParams' | 'ignore_request_failure'>
): RunConfigOnlineModel | undefined => {
  const sanitizedInferenceParams = Object.fromEntries(
    Object.entries(formData.inferenceParams).filter(
      ([, value]) => value !== undefined && value !== null
    )
  );
  const hasInferenceParams = Object.keys(sanitizedInferenceParams).length > 0;
  if (!hasInferenceParams && !formData.ignore_request_failure) {
    return undefined;
  }

  return {
    ...(hasInferenceParams ? { inference: sanitizedInferenceParams } : {}),
    ...(formData.ignore_request_failure ? { ignore_request_failure: true } : {}),
  };
};
