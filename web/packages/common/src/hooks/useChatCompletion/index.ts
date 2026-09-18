// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGatewayProxyGetQueryKey } from '@nemo/sdk/generated/platform/inference-gateway';
import { selectOidcBearerToken } from '@nemo/sdk/src/utils/oidcBearerToken';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';
import OpenAI from 'openai';
import type { ChatCompletion, ChatCompletionCreateParams } from 'openai/resources/index.mjs';
import { useAuth } from 'react-oidc-context';

import { ChatMissingModelError, CHAT_CORS_HEADERS } from '../../constants/chat';
import { PLATFORM_BASE_URL } from '../../constants/environment';
import type { ChatCompletionRequestReturn } from '../../types/chat';

export type UseChatCompletionParams = ChatCompletionCreateParams & {
  workspace?: string;
  baseURL?: string;
  accessToken?: string;
  signal?: AbortSignal;
};

export type UseChatCompletionOptions = Omit<
  UseMutationOptions<ChatCompletionRequestReturn, Error, UseChatCompletionParams>,
  'mutationFn'
>;

// Cache clients by baseURL to avoid creating new instances on every request
const clientCache = new Map<string, OpenAI>();

const getClient = (baseURL: string): OpenAI => {
  if (!clientCache.has(baseURL)) {
    clientCache.set(
      baseURL,
      new OpenAI({
        apiKey: '',
        baseURL,
        dangerouslyAllowBrowser: true,
      })
    );
  }
  return clientCache.get(baseURL)!;
};

export const createChatCompletion = async (
  props: UseChatCompletionParams
): Promise<ChatCompletionRequestReturn> => {
  const {
    model,
    workspace,
    baseURL,
    max_tokens,
    messages,
    stream,
    accessToken,
    signal,
    ...moreOptions
  } = props;
  if (!model) {
    throw new ChatMissingModelError();
  }

  let resolvedBaseURL = baseURL;
  if (!resolvedBaseURL && workspace) {
    resolvedBaseURL = PLATFORM_BASE_URL + getGatewayProxyGetQueryKey(workspace, model, 'v1/')[0];
  } else if (!resolvedBaseURL) {
    throw new Error('Unable to resolve client base URL');
  }

  if (!accessToken) {
    console.warn('Warning: No access token provided. Authentication may be required.');
  }

  const client = getClient(resolvedBaseURL);

  const request = client.chat.completions.create(
    {
      model,
      max_tokens,
      messages,
      temperature: 1,
      stream,
      ...moreOptions,
    },
    {
      headers: {
        ...CHAT_CORS_HEADERS,
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      },
      signal,
    }
  );

  if (stream) {
    const response = await request.asResponse();
    const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
    if (contentType.includes('json')) {
      return (await response.json()) as ChatCompletion;
    }
  }

  return request;
};

/**
 * Return a useMutation object for creating a chat completion using the NIM Proxy microservice.
 * This hook differs from `useChat` in that this hook only simply creates a chat completion response using OpenAI's SDK.
 * This hook does not manage state using the chat provider.
 *
 * @returns {Stream<ChatCompletionChunk> | ChatCompletion} A chat stream or a chat completion. If using a stream, use a combination of
 * useState and useEffect to update the view as the stream chunks arrive.
 * ex. ```
 * if (stream instanceof Stream) {
        for await (const chunk of stream) {
          const content = chunk.choices[0].delta.content ?? '';
          setAnswer((prevAnswer) => prevAnswer + content);
        }
      }
 * ```
 */
export const useChatCompletion = (mutationOptions?: UseChatCompletionOptions) => {
  const auth = useAuth();
  const bearerToken = selectOidcBearerToken(
    auth?.user,
    import.meta.env.VITE_AUTH_BEARER_TOKEN_SOURCE
  );
  return useMutation({
    ...mutationOptions,
    mutationFn: (params: UseChatCompletionParams) =>
      createChatCompletion({ accessToken: bearerToken, ...params }),
  });
};
