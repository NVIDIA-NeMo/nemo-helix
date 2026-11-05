// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionFunctionMessageParam,
  ChatCompletionMessageParam,
} from 'openai/resources/index.mjs';
import { Stream } from 'openai/streaming.mjs';

export type ChatCompletionRequestReturn = Stream<ChatCompletionChunk> | ChatCompletion;
export type ChatCompletionRequestReturnData = ChatCompletionChunk | ChatCompletion;

/**
 * An extension of the OpenAI's ChatCompletionMessageParam type that excludes the function parameter
 * because it's deprecated and would add extra boilerplate.
 */
export type ExcludedChatCompletionMessageParam = Exclude<
  ChatCompletionMessageParam,
  ChatCompletionFunctionMessageParam
>;
