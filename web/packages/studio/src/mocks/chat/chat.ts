// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChatCompletionChunk } from 'openai/resources/index.mjs';

export const chatCompletionRequest1 = {
  model: 'meta/llama-3_1-8b-instruct',
  max_tokens: 128,
  stream: true,
  temperature: 1,
  messages: [
    {
      role: 'system',
      content:
        'You are a Javascript expert who helps users write Javascript code to solve problems',
    },
    {
      role: 'user',
      content: 'How do I concatenate two strings in Javascript?',
    },
  ],
};

export const chatCompletionRequest2 = {
  model: 'meta/llama-3_1-8b-instruct',
  max_tokens: 128,
  stream: true,
  temperature: 1,
  messages: [
    {
      role: 'system',
      content: 'You are a Python expert who helps users write Python code to solve problems',
    },
    {
      role: 'user',
      content: 'How do I concatenate two strings in Python?',
    },
  ],
};

export const chatCompletionRequest3 = {
  model: 'meta/llama-3_1-8b-instruct',
  max_tokens: 128,
  stream: true,
  temperature: 1,
  messages: [
    {
      role: 'system',
      content: 'You are an HTML expert who helps users write HTML code to solve problems',
    },
    {
      role: 'user',
      content: 'How do I concatenate two strings in HTML?',
    },
    {
      role: 'assistant',
      content: `In HTML, you can concatenate two strings using the & symbol and the concatenate operator (&), like this: string1&string2.`,
    },
    {
      role: 'user',
      content: 'Is HTML a programming language?',
    },
  ],
};

export const chatCompletionResponse1 = {
  id: 'cmpl-3720fa9e8ffa4c9eb1eea903b466fb72',
  object: 'chat.completion',
  created: 1729696730,
  model: 'meta/llama-3_1-8b-instruct',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: `In JavaScript, you can concatenate two strings by combining them into one string using a method that ties" the two strings together.\nThere are a couple of ways to do this: 1. Using the plus sign (+) between the two strings: This is a simple and intuitive way to concatenate two strings. 2. Using a function or method that specifically concatenates strings: JavaScript provides a built-in method called concat() that can be used to concatenate strings. Both of these methods essentially create a new string that contains the characters of both original strings in sequence.`,
      },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 30,
    total_tokens: 500,
  },
};

export const chatCompletionResponse2 = {
  id: 'cmpl-e07f7db0100b4d0589f8ad26dcc4a6b0',
  object: 'chat.completion',
  created: 1729696730,
  model: 'meta/llama-3_1-8b-instruct',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: `In Python, you can concatenate two strings using the + operator, like this: "string1" + "string2". This will result in a new string that combines the characters of both strings. Alternatively, you can also use the concat() method or the format() function to concatenate strings in Python.`,
      },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 30,
    total_tokens: 500,
  },
};

export const chatCompletionResponse3 = {
  id: 'cmpl-c1deaac4017b4df695d0d74efbbb84c7',
  object: 'chat.completion',
  created: 1729696730,
  model: 'meta/llama-3_1-8b-instruct',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: `HTML (Hypertext Markup Language) is not a programming language in the classical sense. It's a markup language used to create the structure and content of web pages. While it's used to create content, it's primarily focused on defining the layout, formatting, and links between different parts of a web page, rather than executing logic or performing calculations.\nHTML doesn't have the ability to execute code or manipulate data in the same way that programming languages like Python, Java, or JavaScript do. Instead, it relies on external scripts (often written in a programming language) to perform dynamic functionality on the web page.`,
      },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 30,
    total_tokens: 500,
  },
};

export const chatCompletionStreamResponse1: ChatCompletionChunk = {
  id: 'cmpl-6dcf78b459274f10a2dd2d497c71d415',
  object: 'chat.completion.chunk',
  created: 1724971101,
  model: 'meta/llama-3_1-8b-instruct',

  choices: [
    {
      index: 0,
      delta: {
        role: 'assistant',
        content: 'Hello',
      },
      logprobs: null,
      finish_reason: null,
    },
  ],
};

export const chatCompletionStreamResponse2: ChatCompletionChunk = {
  id: 'cmpl-6dcf78b459274f10a2dd2d497c71d416',
  object: 'chat.completion.chunk',
  created: 1724971101,
  model: 'meta/llama-3_1-8b-instruct',
  choices: [
    {
      index: 0,
      delta: {
        role: 'assistant',
        content: '.',
      },
      logprobs: null,
      finish_reason: 'stop',
    },
  ],
};

export const chatCompletionStreamResponseEnd: string = 'data: [DONE]';

export const streamResponseFinalText =
  chatCompletionStreamResponse1.choices[0].delta.content! +
  chatCompletionStreamResponse2.choices[0].delta.content;

/**
 * Formats a ChatCompletionChunk as a data chunk for the OpenAI stream response.
 * The OpenAI stream response expects a "data: " prefix and a "\n\n" suffix.
 * @param response
 * @returns
 */
export const asDataChunk = (response: ChatCompletionChunk) => {
  return `data: ${JSON.stringify(response)}\n\n`;
};
