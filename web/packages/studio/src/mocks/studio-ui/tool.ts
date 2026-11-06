// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChatCompletionToolsParam } from '@nemo/common/src/zod/tools';

export const mockTool: ChatCompletionToolsParam = {
  type: 'function',
  function: {
    name: 'test_function',
    description: 'A test function for testing purposes',
    parameters: {
      type: 'object',
      properties: {
        param1: {
          type: 'string',
          description: 'First parameter',
        },
        param2: {
          type: 'number',
          description: 'Second parameter',
        },
      },
      required: ['param1'],
    },
  },
};

export const mockDateTool: ChatCompletionToolsParam = {
  type: 'function',
  function: {
    name: 'get_current_date',
    description: 'Get the current date',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
};
