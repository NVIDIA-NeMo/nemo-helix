// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { CUSTOMIZATION_FILESET_FILEPATHS } from '@studio/constants/customization';

export const files: FilesetFileOutput[] = [
  {
    size: 127,
    path: 'hello-world.json',
    file_ref: 'ref-1',
    file_url: 'https://example.com/hello-world.json',
  },
  {
    size: 128,
    path: 'hello-world2.json',
    file_ref: 'ref-2',
    file_url: 'https://example.com/hello-world2.json',
  },
  {
    size: 129,
    path: 'hello-world3.json',
    file_ref: 'ref-3',
    file_url: 'https://example.com/hello-world3.json',
  },
];

export const customizationFiles: FilesetFileOutput[] = [
  {
    size: 130,
    file_ref: 'ref-eval',
    path: 'eval_medical_input_array.json',
    file_url: 'https://example.com/eval_medical_input_array.json',
  },
  {
    size: 131,
    file_ref: 'ref-train',
    path: CUSTOMIZATION_FILESET_FILEPATHS.Training,
    file_url: 'https://example.com/training.json',
  },
  {
    size: 132,
    file_ref: 'ref-val',
    path: CUSTOMIZATION_FILESET_FILEPATHS.Validation,
    file_url: 'https://example.com/validation.json',
  },
];

export const evaluationFiles: FilesetFileOutput[] = [
  {
    file_ref: 'mock-ref-1',
    size: 1234,
    path: 'eval_data.jsonl',
    file_url: 'https://example.com/eval_data.jsonl',
  },
  {
    file_ref: 'mock-ref-2',
    size: 5678,
    path: 'test_data.jsonl',
    file_url: 'https://example.com/test_data.jsonl',
  },
  {
    file_ref: 'mock-ref-3',
    size: 789,
    path: 'invalid_data.jsonl',
    file_url: 'https://example.com/invalid_data.jsonl',
  },
];

/**
 * Mock valid JSONL file content with messages schema format for evaluation
 */
export const messagesJSONLContentValid = (() => {
  const line1 = JSON.stringify({
    messages: [
      { role: 'user', content: 'What is 2+2?' },
      { role: 'assistant', content: '4' },
    ],
    reference: '4',
  });
  const line2 = JSON.stringify({
    messages: [
      { role: 'user', content: 'What is the capital of France?' },
      { role: 'assistant', content: 'Paris' },
    ],
    reference: 'Paris',
  });
  return `${line1}\n${line2}`;
})();

/**
 * Mock invalid JSONL file content for testing validation failures
 */
export const messagesJSONLContentInvalid = (() => {
  // Invalid JSON - missing closing brace
  const line1 = '{"messages": [{"role": "user", "content": "What is 2+2?"}';
  const line2 = 'This is not JSON at all';
  return `${line1}\n${line2}`;
})();
