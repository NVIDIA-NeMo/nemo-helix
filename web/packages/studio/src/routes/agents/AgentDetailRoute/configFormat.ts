// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const SENSITIVE_KEY = /(api[_-]?key|secret|token|password|passwd|credential)/i;
export const MASK = '••••••••';

/** Lowercase, separator-stripped so snake_case and camelCase keys compare equal. */
const normalizeKey = (key: string): string => key.toLowerCase().replace(/[_-]/g, '');

/** Keys that match SENSITIVE_KEY on substring but are ordinary config, not secrets. */
const NON_SENSITIVE_KEYS = new Set(
  [
    'max_tokens',
    'max_new_tokens',
    'max_input_tokens',
    'max_output_tokens',
    'max_completion_tokens',
    'max_prompt_tokens',
    'num_tokens',
    'token_limit',
    'context_window_tokens',
  ].map(normalizeKey)
);

export const isSensitiveKey = (key: string): boolean =>
  !NON_SENSITIVE_KEYS.has(normalizeKey(key)) && SENSITIVE_KEY.test(key);
