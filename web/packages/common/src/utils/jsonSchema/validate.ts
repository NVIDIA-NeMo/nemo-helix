// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ParseAndValidateResult =
  | { valid: true; value: Record<string, unknown> }
  | { valid: false; errors: string[] };

/**
 * Parse editor text as a JSON object. Whether it is a valid JSON Schema is left
 * to the backend's `check_schema` on save: client-side meta-schema validators
 * (Ajv) compile with `new Function`, which Studio's CSP blocks.
 */
export function parseAndValidate(text: string): ParseAndValidateResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    return { valid: false, errors: [`Invalid JSON: ${(err as Error).message}`] };
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { valid: false, errors: ['JSON Schema document must be a JSON object'] };
  }
  return { valid: true, value: parsed as Record<string, unknown> };
}
