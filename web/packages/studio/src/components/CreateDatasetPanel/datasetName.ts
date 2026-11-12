// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Name validation (spec 3.1, 3.2) and name inference (spec 3.3) for the Create Dataset panel.

import { ENTITY_NAME_MAX_LENGTH, ENTITY_NAME_MIN_LENGTH } from '@nemo/common/src/utils/entityName';

/**
 * Returns the first applicable error message for `name`, or undefined when valid.
 * A duplicate takes precedence over every format error; the rest follow spec 3.2 table order.
 */
export function getDatasetNameError(
  name: string,
  opts: { taken: boolean; workspace: string }
): string | undefined {
  if (opts.taken) return `A Dataset named ${name} already exists in ${opts.workspace}`;
  if (name === '') return 'Enter a name for this Dataset';
  if (!/^[a-z]/.test(name)) return 'Names must start with a lowercase letter';
  if (name.length < ENTITY_NAME_MIN_LENGTH)
    return `Names must be at least ${ENTITY_NAME_MIN_LENGTH} characters`;
  if (name.length > ENTITY_NAME_MAX_LENGTH)
    return `Names must be ${ENTITY_NAME_MAX_LENGTH} characters or fewer`;
  if (name.includes('--')) return "Names can't contain two hyphens in a row";
  if (name.endsWith('-')) return "Names can't end with a hyphen";
  if (!/^[a-z0-9.-]+$/.test(name)) return 'Use lowercase letters, digits, hyphens, and dots only';
  return undefined;
}

/** Normalizes a raw name seed (spec 3.3). Does not guarantee the result is a valid name. */
export function inferDatasetName(seed: string): string {
  return seed
    .toLowerCase()
    .replace(/[^a-z0-9.]+/g, '-') // each run of other characters becomes one hyphen
    .replace(/-{2,}/g, '-')
    .replace(/^[-.]+|[-.]+$/g, '')
    .slice(0, ENTITY_NAME_MAX_LENGTH)
    .replace(/[-.]+$/, ''); // truncation can expose a new trailing hyphen/dot
}
