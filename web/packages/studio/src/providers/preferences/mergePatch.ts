// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonObject } from '@studio/providers/preferences/types';

export const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Applies an RFC 7396 JSON merge patch, the same way the server does. */
export const mergePatch = (target: unknown, patch: unknown): unknown => {
  if (!isJsonObject(patch)) return patch;
  const result: JsonObject = isJsonObject(target) ? { ...target } : {};
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete result[key];
    else result[key] = mergePatch(result[key], value);
  }
  return result;
};

/** Combines two pending patches into one; later values win and nulls are kept as deletes. */
export const composePatches = (first: JsonObject, second: JsonObject): JsonObject => {
  const result: JsonObject = { ...first };
  for (const [key, value] of Object.entries(second)) {
    const previous = result[key];
    result[key] =
      isJsonObject(previous) && isJsonObject(value) ? composePatches(previous, value) : value;
  }
  return result;
};
