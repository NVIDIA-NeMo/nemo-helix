// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Narrows an unknown value to a string, else undefined. */
export const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

/** Narrows an unknown value to a non-empty string, else undefined. */
export const asNonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value !== '' ? value : undefined;
