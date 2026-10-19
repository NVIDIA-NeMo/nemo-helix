// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Select options derived from a generated enum, so a value the backend adds reaches the
 * dropdown on its own — a hand-written table drops it silently instead.
 *
 * Labels are the API value as-is. Use `overrides` only where that would mislead.
 */
export const selectItems = <T extends Record<string, string>>(
  values: T,
  overrides: Partial<Record<T[keyof T], string>> = {}
): { value: T[keyof T]; children: string }[] =>
  Object.values(values).map((value) => {
    const key = value as T[keyof T];
    return { value: key, children: overrides[key] ?? value };
  });
