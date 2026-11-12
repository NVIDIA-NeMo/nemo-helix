// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Name seed for an Upload selection (spec 3.3), from the relative upload paths: the common
 * top-level folder, else the first loose filename with its final extension removed.
 */
export function uploadNameSeed(paths: readonly string[]): string {
  const folders = new Set(paths.filter((p) => p.includes('/')).map((p) => p.split('/')[0]));
  const [folder] = folders;
  if (folders.size === 1) return folder;

  const loose = paths.find((p) => !p.includes('/'));
  // A leading dot is part of the name (`.env`), not an extension.
  return loose?.replace(/(.)\.[^.]*$/, '$1') ?? '';
}
