// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Installed version of each shared package Studio ships, keyed by package name. */
export const STUDIO_SHARED_VERSIONS: Readonly<Record<string, string>> =
  typeof __STUDIO_SHARED_VERSIONS__ !== 'undefined' ? __STUDIO_SHARED_VERSIONS__ : {};
