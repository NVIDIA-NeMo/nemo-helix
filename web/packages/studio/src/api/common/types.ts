// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface BaseQueryOptions {
  signal?: AbortSignal;
}

/**
 * @deprecated When migrating to the generated hooks from the per-tag modules under '@nemo/sdk/generated/platform/',
 * use workspace and name parameters directly instead of this interface.
 */
export interface EntityIdentifier {
  workspace?: string;
  name?: string;
}
