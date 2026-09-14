// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useHost } from '@agent-hardener/host';
import type { PluginSdk } from '@agent-hardener/types';

/**
 * Studio's platform SDK, handed in on the host rather than imported: the hooks
 * must run on Studio's authenticated axios and its single QueryClient, so a
 * plugin never bundles or configures the SDK itself.
 */
export const usePlatformSdk = (): PluginSdk['platform'] => useHost().sdk.platform;
