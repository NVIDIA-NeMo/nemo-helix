// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Binds each service's generated `customFetch` (axios client with OIDC auth and
 * base-URL resolution) to a {@link Fetcher} keyed by service name. This is the
 * only file in the capabilities runtime that depends on the generated clients,
 * keeping the rest of the module decoupled and unit-testable.
 */
import { customFetch as agentsFetch } from '../../generated/fetchers/agents';
import { customFetch as dataDesignerFetch } from '../../generated/fetchers/data-designer';
import { customFetch as evalsFetch } from '../../generated/fetchers/evals';
import { customFetch as platformFetch } from '../../generated/fetchers/platform';
import { customFetch as safeSynthesizerFetch } from '../../generated/fetchers/safe-synthesizer';
import type { Fetcher } from './types';

/** Service name → generated axios fetcher. Keys match `CapabilityMeta.service`. */
export const defaultFetchers: Partial<Record<string, Fetcher>> = {
  platform: platformFetch,
  'data-designer': dataDesignerFetch,
  agents: agentsFetch,
  'safe-synthesizer': safeSynthesizerFetch,
  evals: evalsFetch,
};
