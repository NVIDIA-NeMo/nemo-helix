// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NeMo Helix API capabilities for LLM tool use.
 *
 * The generated registry (`generated/capabilities/registry.ts`) describes every
 * API operation. The gateway exposes a handful of capabilities — search,
 * describe, read, run — that let an agent navigate and invoke the whole API
 * without holding hundreds of tool definitions in context. Adapters convert
 * capabilities to Anthropic, OpenAI, or MCP wire formats.
 *
 * Quick start:
 * ```ts
 * import { nemoGateway, toAnthropicTools, defaultFetchers } from '@nemo/sdk/src/capabilities';
 *
 * const tools = toAnthropicTools(nemoGateway);                 // pass to the Messages API
 * const ctx = { fetchers: defaultFetchers, workspace: 'default' };
 * // when the model calls a tool:
 * const cap = nemoGateway.find((c) => c.name === toolUse.name)!;
 * const result = await cap.execute(toolUse.input, ctx);
 * ```
 */
import { capabilities } from '../../generated/capabilities/registry';
import { createGateway } from './gateway';

export * from './types';
export * from './registry';
export * from './invoke';
export * from './gateway';
export * from './fetchers';
export * from './adapters/anthropic';
export * from './adapters/openai';
export * from './adapters/mcp';

/** The full generated registry of API operations. */
export { capabilities };

/** The ready-to-use gateway capabilities (search/describe/read/run). */
export const nemoGateway = createGateway(capabilities);
