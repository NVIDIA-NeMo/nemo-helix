// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Framework-agnostic capability model for the NeMo Helix API.
 *
 * A "capability" is a single API operation (one path + method) described in a
 * way an LLM agent can discover and invoke through tool use. The metadata in
 * {@link CapabilityMeta} is produced by `orval/generate-capabilities.ts` from the
 * same OpenAPI specs that drive the rest of `@nemo/sdk`; the runtime in this
 * module turns that metadata into invocable tools and into the wire formats of
 * specific LLM providers (Anthropic, OpenAI) and MCP via the adapters.
 */

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/**
 * A JSON Schema fragment. OpenAPI 3.1 schemas are JSON Schema 2020-12, which is
 * also what the Anthropic/OpenAI tool `input_schema`/`parameters` fields accept,
 * so we pass these through largely unchanged. References are bundled into a local
 * `$defs` block by the generator so every schema is self-contained.
 */
export type JsonSchema = Record<string, unknown>;

/** Generated, serializable description of a single API operation. */
export interface CapabilityMeta {
  /** Stable, unique identifier used as the tool/operation name. */
  readonly name: string;
  /** Owning service spec, e.g. `platform`, `agents`, `evaluator`. */
  readonly service: string;
  readonly method: HttpMethod;
  /** Path template, e.g. `/apis/models/v2/workspaces/{workspace}/models`. */
  readonly path: string;
  readonly summary?: string;
  readonly description?: string;
  readonly tags: readonly string[];
  /** True for safe, side-effect-free operations (GET). */
  readonly readOnly: boolean;
  /** Names of `{...}` path parameters, in declaration order. */
  readonly pathParams: readonly string[];
  /** Names of query parameters. */
  readonly queryParams: readonly string[];
  readonly hasBody: boolean;
  readonly bodyRequired: boolean;
  /**
   * Self-contained JSON Schema describing the full argument object: every path
   * and query parameter as a top-level property, plus a `body` property when the
   * operation takes a request body. Referenced component schemas live under
   * `$defs`.
   */
  readonly inputSchema: JsonSchema;
}

/** Result returned by a capability's `execute`, shaped for an LLM tool message. */
export interface ToolResult {
  readonly content: string;
  readonly isError?: boolean;
}

/** Minimal request shape shared with the generated `customFetch` per service. */
export interface FetchRequest {
  readonly url: string;
  readonly method: string;
  readonly params?: Record<string, unknown>;
  readonly data?: unknown;
  readonly signal?: AbortSignal;
}

/** A function that performs an HTTP request and resolves the response body. */
export type Fetcher = <T>(request: FetchRequest) => Promise<T>;

/**
 * Execution context threaded through every capability call. Fetchers are
 * injected (rather than imported) so the core stays decoupled from the generated
 * axios clients and is trivially testable.
 */
export interface CapabilityContext {
  /** Per-service fetchers, keyed by {@link CapabilityMeta.service}. */
  readonly fetchers: Partial<Record<string, Fetcher>>;
  /** Fallback fetcher used when no service-specific one is registered. */
  readonly defaultFetcher?: Fetcher;
  /** Auto-filled into a `{workspace}` path param when the caller omits it. */
  readonly workspace?: string;
  readonly signal?: AbortSignal;
}

/**
 * A neutral, invocable tool. The gateway exposes a handful of these
 * (search/describe/read/run); adapters convert them to provider wire formats.
 */
export interface Capability {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: JsonSchema;
  readonly readOnly: boolean;
  /** Hint to the caller that this tool mutates state and should be confirmed. */
  readonly requiresConfirmation: boolean;
  execute(args: unknown, ctx: CapabilityContext): Promise<ToolResult>;
}
