// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { invokeCapability } from './invoke';
import { getCapability, indexByName, searchCapabilities } from './registry';
import type { Capability, CapabilityContext, CapabilityMeta, JsonSchema } from './types';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const ok = (payload: unknown) => ({ content: JSON.stringify(payload) });
const err = (message: string) => ({ content: message, isError: true });

const stringifyResult = (value: unknown): string => {
  if (value === undefined) return 'OK (no content)';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
};

const SEARCH_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    query: {
      type: 'string',
      description: 'Keywords to match against operation names, paths, summaries, and tags.',
    },
    service: {
      type: 'string',
      description: 'Optional: restrict to one service spec (e.g. platform, agents, evaluator).',
    },
    readOnly: {
      type: 'boolean',
      description: 'Optional: true for read-only (GET) operations, false for mutating ones.',
    },
    limit: { type: 'integer', description: 'Max results (default 25).', minimum: 1 },
  },
  required: ['query'],
};

const NAME_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    name: { type: 'string', description: 'The exact operation name from search_capabilities.' },
    args: {
      type: 'object',
      description:
        'Arguments object. Provide path and query parameters as top-level keys, and the request payload under "body". Call describe_capability first to get the exact schema.',
    },
  },
  required: ['name'],
};

const DESCRIBE_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    name: { type: 'string', description: 'The exact operation name from search_capabilities.' },
  },
  required: ['name'],
};

/**
 * Builds the gateway: a small, fixed set of capabilities that let an LLM
 * navigate and call the entire NeMo Helix API without holding hundreds of
 * tool definitions in context. The flow is search → describe → read/run.
 */
export const createGateway = (capabilities: readonly CapabilityMeta[]): Capability[] => {
  const index = indexByName(capabilities);
  const resolve = (name: string) => index.get(name) ?? getCapability(capabilities, name);

  const searchCapability: Capability = {
    name: 'search_capabilities',
    description:
      'Search the NeMo Helix API for operations by keyword. Returns a ranked list of operation names with their method, path, and summary. Use this first to find the right operation, then describe_capability for its arguments.',
    inputSchema: SEARCH_SCHEMA,
    readOnly: true,
    requiresConfirmation: false,
    execute: async (args) => {
      if (!isRecord(args)) return err('Invalid arguments: expected an object.');
      const query = asString(args.query);
      if (!query) return err('Invalid arguments: "query" is required.');
      const results = searchCapabilities(capabilities, query, {
        limit:
          typeof args.limit === 'number' && args.limit >= 1 ? Math.floor(args.limit) : undefined,
        service: asString(args.service),
        readOnly: typeof args.readOnly === 'boolean' ? args.readOnly : undefined,
      });
      return ok({ count: results.length, results });
    },
  };

  const describeCapability: Capability = {
    name: 'describe_capability',
    description:
      'Get the full definition of a single API operation by name: its HTTP method, path, whether it is read-only, and a JSON Schema for its arguments (path/query parameters and request body). Call this before read_capability or run_capability.',
    inputSchema: DESCRIBE_SCHEMA,
    readOnly: true,
    requiresConfirmation: false,
    execute: async (args) => {
      if (!isRecord(args)) return err('Invalid arguments: expected an object.');
      const name = asString(args.name);
      if (!name) return err('Invalid arguments: "name" is required.');
      const meta = resolve(name);
      if (!meta) return err(`Unknown operation "${name}". Use search_capabilities to find one.`);
      return ok({
        name: meta.name,
        service: meta.service,
        method: meta.method,
        path: meta.path,
        summary: meta.summary,
        description: meta.description,
        readOnly: meta.readOnly,
        inputSchema: meta.inputSchema,
      });
    },
  };

  const runOperation = async (args: unknown, ctx: CapabilityContext, mode: 'read' | 'run') => {
    if (!isRecord(args)) return err('Invalid arguments: expected an object.');
    const name = asString(args.name);
    if (!name) return err('Invalid arguments: "name" is required.');
    const meta = resolve(name);
    if (!meta) return err(`Unknown operation "${name}". Use search_capabilities to find one.`);

    if (mode === 'read' && !meta.readOnly) {
      return err(`"${name}" mutates state (${meta.method}). Use run_capability instead.`);
    }
    if (mode === 'run' && meta.readOnly) {
      return err(`"${name}" is read-only (${meta.method}). Use read_capability instead.`);
    }

    try {
      const result = await invokeCapability(meta, args.args, ctx);
      return { content: stringifyResult(result) };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return err(`${name} failed: ${message}`);
    }
  };

  const readCapability: Capability = {
    name: 'read_capability',
    description:
      'Execute a read-only (GET) API operation and return its response. Provide the operation "name" and an "args" object as described by describe_capability.',
    inputSchema: NAME_SCHEMA,
    readOnly: true,
    requiresConfirmation: false,
    execute: (args, ctx) => runOperation(args, ctx, 'read'),
  };

  const runCapability: Capability = {
    name: 'run_capability',
    description:
      'Execute a state-changing (POST/PUT/PATCH/DELETE) API operation. This creates, updates, or deletes resources, so it should be confirmed with the user first. Provide the operation "name" and an "args" object as described by describe_capability.',
    inputSchema: NAME_SCHEMA,
    readOnly: false,
    requiresConfirmation: true,
    execute: (args, ctx) => runOperation(args, ctx, 'run'),
  };

  return [searchCapability, describeCapability, readCapability, runCapability];
};
