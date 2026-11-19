// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generates an LLM-tool "capability registry" from the OpenAPI specs that drive
 * `@nemo/sdk`. For every operation (path + method) it emits a serializable
 * {@link CapabilityMeta} with a self-contained JSON Schema for its arguments.
 *
 * Output: `generated/capabilities/registry.ts` and `registry.json`.
 *
 * Run with: `pnpm gen:capabilities` (all services) or
 * `pnpm gen:capabilities platform agents` (a subset).
 *
 * The runtime in `src/capabilities/` turns this registry into discoverable,
 * invocable tools and into provider wire formats (Anthropic/OpenAI/MCP).
 */
import fs from 'fs';
import path from 'path';
import { parse as parseYaml } from 'yaml';
import { serviceConfigs } from './constants';
import { isRecord } from './utils';
import type { CapabilityMeta, HttpMethod, JsonSchema } from '../src/capabilities/types';

const HTTP_METHODS: readonly HttpMethod[] = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];
const COMPONENT_SCHEMA_PREFIX = '#/components/schemas/';

interface OpenApiSpec {
  paths?: Record<string, unknown>;
  components?: { schemas?: Record<string, unknown> };
}

interface OpenApiParameter {
  name: string;
  in: 'path' | 'query' | 'header' | 'cookie';
  required?: boolean;
  description?: string;
  schema?: JsonSchema;
}

/**
 * Recursively rewrites `#/components/schemas/X` references to local `#/$defs/X`
 * references, recording every component name reached so the caller can bundle
 * them. Returns a deep clone; the input is never mutated.
 */
const rewriteRefs = (node: unknown, needed: Set<string>): unknown => {
  if (Array.isArray(node)) return node.map((item) => rewriteRefs(item, needed));
  if (!isRecord(node)) return node;

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(node)) {
    // Rewrite both `$ref` values and discriminator `mapping` values, which are
    // both whole-string pointers into the component schemas.
    if (typeof value === 'string' && value.startsWith(COMPONENT_SCHEMA_PREFIX)) {
      const name = value.slice(COMPONENT_SCHEMA_PREFIX.length);
      needed.add(name);
      result[key] = `#/$defs/${name}`;
    } else {
      result[key] = rewriteRefs(value, needed);
    }
  }
  return result;
};

/**
 * Walks the dependency graph of referenced component schemas and returns a
 * `$defs` map containing every transitively-referenced schema (refs rewritten).
 */
const collectDefs = (
  initial: Set<string>,
  componentSchemas: Record<string, unknown>
): Record<string, unknown> => {
  const defs: Record<string, unknown> = {};
  const queue = [...initial];
  const seen = new Set<string>(initial);

  while (queue.length > 0) {
    const name = queue.shift() as string;
    const schema = componentSchemas[name];
    if (schema === undefined) {
      console.warn(`  ⚠ referenced schema "${name}" not found in components.schemas`);
      continue;
    }
    const moreNeeded = new Set<string>();
    defs[name] = rewriteRefs(schema, moreNeeded);
    for (const next of moreNeeded) {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push(next);
      }
    }
  }
  return defs;
};

/** Merges a parameter's outer `description` into its schema for the LLM. */
const paramSchema = (param: OpenApiParameter, needed: Set<string>): JsonSchema => {
  const base = isRecord(param.schema) ? (rewriteRefs(param.schema, needed) as JsonSchema) : {};
  if (param.description && base.description === undefined) {
    return { ...base, description: param.description };
  }
  return base;
};

/** Extracts the `application/json` (or first available) request body schema. */
const requestBodySchema = (
  operation: Record<string, unknown>,
  needed: Set<string>
): { schema?: JsonSchema; required: boolean } => {
  const body = operation.requestBody;
  if (!isRecord(body)) return { required: false };
  const content = isRecord(body.content) ? body.content : undefined;
  if (!content) return { required: false };
  const json = isRecord(content['application/json'])
    ? (content['application/json'] as Record<string, unknown>)
    : Object.values(content).find(isRecord);
  const schema = json && isRecord(json.schema) ? (json.schema as JsonSchema) : undefined;
  return {
    schema: schema ? (rewriteRefs(schema, needed) as JsonSchema) : undefined,
    required: body.required === true,
  };
};

interface BuildResult {
  meta: CapabilityMeta;
}

const buildCapability = (
  service: string,
  pathTemplate: string,
  method: HttpMethod,
  operation: Record<string, unknown>,
  pathLevelParams: OpenApiParameter[],
  componentSchemas: Record<string, unknown>,
  usedNames: Set<string>
): BuildResult => {
  const needed = new Set<string>();

  const opParams = Array.isArray(operation.parameters)
    ? (operation.parameters as OpenApiParameter[])
    : [];
  // Operation-level params override path-level params of the same (name, in).
  const paramKey = (p: OpenApiParameter) => `${p.in}:${p.name}`;
  const merged = new Map<string, OpenApiParameter>();
  for (const p of [...pathLevelParams, ...opParams]) {
    if (isRecord(p) && typeof p.name === 'string') merged.set(paramKey(p), p);
  }
  const parameters = [...merged.values()];

  const properties: Record<string, JsonSchema> = {};
  const required: string[] = [];
  const pathParams: string[] = [];
  const queryParams: string[] = [];

  for (const param of parameters) {
    if (param.in === 'path') {
      pathParams.push(param.name);
      properties[param.name] = paramSchema(param, needed);
      required.push(param.name);
    } else if (param.in === 'query') {
      queryParams.push(param.name);
      properties[param.name] = paramSchema(param, needed);
      if (param.required) required.push(param.name);
    }
  }

  const { schema: bodySchema, required: bodyRequired } = requestBodySchema(operation, needed);
  const hasBody = bodySchema !== undefined;
  if (hasBody) {
    properties.body = bodySchema as JsonSchema;
    if (bodyRequired) required.push('body');
  }

  const defs = collectDefs(needed, componentSchemas);

  const inputSchema: JsonSchema = {
    type: 'object',
    properties,
    ...(required.length > 0 ? { required } : {}),
    ...(Object.keys(defs).length > 0 ? { $defs: defs } : {}),
  };

  const operationId = typeof operation.operationId === 'string' ? operation.operationId : undefined;
  const baseName = operationId ?? `${method.toLowerCase()}_${pathTemplate}`.replace(/\W+/g, '_');
  let name = baseName;
  if (usedNames.has(name)) {
    name = `${service}_${baseName}`;
    let i = 2;
    while (usedNames.has(name)) name = `${service}_${baseName}_${i++}`;
  }
  usedNames.add(name);

  const tags = Array.isArray(operation.tags)
    ? (operation.tags.filter((t) => typeof t === 'string') as string[])
    : [];

  const meta: CapabilityMeta = {
    name,
    service,
    method,
    path: pathTemplate,
    summary: typeof operation.summary === 'string' ? operation.summary : undefined,
    description: typeof operation.description === 'string' ? operation.description : undefined,
    tags,
    readOnly: method === 'GET',
    pathParams,
    queryParams,
    hasBody,
    bodyRequired: hasBody && bodyRequired,
    inputSchema,
  };

  return { meta };
};

const buildServiceCapabilities = (
  service: string,
  spec: OpenApiSpec,
  usedNames: Set<string>
): CapabilityMeta[] => {
  const componentSchemas = spec.components?.schemas ?? {};
  const out: CapabilityMeta[] = [];

  for (const [pathTemplate, pathItemRaw] of Object.entries(spec.paths ?? {})) {
    if (!isRecord(pathItemRaw)) continue;
    const pathLevelParams = Array.isArray(pathItemRaw.parameters)
      ? (pathItemRaw.parameters as OpenApiParameter[])
      : [];

    for (const method of HTTP_METHODS) {
      const operation = pathItemRaw[method.toLowerCase()];
      if (!isRecord(operation)) continue;
      const { meta } = buildCapability(
        service,
        pathTemplate,
        method,
        operation,
        pathLevelParams,
        componentSchemas,
        usedNames
      );
      out.push(meta);
    }
  }
  return out;
};

const HEADER = [
  '// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
  '// SPDX-License-Identifier: Apache-2.0',
  '',
  '// Generated by orval/generate-capabilities.ts. Do not edit manually.',
  '',
].join('\n');

const main = (): void => {
  const requested = process.argv.slice(2);
  const services = requested.length > 0 ? requested : Object.keys(serviceConfigs);

  const all: CapabilityMeta[] = [];
  const usedNames = new Set<string>();

  for (const service of services) {
    const config = serviceConfigs[service as keyof typeof serviceConfigs];
    if (!config) {
      throw new Error(
        `Unknown service "${service}". Known: ${Object.keys(serviceConfigs).join(', ')}`
      );
    }
    const specPath = path.resolve(__dirname, config.url);
    const spec = parseYaml(fs.readFileSync(specPath, 'utf8')) as OpenApiSpec;
    const caps = buildServiceCapabilities(service, spec, usedNames);
    console.log(`  ${service}: ${caps.length} operations`);
    all.push(...caps);
  }

  all.sort((a, b) => a.name.localeCompare(b.name));

  const outDir = path.resolve(__dirname, '../generated/capabilities');
  fs.mkdirSync(outDir, { recursive: true });

  const ts =
    HEADER +
    `import type { CapabilityMeta } from '../../src/capabilities/types';\n\n` +
    `export const capabilities: CapabilityMeta[] = ${JSON.stringify(all, null, 2)};\n`;
  fs.writeFileSync(path.join(outDir, 'registry.ts'), ts);
  fs.writeFileSync(path.join(outDir, 'registry.json'), JSON.stringify(all, null, 2) + '\n');

  console.log(`✓ Wrote ${all.length} capabilities to ${path.relative(process.cwd(), outDir)}`);
};

main();
