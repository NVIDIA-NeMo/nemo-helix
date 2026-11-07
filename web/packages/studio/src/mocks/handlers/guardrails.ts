// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { GuardrailCheckRequest, GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import {
  GUARDRAIL_CHECKS_ENTITY_TYPE,
  type GuardrailCheckData,
  type GuardrailCheckEntity,
} from '@studio/api/guardrail-checks/types';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

/** Seed fixtures. Handlers serve mutable clones of these; call `resetGuardrailMocks` per test. */
export const mockGuardrailConfigs: GuardrailConfig[] = [
  {
    id: 'cfg-1',
    entity_id: 'cfg-1',
    parent: 'ws-default',
    db_version: 1,
    name: 'pii-filter',
    workspace: 'default',
    description: 'Blocks PII in user inputs and outputs',
    created_at: '2026-04-12T10:00:00.000Z',
    created_by: 'user@example.com',
    updated_at: '2026-04-12T10:00:00.000Z',
    updated_by: 'user@example.com',
    data: {
      models: [
        { type: 'main', engine: 'openai', model: 'gpt-4' },
        { type: 'embeddings', engine: 'openai', model: 'text-embedding-ada-002' },
      ],
      rails: {
        input: { flows: ['check pii', 'check toxicity'] },
        output: { flows: ['mask pii output', 'check output facts'] },
      },
    },
  },
  {
    id: 'cfg-2',
    entity_id: 'cfg-2',
    parent: 'ws-default',
    db_version: 1,
    name: 'toxicity-guard',
    workspace: 'default',
    description: 'Detects and blocks toxic language',
    created_at: '2026-04-11T10:00:00.000Z',
    created_by: 'user@example.com',
    updated_at: '2026-04-11T10:00:00.000Z',
    updated_by: 'user@example.com',
    data: {
      models: [{ type: 'main', engine: 'openai', model: 'gpt-4' }],
      rails: {
        input: { flows: ['check toxicity'] },
        output: { flows: ['filter toxic output'] },
      },
    },
  },
];

/** Guardrail checks are entity-store children of a config; these hang off `cfg-1` (pii-filter). */
const mockGuardrailChecks: GuardrailCheckEntity[] = [
  {
    entity_type: GUARDRAIL_CHECKS_ENTITY_TYPE,
    id: 'chk-1',
    parent: 'cfg-1',
    db_version: 1,
    name: 'leaks-ssn',
    workspace: 'default',
    created_at: '2026-04-12T11:00:00.000Z',
    created_by: 'user@example.com',
    updated_at: '2026-04-12T11:00:00.000Z',
    updated_by: 'user@example.com',
    data: {
      messages: [{ role: 'user', content: 'My SSN is 123-45-6789' }],
      runs: [
        {
          run_at: '2026-04-12T11:05:00.000Z',
          status: 'blocked',
          rails_status: { 'check pii': { status: 'blocked' } },
          config_version: 1,
        },
      ],
    },
  },
  {
    entity_type: GUARDRAIL_CHECKS_ENTITY_TYPE,
    id: 'chk-2',
    parent: 'cfg-1',
    db_version: 1,
    name: 'benign-greeting',
    workspace: 'default',
    created_at: '2026-04-12T11:00:00.000Z',
    created_by: 'user@example.com',
    updated_at: '2026-04-12T11:00:00.000Z',
    updated_by: 'user@example.com',
    data: {
      messages: [{ role: 'user', content: 'Hello there' }],
      runs: [],
    },
  },
];

// Stateful on purpose: the run flow is a read-modify-write guarded by `expected_db_version`,
// and a version-blind stub cannot tell a fresh client from one on a stale snapshot.

let guardrailConfigs: GuardrailConfig[] = [];
let guardrailChecks: GuardrailCheckEntity[] = [];
let createdCheckCount = 0;

/** Every `POST /checks` body seen since the last reset, for asserting what a run actually sent. */
export const recordedCheckRequests: GuardrailCheckRequest[] = [];

/** Restore seed fixtures and clear recorded traffic. Call in `beforeEach`. */
export const resetGuardrailMocks = (): void => {
  guardrailConfigs = structuredClone(mockGuardrailConfigs);
  guardrailChecks = structuredClone(mockGuardrailChecks);
  createdCheckCount = 0;
  recordedCheckRequests.length = 0;
};

resetGuardrailMocks();

/** Current server-side state of a check, for asserting persisted runs and versions. */
export const getMockGuardrailCheck = (name: string): GuardrailCheckEntity | undefined =>
  guardrailChecks.find((check) => check.name === name);

/** Append a check, defaulting to exactly what "Add Another Test" persists. */
export const seedMockGuardrailCheck = (
  name: string,
  overrides: Partial<GuardrailCheckEntity> = {}
): GuardrailCheckEntity => {
  const entity: GuardrailCheckEntity = {
    entity_type: GUARDRAIL_CHECKS_ENTITY_TYPE,
    id: `chk-seed-${name}`,
    parent: 'cfg-1',
    db_version: 1,
    name,
    workspace: 'default',
    created_at: '2026-04-12T12:00:00.000Z',
    created_by: 'user@example.com',
    updated_at: '2026-04-12T12:00:00.000Z',
    updated_by: 'user@example.com',
    data: { messages: [{ role: 'user', content: '' }], runs: [] },
    ...overrides,
  };
  guardrailChecks.push(entity);
  return entity;
};

const page = <T>(data: T[]) => ({
  data,
  pagination: {
    page: 1,
    page_size: 1000,
    current_page_size: data.length,
    total_pages: 1,
    total_results: data.length,
  },
});

/** The entity-store envelope for a guardrail config: rails config nests under `data.data`. */
const toConfigEntity = (config: GuardrailConfig) => ({
  ...config,
  entity_type: 'guardrail_config',
  data: { description: config.description, data: config.data },
});

export const guardrailsHandlers = [
  // --- guardrail_checks entities -------------------------------------------------------
  http.get(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${GUARDRAIL_CHECKS_ENTITY_TYPE}`,
    ({ request }) => {
      const filter = new URL(request.url).searchParams.get('filter');
      const parent = filter ? (JSON.parse(filter) as { parent?: string }).parent : undefined;
      return HttpResponse.json(
        page(parent ? guardrailChecks.filter((check) => check.parent === parent) : guardrailChecks)
      );
    }
  ),

  http.post(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${GUARDRAIL_CHECKS_ENTITY_TYPE}`,
    async ({ params, request }) => {
      const body = (await request.json()) as {
        name?: string;
        parent: string;
        data: GuardrailCheckData;
      };
      createdCheckCount += 1;
      const entity: GuardrailCheckEntity = {
        entity_type: GUARDRAIL_CHECKS_ENTITY_TYPE,
        id: `chk-new-${createdCheckCount}`,
        parent: body.parent,
        db_version: 1,
        name: body.name ?? `generated-check-${createdCheckCount}`,
        workspace: String(params.workspace),
        created_at: '2026-04-12T12:00:00.000Z',
        created_by: 'user@example.com',
        updated_at: '2026-04-12T12:00:00.000Z',
        updated_by: 'user@example.com',
        data: body.data,
      };
      guardrailChecks.push(entity);
      return HttpResponse.json(entity, { status: 201 });
    }
  ),

  // By-name lookup, scoped by `parent`; used to re-read after a version conflict.
  http.get(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${GUARDRAIL_CHECKS_ENTITY_TYPE}/:name`,
    ({ params, request }) => {
      const parent = new URL(request.url).searchParams.get('parent');
      const check = guardrailChecks.find(
        (candidate) =>
          candidate.name === params.name && (parent === null || candidate.parent === parent)
      );
      if (!check) return new HttpResponse(null, { status: 404 });
      return HttpResponse.json(check);
    }
  ),

  http.put(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${GUARDRAIL_CHECKS_ENTITY_TYPE}/:name`,
    async ({ params, request }) => {
      const parent = new URL(request.url).searchParams.get('parent');
      const check = guardrailChecks.find(
        (candidate) =>
          candidate.name === params.name && (parent === null || candidate.parent === parent)
      );
      if (!check) return new HttpResponse(null, { status: 404 });

      const body = (await request.json()) as {
        data: GuardrailCheckData;
        expected_db_version?: number;
        new_name?: string;
      };

      // Mirrors EntityVersionConflictError -> 409 in the entity-store repository.
      if (body.expected_db_version !== undefined && body.expected_db_version !== check.db_version) {
        return HttpResponse.json(
          {
            detail:
              `Entity '${String(params.name)}' of type '${GUARDRAIL_CHECKS_ENTITY_TYPE}' was modified by another request. ` +
              `Expected version ${body.expected_db_version}, but current version is ${check.db_version}. Please refetch and retry.`,
          },
          { status: 409 }
        );
      }

      check.data = body.data;
      if (body.new_name) check.name = body.new_name;
      check.db_version += 1;
      return HttpResponse.json(check);
    }
  ),

  http.delete(
    `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${GUARDRAIL_CHECKS_ENTITY_TYPE}/:name`,
    ({ params }) => {
      guardrailChecks = guardrailChecks.filter((check) => check.name !== params.name);
      return new HttpResponse(null, { status: 200 });
    }
  ),

  // Entity-by-id: the run path resolves a check's parent config here to pick a model.
  http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/entities/:id`, ({ params }) => {
    const config = guardrailConfigs.find((candidate) => candidate.id === params.id);
    if (config) return HttpResponse.json(toConfigEntity(config));
    const check = guardrailChecks.find((candidate) => candidate.id === params.id);
    if (check) return HttpResponse.json(check);
    return new HttpResponse(null, { status: 404 });
  }),

  // --- guardrails service ---------------------------------------------------------------
  http.post(
    `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/checks`,
    async ({ request }) => {
      const body = (await request.json()) as GuardrailCheckRequest;
      recordedCheckRequests.push(body);
      const blocked = body.messages.some(
        (message) =>
          typeof message.content === 'string' && /\d{3}-\d{2}-\d{4}/.test(message.content)
      );
      // Mirrors the service's two addressing modes: a whole RailsConfig runs the rails it
      // declares, an id runs the saved config's. Without this a draft that changes its
      // rails would be indistinguishable from one that didn't.
      const inline = body.guardrails?.config;
      const inlineRail = typeof inline === 'object' ? inline.rails?.input?.flows?.[0] : undefined;
      const railName = inlineRail ?? 'check pii';
      return HttpResponse.json({
        status: blocked ? 'blocked' : 'success',
        rails_status: { [railName]: { status: blocked ? 'blocked' : 'success' } },
        // The real service hardcodes this to null; the echo is a mock-only convenience.
        guardrails_data: { config_ids: body.guardrails?.config_ids },
      });
    }
  ),

  // --- guardrail configs ----------------------------------------------------------------
  http.get(`${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`, () =>
    HttpResponse.json({
      data: guardrailConfigs,
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: guardrailConfigs.length,
        total_pages: 1,
        total_results: guardrailConfigs.length,
      },
    })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs/:name`,
    ({ params }) => {
      const config = guardrailConfigs.find((c) => c.name === params.name);
      if (!config) return new HttpResponse(null, { status: 404 });
      return HttpResponse.json(config);
    }
  ),
  http.patch(
    `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs/:name`,
    async ({ params, request }) => {
      const config = guardrailConfigs.find((c) => c.name === params.name);
      if (!config) return new HttpResponse(null, { status: 404 });
      const body = (await request.json()) as Partial<GuardrailConfig>;
      Object.assign(config, body);
      return HttpResponse.json(config);
    }
  ),
  http.post(
    `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`,
    async ({ params, request }) => {
      const input = (await request.json()) as { name: string; description?: string };
      const config: GuardrailConfig = {
        id: `cfg-${Date.now()}`,
        entity_id: `cfg-${Date.now()}`,
        parent: `ws-${params.workspace}`,
        db_version: 1,
        name: input.name,
        workspace: params.workspace as string,
        description: input.description,
        created_at: new Date().toISOString(),
        created_by: 'user@example.com',
        updated_at: new Date().toISOString(),
        updated_by: 'user@example.com',
        data: undefined,
      };
      mockGuardrailConfigs.push(config);
      return HttpResponse.json(config, { status: 201 });
    }
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),
];
