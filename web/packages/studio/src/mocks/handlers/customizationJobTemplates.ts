// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE,
  type CustomizationJobTemplateData,
  type CustomizationJobTemplateEntity,
} from '@studio/api/customization-job-templates/types';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { FORM_DEFAULTS } from '@studio/util/forms/customization';
import { http, HttpResponse } from 'msw';

const TEMPLATES_URL = `${PLATFORM_BASE_URL}/apis/entities/v2/workspaces/:workspace/entities/${CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE}`;

const entity = (
  overrides: Partial<CustomizationJobTemplateEntity> &
    Pick<CustomizationJobTemplateEntity, 'id' | 'name' | 'data'>
): CustomizationJobTemplateEntity => ({
  entity_type: CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE,
  db_version: 1,
  workspace: 'default',
  created_at: '2026-04-12T12:00:00.000Z',
  created_by: 'user@example.com',
  updated_at: '2026-04-12T12:00:00.000Z',
  updated_by: 'user@example.com',
  ...overrides,
});

/** Seed fixtures. Handlers serve mutable clones; call `resetCustomizationJobTemplateMocks` per test. */
export const mockCustomizationJobTemplates: CustomizationJobTemplateEntity[] = [
  entity({
    id: 'tmpl-1',
    name: 'llama-lora-baseline',
    data: {
      description: 'LoRA baseline on the support-tickets dataset',
      backend: 'automodel',
      fields: {
        ...FORM_DEFAULTS,
        outputName: 'llama-lora-baseline',
        automodel: {
          ...FORM_DEFAULTS.automodel,
          model: 'meta/llama-3.1-8b-instruct',
          dataset: { training: 'default/support-tickets' },
        },
      },
    },
  }),
  entity({
    id: 'tmpl-2',
    name: 'unsloth-quick-iterate',
    data: {
      description: 'Single-GPU Unsloth run for fast iteration',
      backend: 'unsloth',
      fields: {
        ...FORM_DEFAULTS,
        backend: 'unsloth',
        outputName: 'unsloth-quick-iterate',
        unsloth: {
          ...FORM_DEFAULTS.unsloth,
          model: { ...FORM_DEFAULTS.unsloth.model, name: 'meta/llama-3.1-8b-instruct' },
        },
      },
    },
  }),
];

let customizationJobTemplates: CustomizationJobTemplateEntity[] = [];
let createdTemplateCount = 0;

export const resetCustomizationJobTemplateMocks = (): void => {
  customizationJobTemplates = structuredClone(mockCustomizationJobTemplates);
  createdTemplateCount = 0;
};

resetCustomizationJobTemplateMocks();

/** Current server-side state of a template, for asserting persisted writes. */
export const getMockCustomizationJobTemplate = (
  name: string
): CustomizationJobTemplateEntity | undefined =>
  customizationJobTemplates.find((template) => template.name === name);

const DEFAULT_PAGE_SIZE = 10;

/** Slices `items` the way the entity-store does, so paging is not silently ignored. */
const page = (items: CustomizationJobTemplateEntity[], url: URL) => {
  const params = url.searchParams;
  const pageNumber = Number(params.get('page')) || 1;
  const pageSize = Number(params.get('page_size')) || DEFAULT_PAGE_SIZE;
  const start = (pageNumber - 1) * pageSize;
  const rows = items.slice(start, start + pageSize);

  return {
    object: 'list',
    data: rows,
    pagination: {
      page: pageNumber,
      page_size: pageSize,
      current_page_size: rows.length,
      total_pages: Math.max(1, Math.ceil(items.length / pageSize)),
      total_results: items.length,
    },
  };
};

export const customizationJobTemplatesHandlers = [
  http.get(TEMPLATES_URL, ({ request }) => {
    const url = new URL(request.url);
    const filter = url.searchParams.get('filter');
    const backend = filter
      ? (JSON.parse(filter) as Record<string, string>)['data.backend']
      : undefined;
    return HttpResponse.json(
      page(
        backend
          ? customizationJobTemplates.filter((t) => t.data.backend === backend)
          : customizationJobTemplates,
        url
      )
    );
  }),

  http.post(TEMPLATES_URL, async ({ params, request }) => {
    const body = (await request.json()) as {
      name?: string;
      project?: string;
      data: CustomizationJobTemplateData;
    };
    createdTemplateCount += 1;
    const created = entity({
      id: `tmpl-new-${createdTemplateCount}`,
      name: body.name ?? `generated-template-${createdTemplateCount}`,
      workspace: String(params.workspace),
      data: body.data,
    });
    customizationJobTemplates.push(created);
    return HttpResponse.json(created, { status: 201 });
  }),

  http.get(`${TEMPLATES_URL}/:name`, ({ params }) => {
    const template = customizationJobTemplates.find((t) => t.name === params.name);
    if (!template) return new HttpResponse(null, { status: 404 });
    return HttpResponse.json(template);
  }),

  http.put(`${TEMPLATES_URL}/:name`, async ({ params, request }) => {
    const template = customizationJobTemplates.find((t) => t.name === params.name);
    if (!template) return new HttpResponse(null, { status: 404 });

    const body = (await request.json()) as {
      data: CustomizationJobTemplateData;
      new_name?: string;
      expected_db_version?: number;
    };
    if (
      body.expected_db_version !== undefined &&
      body.expected_db_version !== template.db_version
    ) {
      return HttpResponse.json({ detail: 'Version conflict' }, { status: 409 });
    }
    template.data = body.data;
    template.db_version += 1;
    if (body.new_name) template.name = body.new_name;
    return HttpResponse.json(template);
  }),

  http.delete(`${TEMPLATES_URL}/:name`, ({ params }) => {
    const index = customizationJobTemplates.findIndex((t) => t.name === params.name);
    if (index === -1) return new HttpResponse(null, { status: 404 });
    customizationJobTemplates.splice(index, 1);
    return HttpResponse.json({ id: 'deleted', deleted_at: '2026-04-12T12:00:00.000Z' });
  }),
];
