// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  createCustomizationJobTemplate,
  customizationJobTemplatesForBackendFilter,
  deleteCustomizationJobTemplate,
  formFieldsToTemplateData,
  getCustomizationJobTemplate,
  listCustomizationJobTemplates,
  listCustomizationJobTemplatesForBackend,
  templateToFormFields,
  updateCustomizationJobTemplate,
} from '@studio/api/customization-job-templates/customizationJobTemplates';
import {
  CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE,
  type CustomizationJobTemplateEntity,
} from '@studio/api/customization-job-templates/types';
import {
  getMockCustomizationJobTemplate,
  resetCustomizationJobTemplateMocks,
} from '@studio/mocks/handlers/customizationJobTemplates';
import { FORM_DEFAULTS, type CustomizationFormFields } from '@studio/util/forms/customization';

const WORKSPACE = 'default';

beforeEach(() => {
  resetCustomizationJobTemplateMocks();
});

describe('CRUD over the generic entity-store', () => {
  it('lists templates for a workspace', async () => {
    const page = await listCustomizationJobTemplates(WORKSPACE);
    expect(page.data.map((t) => t.name)).toEqual(['llama-lora-baseline', 'unsloth-quick-iterate']);
    expect(page.data[0].entity_type).toBe(CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE);
  });

  it('filters by backend server-side via a data.backend filter', async () => {
    const page = await listCustomizationJobTemplatesForBackend(WORKSPACE, 'unsloth');
    expect(page.data.map((t) => t.name)).toEqual(['unsloth-quick-iterate']);
  });

  it('builds a data.backend filter expression', () => {
    expect(customizationJobTemplatesForBackendFilter('automodel')).toBe(
      '{"data.backend":"automodel"}'
    );
  });

  it('fetches a single template by name', async () => {
    const template = await getCustomizationJobTemplate(WORKSPACE, 'llama-lora-baseline');
    expect(template.data.backend).toBe('automodel');
    expect(template.data.fields.automodel.model).toBe('meta/llama-3.1-8b-instruct');
  });

  it('creates a template and derives backend from the captured fields', async () => {
    const fields: CustomizationFormFields = { ...FORM_DEFAULTS, backend: 'unsloth' };
    const created = await createCustomizationJobTemplate(WORKSPACE, {
      name: 'my-template',
      data: { fields, description: 'a description' },
    });

    expect(created.name).toBe('my-template');
    expect(created.data.backend).toBe('unsloth');
    expect(getMockCustomizationJobTemplate('my-template')?.data.description).toBe('a description');
  });

  it('updates a template by name', async () => {
    const template = await getCustomizationJobTemplate(WORKSPACE, 'llama-lora-baseline');
    await updateCustomizationJobTemplate(WORKSPACE, template.name, {
      data: { ...template.data, description: 'updated' },
      expected_db_version: template.db_version,
    });
    expect(getMockCustomizationJobTemplate('llama-lora-baseline')?.data.description).toBe(
      'updated'
    );
  });

  it('surfaces a version conflict on a stale update', async () => {
    const template = await getCustomizationJobTemplate(WORKSPACE, 'llama-lora-baseline');
    await expect(
      updateCustomizationJobTemplate(WORKSPACE, template.name, {
        data: template.data,
        expected_db_version: template.db_version + 5,
      })
    ).rejects.toThrow();
  });

  it('deletes a template by name', async () => {
    await deleteCustomizationJobTemplate(WORKSPACE, 'llama-lora-baseline');
    expect(getMockCustomizationJobTemplate('llama-lora-baseline')).toBeUndefined();
  });
});

describe('formFieldsToTemplateData', () => {
  it('captures the backend and the full form state, dataset included', () => {
    const fields: CustomizationFormFields = {
      ...FORM_DEFAULTS,
      automodel: { ...FORM_DEFAULTS.automodel, dataset: { training: 'default/my-data' } },
    };
    const data = formFieldsToTemplateData(fields, 'why');

    expect(data.backend).toBe('automodel');
    expect(data.description).toBe('why');
    expect(data.fields.automodel.dataset).toEqual({ training: 'default/my-data' });
  });
});

describe('templateToFormFields', () => {
  const template = (fields: unknown): CustomizationJobTemplateEntity =>
    ({
      entity_type: CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE,
      id: 'tmpl-x',
      name: 'tmpl-x',
      workspace: WORKSPACE,
      db_version: 1,
      created_at: '2026-04-12T12:00:00.000Z',
      created_by: 'user@example.com',
      updated_at: '2026-04-12T12:00:00.000Z',
      updated_by: 'user@example.com',
      data: { backend: 'automodel', fields },
    }) as CustomizationJobTemplateEntity;

  it('restores the stored selections', () => {
    const restored = templateToFormFields(
      template({
        ...FORM_DEFAULTS,
        backend: 'unsloth',
        unsloth: {
          ...FORM_DEFAULTS.unsloth,
          model: { ...FORM_DEFAULTS.unsloth.model, name: 'a-model' },
        },
      })
    );

    expect(restored?.backend).toBe('unsloth');
    expect(restored?.unsloth.model.name).toBe('a-model');
  });

  it('regenerates outputName so applying a template twice cannot collide', () => {
    const stored = { ...FORM_DEFAULTS, outputName: 'previously-saved' };
    expect(templateToFormFields(template(stored))?.outputName).not.toBe('previously-saved');
  });

  it('returns undefined for a payload whose backend it does not understand', () => {
    // Forward-compat: a template saved by a newer Studio, naming a backend this build has
    // never heard of, must be rejected outright rather than silently coerced into an
    // automodel job the user never configured.
    expect(templateToFormFields(template({ backend: 'some-future-backend' }))).toBeUndefined();
  });

  /**
   * `backend` arrives as opaque JSON, so the guard must test membership rather than use
   * `in`, which would also accept inherited keys like "toString" and coerce a nonsense
   * payload into a real form.
   */
  it('rejects a backend that only matches an inherited object key', () => {
    expect(templateToFormFields(template({ backend: 'toString' }))).toBeUndefined();
    expect(templateToFormFields(template({ backend: 'constructor' }))).toBeUndefined();
  });

  it('restores a template saved against the rl backend', () => {
    const restored = templateToFormFields(template({ ...FORM_DEFAULTS, backend: 'rl' }));
    expect(restored?.backend).toBe('rl');
  });

  it('backfills defaults for fields missing from an older stored template', () => {
    // `data` is opaque to the entity-store, so a template written by an older Studio
    // build can omit whole sections. Those must come back as defaults, not undefined,
    // or the form's inputs flip from controlled to uncontrolled.
    const restored = templateToFormFields(template({ backend: 'automodel' }));

    expect(restored?.unsloth).toEqual(FORM_DEFAULTS.unsloth);
    expect(restored?.automodel).toEqual(FORM_DEFAULTS.automodel);
  });

  /**
   * The backfill has to reach nested sections too. A one-level merge replaces a whole
   * sub-object, so a template naming one key under `rl.training` would drop every other
   * training default and leave those inputs uncontrolled.
   */
  it('backfills defaults inside a partially stored nested section', () => {
    const restored = templateToFormFields(
      template({ backend: 'rl', rl: { training: { epochs: 7 } } })
    );

    expect(restored?.rl.training.epochs).toBe(7);
    // Everything the template did not name still comes back as its default.
    expect(restored?.rl.training).toMatchObject({
      ...FORM_DEFAULTS.rl.training,
      epochs: 7,
    });
    expect(restored?.rl.model).toBe(FORM_DEFAULTS.rl.model);
  });

  it('replaces a stored array rather than merging it with the default', () => {
    const restored = templateToFormFields(
      template({
        backend: 'automodel',
        automodel: { training: { lora: { target_modules: ['q'] } } },
      })
    );

    expect(restored?.automodel.training.lora?.target_modules).toEqual(['q']);
    // Sibling lora defaults survive the partial section.
    expect(restored?.automodel.training.lora?.alpha).toBe(
      FORM_DEFAULTS.automodel.training.lora?.alpha
    );
  });
});
