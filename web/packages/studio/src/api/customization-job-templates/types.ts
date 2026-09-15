// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EntitiesPage, Entity } from '@nemo/sdk/generated/platform/schema';
import type { CustomizationBackend } from '@studio/util/customizationBackend';
import type { CustomizationFormFields } from '@studio/util/forms/customization';

/**
 * Entity type discriminator used in the entity-store for customization job templates.
 *
 * The entity-store is schema-agnostic — `entity_type` is a free-form path segment and
 * `data` is opaque JSON validated by the client. Nothing server-side knows about this
 * type, so this constant is the single source of truth for the wire value.
 */
export const CUSTOMIZATION_JOB_TEMPLATE_ENTITY_TYPE = 'customization_job_template';

/**
 * Studio-owned payload stored in a `customization_job_template` entity's `data`.
 *
 * A template is a complete, re-runnable job spec — dataset included — captured as the
 * form state rather than a backend request body. Storing `CustomizationFormFields` (the
 * same shape `jobToFormFields` produces when cloning a job) means a template feeds
 * straight into `NewCustomizationForm`'s `initialValues` with no translation, and the
 * per-backend `formTo*Create` mappers stay the only place that knows the wire format.
 */
export type CustomizationJobTemplateData = {
  /** Optional human description of what this template is for. */
  description?: string;
  /**
   * Backend the template targets. Duplicated from `fields.backend` so the list view can
   * filter server-side via `filter={"data.backend":"automodel"}` without reading `fields`.
   */
  backend: CustomizationBackend;
  /** Full form state, including the dataset selection. */
  fields: CustomizationFormFields;
};

/** `data` accepted when creating a template; `backend` is derived from `fields`. */
export type CustomizationJobTemplateDataInput = Omit<CustomizationJobTemplateData, 'backend'> & {
  backend?: CustomizationBackend;
};

/** A `customization_job_template` entity — the entity-store envelope with a typed `data`. */
export type CustomizationJobTemplateEntity = Omit<Entity, 'data'> & {
  data: CustomizationJobTemplateData;
};

/** A page of customization job template entities. */
export type CustomizationJobTemplatesPage = Omit<EntitiesPage, 'data'> & {
  data: CustomizationJobTemplateEntity[];
};
