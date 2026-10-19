// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';

const MODEL_ENTITY_NAME_PATTERN = /^[a-z][a-z0-9]*(-[a-z0-9]+)*$/;

describe('FILESET_TEMPLATES', () => {
  const templateModels = FILESET_TEMPLATES.flatMap((template) =>
    (template.models ?? []).flatMap((spec) =>
      spec.model ? [{ template: template.id, model: spec.model }] : []
    )
  );

  it.each(templateModels)('$template names model $model by its entity name', ({ model }) => {
    expect(model).toMatch(MODEL_ENTITY_NAME_PATTERN);
  });
});
