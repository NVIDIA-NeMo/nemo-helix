// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput, HuggingfaceStorageConfig } from '@nemo/sdk/generated/platform/schema';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { getMetadataSections } from '@studio/routes/FilesetDetailRoute/FilesetMetadataPanel/utils';
import { isValidElement } from 'react';

const huggingfaceFileset: FilesetOutput = {
  id: 'model-1',
  name: 'my-model',
  workspace: 'ws',
  description: '',
  purpose: FilesetPurpose.model,
  storage: {
    type: 'huggingface',
    repo_id: 'meta-llama/Llama-2-7b',
  } as HuggingfaceStorageConfig,
  metadata: {},
  custom_fields: {},
  project: '',
  created_at: '',
  updated_at: '',
};

const findRow = (
  sections: ReturnType<typeof getMetadataSections>,
  sectionValue: string,
  rowLabel: string
) =>
  sections
    .find((section) => section.value === sectionValue)
    ?.rows.find((row) => row.label === rowLabel);

describe('getMetadataSections', () => {
  // README frontmatter is user-supplied. `license_link` must be gated against
  // `javascript:` / `data:` so a malicious model card can't inject an
  // executable URL into the rendered <Anchor>.
  describe('license_link safety', () => {
    it('renders license as plain text when license_link uses an unsafe scheme', () => {
      const sections = getMetadataSections(
        huggingfaceFileset,
        {
          license: 'MIT',
          license_link: 'javascript:alert(1)',
        },
        []
      );
      const license = findRow(sections, 'details', 'License');
      expect(license?.value).toBe('MIT');
    });

    it('renders license as an anchor when license_link is https', () => {
      const sections = getMetadataSections(
        huggingfaceFileset,
        {
          license: 'MIT',
          license_link: 'https://opensource.org/license/mit',
        },
        []
      );
      const license = findRow(sections, 'details', 'License');
      expect(isValidElement(license?.value)).toBe(true);
    });
  });

  it('always includes a Source section with a Storage row', () => {
    const sections = getMetadataSections(huggingfaceFileset, undefined, []);
    expect(findRow(sections, 'source', 'Storage')?.value).toBe('Hugging Face');
  });

  it('omits the Details section when readme metadata is empty', () => {
    const sections = getMetadataSections(huggingfaceFileset, {}, []);
    expect(sections.find((section) => section.value === 'details')).toBeUndefined();
  });

  describe('model entity sections', () => {
    it('adds a model entity section when entities are linked to the fileset', () => {
      const entity = {
        id: 'ent-1',
        name: 'my-model-entity',
        workspace: 'ws',
        created_at: '',
        updated_at: '',
        base_model: 'meta-llama/Llama-2-7b',
        model_providers: [],
      };
      const sections = getMetadataSections(huggingfaceFileset, undefined, [entity]);
      const entitySection = sections.find((s) => s.value === 'model-entity-0');
      expect(entitySection).toBeDefined();
      expect(entitySection?.title).toContain('my-model-entity');
    });

    it('shows "Not deployed" when model entity has no model providers', () => {
      const entity = {
        id: 'ent-1',
        name: 'my-model-entity',
        workspace: 'ws',
        created_at: '',
        updated_at: '',
        model_providers: [],
      };
      const sections = getMetadataSections(huggingfaceFileset, undefined, [entity]);
      const deploymentRow = findRow(sections, 'model-entity-0', 'Deployment');
      expect(deploymentRow?.value).toBe('Not deployed');
    });

    it('shows a link when model entity has model providers', () => {
      const entity = {
        id: 'ent-1',
        name: 'my-model-entity',
        workspace: 'ws',
        created_at: '',
        updated_at: '',
        model_providers: ['ws/my-provider'],
      };
      const sections = getMetadataSections(huggingfaceFileset, undefined, [entity]);
      const deploymentRow = findRow(sections, 'model-entity-0', 'Deployment');
      expect(isValidElement(deploymentRow?.value)).toBe(true);
    });

    it('omits model entity sections when entities array is empty', () => {
      const sections = getMetadataSections(huggingfaceFileset, undefined, []);
      expect(sections.filter((s) => s.value.startsWith('model-entity-'))).toHaveLength(0);
    });
  });
});
