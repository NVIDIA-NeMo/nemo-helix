// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  OTHER_SECTION,
  SECTION_ACCENTS,
  TEMPLATE_SECTIONS,
} from '@studio/components/DataDesignerStart/constants';
import { FILESET_TEMPLATES } from '@studio/components/DataDesignerStart/templates';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';

/** A template's tag names its section; tags naming none collect under "Other". */
export const buildTemplateGroups = (): StartTemplateGroup[] => {
  const sections = new Map<string, StartTemplateGroup>(
    [...TEMPLATE_SECTIONS, OTHER_SECTION].map((title) => [
      title,
      { id: title.toLowerCase(), title, templates: [], accent: SECTION_ACCENTS[title] },
    ])
  );

  for (const template of FILESET_TEMPLATES) {
    const section = sections.get(template.tag.label) ?? sections.get(OTHER_SECTION);
    section?.templates.push({
      id: template.id,
      name: template.title,
      description: template.description,
      icon: template.icon,
    });
  }

  return [...sections.values()].filter((section) => section.templates.length > 0);
};
