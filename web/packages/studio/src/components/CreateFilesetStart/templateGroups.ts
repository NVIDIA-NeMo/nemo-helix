// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  OTHER_SECTION,
  SECTION_ACCENTS,
  TEMPLATE_SECTIONS,
} from '@studio/components/CreateFilesetStart/constants';
import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';
import type { StartTemplateGroup } from '@studio/components/StartOptions/types';

/**
 * Groups the templates into the sections the page renders, in the order
 * {@link TEMPLATE_SECTIONS} names them. A template's own tag decides its section, and the
 * ones whose tag names no section collect under "Other" rather than each earning a
 * heading of one.
 */
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
