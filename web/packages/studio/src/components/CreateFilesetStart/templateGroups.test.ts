// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OTHER_SECTION, SECTION_ORDER } from '@studio/components/CreateFilesetStart/constants';
import { buildTemplateGroups } from '@studio/components/CreateFilesetStart/templateGroups';
import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';

describe('buildTemplateGroups', () => {
  it('puts a template in the section its tag names', () => {
    const groups = buildTemplateGroups();
    const evaluation = groups.find((group) => group.title === 'Evaluation');

    const expected = FILESET_TEMPLATES.filter((t) => t.tag.label === 'Evaluation').map(
      (t) => t.title
    );
    expect(evaluation?.templates.map((t) => t.name)).toEqual(expected);
  });

  it('collects the tags that name no section into one', () => {
    const groups = buildTemplateGroups();
    const other = groups.find((group) => group.title === OTHER_SECTION);

    // A heading over a single card is padding, so these share one rather than each
    // getting their own.
    const named: readonly string[] = SECTION_ORDER;
    const expected = FILESET_TEMPLATES.filter((t) => !named.includes(t.tag.label)).map(
      (t) => t.title
    );
    expect(other?.templates.map((t) => t.name)).toEqual(expected);
  });

  it('accounts for every template exactly once', () => {
    const placed = buildTemplateGroups().flatMap((group) => group.templates.map((t) => t.id));

    expect(new Set(placed).size).toBe(placed.length);
    expect(placed.length).toBe(FILESET_TEMPLATES.length);
  });

  it('gives each section its own accent', () => {
    const accents = buildTemplateGroups().map((group) => group.accent);

    expect(accents.every(Boolean)).toBe(true);
    expect(new Set(accents).size).toBe(accents.length);
  });

  it('drops a section nothing landed in', () => {
    // Sections are declared up front, so an empty one must not render a bare heading.
    expect(buildTemplateGroups().every((group) => group.templates.length > 0)).toBe(true);
  });
});
