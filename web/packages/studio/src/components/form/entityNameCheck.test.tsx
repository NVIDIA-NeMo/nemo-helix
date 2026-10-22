// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { nameFieldSlots, nameTakenMessage } from '@studio/components/form/entityNameCheck';
import { render, screen } from '@testing-library/react';

describe('nameTakenMessage', () => {
  it.each([
    ['fileset', 'A fileset named x already exists'],
    ['configuration', 'A configuration named x already exists'],
    ['experiment', 'An experiment named x already exists'],
    ['evaluation', 'An evaluation named x already exists'],
  ])('uses the right article for %s', (entity, expected) => {
    expect(nameTakenMessage(entity, 'x')).toBe(expected);
  });

  it('names the workspace the name must be unique in', () => {
    expect(nameTakenMessage('fileset', 'x', 'default')).toBe(
      'A fileset named x already exists in the default workspace'
    );
  });
});

describe('nameFieldSlots', () => {
  const base = { entity: 'fileset', describe: 'Must be unique.' };

  it('reports a conflict as an error naming the workspace', () => {
    expect(
      nameFieldSlots({ ...base, preview: 'x', status: 'conflict', workspace: 'default' })
    ).toEqual({
      slotError: 'A fileset named x already exists in the default workspace',
      status: 'error',
    });
  });

  it('shows the description until there is a name', () => {
    expect(nameFieldSlots({ ...base, preview: '', status: undefined })).toEqual({
      slotHelp: 'Must be unique.',
    });
  });

  it('adds the workspace to the preview when given one', () => {
    const { slotHelp } = nameFieldSlots({
      ...base,
      preview: 'x',
      status: 'available',
      workspace: 'default',
    });
    render(<p>{slotHelp}</p>);

    expect(screen.getByRole('paragraph')).toHaveTextContent(
      'Your fileset will be created as x in the default workspace'
    );
  });

  it('leaves the preview unscoped without a workspace', () => {
    const { slotHelp } = nameFieldSlots({ ...base, preview: 'x', status: 'available' });
    render(<p>{slotHelp}</p>);

    expect(screen.getByRole('paragraph')).toHaveTextContent(/^Your fileset will be created as x$/);
  });
});
