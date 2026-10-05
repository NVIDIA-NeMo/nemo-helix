// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPluginIconName } from '@studio/plugins/iconMap';

describe('isPluginIconName', () => {
  it('accepts a multi-word kebab-case Lucide name', () => {
    expect(isPluginIconName('flask-conical')).toBe(true);
  });

  it('accepts a single-word Lucide name', () => {
    expect(isPluginIconName('settings')).toBe(true);
  });

  it('rejects an unknown icon name', () => {
    expect(isPluginIconName('this-icon-does-not-exist')).toBe(false);
  });

  it('rejects non-icon lucide exports like the generic Icon', () => {
    expect(isPluginIconName('icon')).toBe(false);
  });

  it('rejects an empty string', () => {
    expect(isPluginIconName('')).toBe(false);
  });

  it('rejects a name with a trailing hyphen', () => {
    expect(isPluginIconName('flask-')).toBe(false);
  });
});
