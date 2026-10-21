// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getSourceDisplay } from '@studio/components/dataViews/JobsDataView/sourceDisplay';

describe('getSourceDisplay', () => {
  it('labels a known source', () => {
    expect(getSourceDisplay('data-designer').label).toBe('Data Designer');
  });

  it.each(['custom-source', 'constructor', 'toString', '__proto__'])(
    'falls back to the raw name and an icon for %s',
    (source) => {
      const { label, icon } = getSourceDisplay(source);
      expect(label).toBe(source);
      expect(icon).toBeTruthy();
    }
  );
});
