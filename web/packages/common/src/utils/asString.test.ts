// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asNonEmptyString, asString } from '@nemo/common/src/utils/asString';

describe('asString', () => {
  it('returns strings, including empty ones', () => {
    expect(asString('safe')).toBe('safe');
    expect(asString('')).toBe('');
  });

  it('returns undefined for non-strings', () => {
    expect(asString(3)).toBeUndefined();
    expect(asString(null)).toBeUndefined();
    expect(asString(undefined)).toBeUndefined();
    expect(asString(['a'])).toBeUndefined();
  });
});

describe('asNonEmptyString', () => {
  it('returns non-empty strings, including whitespace', () => {
    expect(asNonEmptyString('safe')).toBe('safe');
    expect(asNonEmptyString(' ')).toBe(' ');
  });

  it('returns undefined for empty strings and non-strings', () => {
    expect(asNonEmptyString('')).toBeUndefined();
    expect(asNonEmptyString(3)).toBeUndefined();
    expect(asNonEmptyString(null)).toBeUndefined();
    expect(asNonEmptyString(undefined)).toBeUndefined();
  });
});
